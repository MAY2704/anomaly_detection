"""Windowing, splitting, and scaling.

The pipeline turns a long-format frame into supervised next-step-prediction
windows, splits them **by company** to avoid leakage, and scales features
using statistics learned from normal training data only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from anomaly_detection.config import MIN_TRAIN_FRACTION
from anomaly_detection.input.io import (
    DEFAULT_ID_COLUMN,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MONTH_COLUMN,
)


@dataclass(frozen=True)
class Sequences:
    """A set of sliding windows and their metadata.

    Attributes:
        inputs: ``(n, time_steps, n_features)`` model inputs.
        targets: ``(n, time_steps, n_features)`` next-step targets, i.e. the
            input window shifted forward one month.
        labels: ``(n,)`` ground truth for the final target month.
        groups: ``(n,)`` company id each window came from.
    """

    inputs: np.ndarray
    targets: np.ndarray
    labels: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        """Return the number of windows."""
        return int(self.inputs.shape[0])

    @property
    def n_anomalies(self) -> int:
        """Count of windows whose target month is anomalous."""
        return int(self.labels.sum())

    def normal_only(self) -> Sequences:
        """Return the subset of windows with no anomaly in the target month."""
        mask = self.labels == 0
        return Sequences(
            inputs=self.inputs[mask],
            targets=self.targets[mask],
            labels=self.labels[mask],
            groups=self.groups[mask],
        )


@dataclass(frozen=True)
class SplitData:
    """Train/validation/test sequences plus the fitted scaler."""

    train: Sequences
    val: Sequences
    test: Sequences
    scaler: MinMaxScaler


def prepare_sequences(
    df: pd.DataFrame,
    features: list[str],
    time_steps: int,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    month_column: str = DEFAULT_MONTH_COLUMN,
) -> Sequences:
    """Build next-step-prediction windows, one company at a time.

    For a window starting at month ``i``, the input spans ``[i, i+time_steps)``
    and the target spans ``[i+1, i+time_steps]`` — the same span shifted one
    month forward. The window's label is that of the final target month.

    The loop runs to ``len(vals) - time_steps`` so the last month of every
    series is reachable as a target. Stopping one short (a subtle but costly
    off-by-one) would silently drop every anomaly landing on the final month.

    Args:
        df: Long-format frame containing `features`, the id column, and the
            label column.
        features: Feature column names, in model input order.
        time_steps: Window length.
        id_column: Company identifier column.
        label_column: Ground-truth column. All-zero in unsupervised mode.
        month_column: Period column, used only for ordering when present.

    Returns:
        A :class:`Sequences` bundle. Empty (but correctly shaped) if no
        company has enough history.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If `time_steps` is not positive.
    """
    if time_steps < 1:
        raise ValueError(f"time_steps must be >= 1, got {time_steps}")

    missing = [c for c in [*features, id_column, label_column] if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[int] = []
    # `object`, not `str`: pandas groupby keys are not guaranteed to be strings
    # and the array below is object-dtype regardless.
    groups: list[object] = []

    # Sort within each company so windows follow chronological order even if
    # the caller handed us a shuffled frame.
    sort_cols = [id_column, month_column] if month_column in df.columns else [id_column]
    ordered = df.sort_values(sort_cols, kind="stable")

    for company, group in ordered.groupby(id_column, sort=True):
        vals = group[features].to_numpy(dtype=np.float64)
        group_labels = group[label_column].to_numpy()

        for i in range(len(vals) - time_steps):
            inputs.append(vals[i : i + time_steps])
            targets.append(vals[i + 1 : i + time_steps + 1])
            labels.append(int(group_labels[i + time_steps]))
            groups.append(company)

    if not inputs:
        empty = np.empty((0, time_steps, len(features)), dtype=np.float64)
        return Sequences(
            inputs=empty,
            targets=empty.copy(),
            labels=np.empty((0,), dtype=np.int64),
            groups=np.empty((0,), dtype=object),
        )

    return Sequences(
        inputs=np.asarray(inputs, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
    )


def _group_anomaly_flags(
    groups: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique groups and whether each contains any anomaly."""
    unique, inverse = np.unique(groups, return_inverse=True)
    has_anomaly = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(has_anomaly, inverse, labels.astype(bool))
    return unique, has_anomaly


def _stratify_or_none(flags: np.ndarray) -> np.ndarray | None:
    """Use stratification only when both classes can populate both sides."""
    positives = int(flags.sum())
    negatives = len(flags) - positives
    return flags if positives >= 2 and negatives >= 2 else None


def split_by_group(
    sequences: Sequences,
    *,
    test_size: float,
    val_size: float,
    seed: int = 42,
) -> tuple[Sequences, Sequences, Sequences]:
    """Split windows into train/val/test **by company**.

    Sliding windows overlap heavily, so splitting individual windows at random
    puts near-duplicate rows on both sides of the boundary and leaks test
    information into training. Splitting whole companies keeps every window
    from a given series on one side.

    Splits are stratified on whether a company contains any anomaly, so rare
    positives are spread across all three sets rather than clustering by luck.

    Args:
        sequences: Windows to split.
        test_size: Fraction of companies for the test set.
        val_size: Fraction of companies for the validation set.
        seed: Seed controlling the split.

    Returns:
        ``(train, val, test)`` sequence bundles.

    Raises:
        ValueError: If the fractions are out of range, or too few companies
            exist to fill three non-empty splits.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    # Tolerance, not an exact comparison: 0.7 + 0.3 sums to 0.9999999999999999.
    if 1.0 - test_size - val_size < MIN_TRAIN_FRACTION:
        raise ValueError(
            f"test_size + val_size ({test_size + val_size}) must leave a "
            "positive training fraction"
        )

    unique, has_anomaly = _group_anomaly_flags(sequences.groups, sequences.labels)
    if len(unique) < 3:
        raise ValueError(
            f"need at least 3 companies to build train/val/test splits, "
            f"got {len(unique)}"
        )

    trainval_groups, test_groups, trainval_flags, _ = train_test_split(
        unique,
        has_anomaly,
        test_size=test_size,
        random_state=seed,
        stratify=_stratify_or_none(has_anomaly),
    )

    # val_size is a fraction of the whole; rescale against what remains.
    relative_val = val_size / (1.0 - test_size)
    train_groups, val_groups = train_test_split(
        trainval_groups,
        test_size=relative_val,
        random_state=seed,
        stratify=_stratify_or_none(trainval_flags),
    )

    return tuple(  # type: ignore[return-value]
        _subset(sequences, g) for g in (train_groups, val_groups, test_groups)
    )


def _subset(sequences: Sequences, groups: np.ndarray) -> Sequences:
    """Select the windows belonging to `groups`."""
    mask = np.isin(sequences.groups, groups)
    return Sequences(
        inputs=sequences.inputs[mask],
        targets=sequences.targets[mask],
        labels=sequences.labels[mask],
        groups=sequences.groups[mask],
    )


def fit_scaler(sequences: Sequences) -> MinMaxScaler:
    """Fit a :class:`MinMaxScaler` on the flattened feature axis.

    Callers should pass *normal training windows only*. Fitting on data that
    includes anomalies lets extreme injected values define the feature range,
    compressing normal variation toward zero and making anomalies look
    ordinary.

    Args:
        sequences: Windows whose inputs define the scaling range.

    Returns:
        A fitted scaler.

    Raises:
        ValueError: If `sequences` is empty.
    """
    if len(sequences) == 0:
        raise ValueError("cannot fit a scaler on zero sequences")

    flat = sequences.inputs.reshape(-1, sequences.inputs.shape[-1])
    return MinMaxScaler().fit(flat)


def apply_scaler(sequences: Sequences, scaler: MinMaxScaler) -> Sequences:
    """Scale a bundle's inputs and targets with an already-fitted scaler.

    Values outside the fitted range fall outside ``[0, 1]``. That is expected
    and desirable here — an anomaly that exceeds anything seen in training
    *should* stand out rather than be clipped back into range.
    """

    def scale(array: np.ndarray) -> np.ndarray:
        if array.shape[0] == 0:
            return array
        flat = array.reshape(-1, array.shape[-1])
        scaled: np.ndarray = np.asarray(scaler.transform(flat))
        return scaled.reshape(array.shape)

    return Sequences(
        inputs=scale(sequences.inputs),
        targets=scale(sequences.targets),
        labels=sequences.labels,
        groups=sequences.groups,
    )


def split_and_scale(
    sequences: Sequences,
    *,
    test_size: float,
    val_size: float,
    seed: int = 42,
) -> SplitData:
    """Split by company, then scale using normal training windows only.

    Args:
        sequences: All windows.
        test_size: Fraction of companies for testing.
        val_size: Fraction of companies for validation.
        seed: Seed controlling the split.

    Returns:
        A :class:`SplitData` bundle of scaled splits and the fitted scaler.
    """
    train, val, test = split_by_group(
        sequences, test_size=test_size, val_size=val_size, seed=seed
    )

    scaler = fit_scaler(train.normal_only())

    return SplitData(
        train=apply_scaler(train, scaler),
        val=apply_scaler(val, scaler),
        test=apply_scaler(test, scaler),
        scaler=scaler,
    )
