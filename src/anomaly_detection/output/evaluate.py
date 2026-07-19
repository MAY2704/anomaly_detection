"""Anomaly scoring, threshold selection, and metrics.

Kept free of TensorFlow imports so scoring logic can be tested without the
heavy dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Detection metrics for one thresholded run.

    Attributes:
        threshold: Score above which a window is flagged.
        precision: Of the flagged windows, the fraction truly anomalous.
        recall: Of the truly anomalous windows, the fraction flagged.
        f1: Harmonic mean of precision and recall.
        average_precision: Area under the precision-recall curve. Threshold
            free, and the headline number for rare-event detection.
        roc_auc: Area under the ROC curve. Optimistic under heavy imbalance;
            reported for comparison only.
        true_negatives: Count of correctly unflagged normal windows.
        false_positives: Count of normal windows wrongly flagged.
        false_negatives: Count of anomalies missed.
        true_positives: Count of anomalies caught.
        n_samples: Total windows evaluated.
        n_anomalies: Total anomalous windows present.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    roc_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    n_samples: int
    n_anomalies: int
    warnings: list[str] = field(default_factory=list)

    @property
    def anomaly_rate(self) -> float:
        """Fraction of evaluated windows that are anomalous."""
        return self.n_anomalies / self.n_samples if self.n_samples else 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    def summary(self) -> str:
        """Return a human-readable multi-line report."""
        lines = [
            f"Samples:            {self.n_samples:,}",
            f"Anomalies:          {self.n_anomalies:,} "
            f"({self.anomaly_rate:.4%} of samples)",
            f"Threshold:          {self.threshold:.6g}",
            "",
            f"Precision:          {self.precision:.4f}",
            f"Recall:             {self.recall:.4f}",
            f"F1:                 {self.f1:.4f}",
            f"Average precision:  {self.average_precision:.4f}  (PR-AUC)",
            f"ROC AUC:            {self.roc_auc:.4f}",
            "",
            f"TP: {self.true_positives:,}   FP: {self.false_positives:,}",
            f"FN: {self.false_negatives:,}   TN: {self.true_negatives:,}",
        ]
        if self.warnings:
            lines += ["", "Warnings:", *(f"  - {w}" for w in self.warnings)]
        return "\n".join(lines)


def sequence_errors(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Mean squared error per sequence.

    Args:
        targets: ``(n, time_steps, n_features)`` ground truth.
        predictions: Model output of the same shape.

    Returns:
        ``(n,)`` array of per-sequence MSE — the anomaly score.

    Raises:
        ValueError: If the shapes disagree or are not 3-dimensional.
    """
    if targets.shape != predictions.shape:
        raise ValueError(
            f"shape mismatch: targets {targets.shape} vs "
            f"predictions {predictions.shape}"
        )
    if targets.ndim != 3:
        raise ValueError(f"expected 3-D arrays, got {targets.ndim}-D")

    return np.mean((targets - predictions) ** 2, axis=(1, 2))


def threshold_from_errors(errors: np.ndarray, percentile: float = 99.0) -> float:
    """Pick a cutoff from a distribution of held-out normal errors.

    The percentile is the expected false-positive rate on normal data: at 99,
    roughly 1% of normal windows are flagged. Anchoring to a percentile of a
    held-out distribution keeps the cutoff stable, unlike min-max normalising
    against a validation set where a single outlier defines the range.

    Args:
        errors: Per-sequence errors from held-out **normal** windows.
        percentile: Percentile in ``[0, 100]``.

    Returns:
        The error value at `percentile`.

    Raises:
        ValueError: If `errors` is empty or `percentile` is out of range.
    """
    if errors.size == 0:
        raise ValueError("cannot derive a threshold from zero errors")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    return float(np.percentile(errors, percentile))


def threshold_from_budget(scores: np.ndarray, budget: int) -> float:
    """Pick a cutoff that flags roughly `budget` of the highest-scoring items.

    An *alert budget* is the operational counterpart to a percentile: instead
    of flagging a fixed fraction of the data, it flags a fixed count — the
    number of alerts someone can actually investigate.

    This matters when positives are rare. A percentile flags a fixed
    proportion of normal data, which at a low base rate is a large absolute
    number of false positives, capping precision no matter how well the model
    ranks. See :func:`check_threshold_feasibility`.

    Unlike :func:`threshold_from_errors`, this reads the scores it is about to
    threshold rather than a held-out set. It consumes **no labels**, so it
    leaks no ground truth — but the cutoff does depend on the scored batch, so
    it is not a fixed decision rule you can carry to new data unchanged.

    Args:
        scores: Anomaly scores for the items being flagged.
        budget: How many items to flag.

    Returns:
        A threshold for which ``scores > threshold`` selects at most `budget`
        items. Ties spanning the cutoff yield fewer, never more.

    Raises:
        ValueError: If `scores` is empty or `budget` is below 1.
    """
    if scores.size == 0:
        raise ValueError("cannot derive a threshold from zero scores")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    if budget >= scores.size:
        # Flag everything: sit just below the smallest score.
        return float(np.nextafter(scores.min(), -np.inf))

    # The (budget+1)-th largest score. Everything strictly above it is flagged.
    return float(np.sort(scores)[::-1][budget])


def precision_ceiling(n_anomalies: int, n_false_positives: float) -> float:
    """Best precision achievable at a given false-positive count.

    Assumes every anomaly is caught, so this is an upper bound that no model
    quality can exceed — it depends only on counts.

    Args:
        n_anomalies: Number of true anomalies present.
        n_false_positives: Number of false positives expected or observed.

    Returns:
        The ceiling in ``[0, 1]``.
    """
    total = n_anomalies + n_false_positives
    return float(n_anomalies / total) if total > 0 else 0.0


def check_threshold_feasibility(
    n_normals: int, n_anomalies: int, percentile: float
) -> str | None:
    """Warn if a percentile threshold is incompatible with the base rate.

    A percentile flags a fixed *fraction* of normal data, which is a fixed
    *count*. When that count exceeds the number of anomalies present, most
    flags are wrong before the model does anything — precision is capped by
    arithmetic, not by model quality.

    Args:
        n_normals: Number of normal items to be scored.
        n_anomalies: Number of anomalies present.
        percentile: The percentile threshold in use.

    Returns:
        An explanatory message, or ``None`` if the threshold is workable.
    """
    if n_anomalies <= 0:
        return None

    expected_fp = n_normals * (100.0 - percentile) / 100.0

    # Tolerance, so the percentile this function suggests does not re-trigger
    # the warning: at exact parity the arithmetic lands a hair above.
    if expected_fp <= n_anomalies * (1.0 + 1e-9):
        return None

    ceiling = precision_ceiling(n_anomalies, expected_fp)
    suggested = 100.0 * (1.0 - n_anomalies / n_normals) if n_normals else percentile

    return (
        f"p{percentile:g} flags ~{expected_fp:.0f} of {n_normals:,} normal items "
        f"but only {n_anomalies} anomalies exist, so precision cannot exceed "
        f"{ceiling:.3f} however well the model ranks. Raise the percentile "
        f"(~p{suggested:.3f} balances them) or set an alert budget."
    )


def evaluate_scores(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> EvaluationResult:
    """Score predictions against ground truth at a fixed threshold.

    Threshold-free metrics (average precision, ROC AUC) are computed from the
    raw scores, so they stay meaningful even when the chosen threshold is bad.

    Args:
        labels: ``(n,)`` binary ground truth.
        scores: ``(n,)`` anomaly scores, higher meaning more anomalous.
        threshold: Scores strictly above this are flagged.

    Returns:
        An :class:`EvaluationResult`, with `warnings` populated when the
        result is degenerate (e.g. no anomalies present to detect).

    Raises:
        ValueError: If the inputs are empty or of differing length.
    """
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"length mismatch: labels {labels.shape[0]} vs scores {scores.shape[0]}"
        )
    if labels.shape[0] == 0:
        raise ValueError("cannot evaluate zero samples")

    labels = labels.astype(int)
    predictions = (scores > threshold).astype(int)

    warnings: list[str] = []
    n_anomalies = int(labels.sum())

    # Ranking metrics are undefined when one class is absent.
    if n_anomalies == 0:
        warnings.append(
            "no anomalies in the evaluation set; recall, PR-AUC and ROC AUC "
            "are undefined and reported as 0.0"
        )
        average_precision = roc_auc = 0.0
    elif n_anomalies == len(labels):
        warnings.append(
            "evaluation set contains only anomalies; ranking metrics are "
            "undefined and reported as 0.0"
        )
        average_precision = roc_auc = 0.0
    else:
        average_precision = float(average_precision_score(labels, scores))
        roc_auc = float(roc_auc_score(labels, scores))

    if 0 < n_anomalies < 30:
        # Note `n_anomalous_companies` is an absolute count, not a proportion,
        # so raising `n_companies` alone makes anomalies *rarer*, not commoner.
        warnings.append(
            f"only {n_anomalies} anomalies present; metrics will be unstable "
            "across seeds. Raise n_anomalous_companies or anomaly_probability "
            "(raising n_companies alone makes anomalies rarer, not more common)."
        )

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    # Distinguish "the model ranks badly" from "the cutoff admits more false
    # positives than there are anomalies to find" — the latter caps precision
    # regardless of ranking quality, and is the usual culprit at a low base
    # rate.
    if n_anomalies > 0 and fp > n_anomalies:
        warnings.append(
            f"{fp} false positives against {n_anomalies} anomalies caps precision "
            f"at {precision_ceiling(n_anomalies, int(fp)):.3f}. Compare "
            f"average_precision (threshold-free) to see whether the ranking or "
            f"the threshold is the problem; an alert budget bounds the flag count "
            f"directly."
        )

    return EvaluationResult(
        threshold=float(threshold),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        average_precision=average_precision,
        roc_auc=roc_auc,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        n_samples=int(labels.shape[0]),
        n_anomalies=n_anomalies,
        warnings=warnings,
    )
