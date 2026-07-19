"""Persisting a trained detector and applying it to new data.

Training and scoring are separate acts: you train occasionally and score
continually. Everything needed to reproduce a scoring decision — model,
scaler, threshold, and the feature contract they assume — is saved together,
because a model applied with the wrong scaler or feature order fails silently
rather than loudly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from anomaly_detection.output.evaluate import sequence_errors
from anomaly_detection.process.preprocess import apply_scaler, prepare_sequences

if TYPE_CHECKING:
    import keras
    from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.keras"
SCALER_FILENAME = "scaler.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"


@dataclass
class ScoringArtifacts:
    """Everything required to score new data the same way training did.

    Attributes:
        model: Trained next-step predictor.
        scaler: Scaler fitted on training data.
        threshold: Score above which a window is flagged.
        features: Feature columns, in the order the model expects.
        time_steps: Window length the model was built for.
        metadata: Free-form provenance recorded at training time.
    """

    model: keras.Model
    scaler: MinMaxScaler
    threshold: float
    features: tuple[str, ...]
    time_steps: int
    metadata: dict[str, Any] | None = None


def save_artifacts(
    directory: str | Path,
    *,
    model: keras.Model,
    scaler: MinMaxScaler,
    threshold: float,
    features: tuple[str, ...],
    time_steps: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a scorable bundle to `directory`.

    Args:
        directory: Destination, created if absent.
        model: Trained model.
        scaler: Fitted scaler.
        threshold: Decision threshold.
        features: Feature columns in model order.
        time_steps: Window length.
        metadata: Extra provenance to record.

    Returns:
        The directory written to.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model.save(directory / MODEL_FILENAME)
    joblib.dump(scaler, directory / SCALER_FILENAME)

    payload = {
        "threshold": float(threshold),
        "features": list(features),
        "time_steps": int(time_steps),
        "metadata": metadata or {},
    }
    (directory / METADATA_FILENAME).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    return directory


def load_artifacts(directory: str | Path) -> ScoringArtifacts:
    """Load a bundle written by :func:`save_artifacts`.

    Args:
        directory: Directory containing the saved bundle.

    Returns:
        The loaded :class:`ScoringArtifacts`.

    Raises:
        FileNotFoundError: If the directory or any required file is missing.
    """
    import keras

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {directory}")

    for filename in (MODEL_FILENAME, SCALER_FILENAME, METADATA_FILENAME):
        if not (directory / filename).exists():
            raise FileNotFoundError(
                f"missing {filename} in {directory}; not a complete bundle"
            )

    payload = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))

    return ScoringArtifacts(
        model=keras.models.load_model(directory / MODEL_FILENAME),
        scaler=joblib.load(directory / SCALER_FILENAME),
        threshold=float(payload["threshold"]),
        features=tuple(payload["features"]),
        time_steps=int(payload["time_steps"]),
        metadata=payload.get("metadata"),
    )


def score_frame(
    artifacts: ScoringArtifacts,
    df: pd.DataFrame,
    *,
    id_column: str = "CP_ID",
    label_column: str = "IS_TRUE_ANOMALY",
    month_column: str = "MONTH",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a prepared frame with a loaded bundle.

    The frame is windowed and scaled exactly as in training, using the feature
    order recorded in the bundle rather than whatever order the caller's
    columns happen to be in.

    Args:
        artifacts: Loaded model, scaler, and feature contract.
        df: Frame already cleaned by
            :func:`anomaly_detection.input.io.prepare_input`.
        id_column: Company identifier column.
        label_column: Ground-truth column, all-zero when unlabelled.
        month_column: Period column.

    Returns:
        ``(groups, scores, labels)`` — one entry per window.

    Raises:
        ValueError: If the frame yields no windows.
        KeyError: If a feature the model expects is absent.
    """
    features = list(artifacts.features)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"input is missing features the model was trained on: {missing}")

    sequences = prepare_sequences(
        df,
        features,
        artifacts.time_steps,
        id_column=id_column,
        label_column=label_column,
        month_column=month_column,
    )

    if len(sequences) == 0:
        raise ValueError(
            f"no windows could be built; every company needs more than "
            f"{artifacts.time_steps} periods"
        )

    scaled = apply_scaler(sequences, artifacts.scaler)
    predictions = artifacts.model.predict(scaled.inputs, verbose=0)
    scores = sequence_errors(scaled.targets, predictions)

    logger.info(
        "scored %s windows across %s companies",
        f"{len(scores):,}",
        f"{len(np.unique(sequences.groups)):,}",
    )

    return sequences.groups, scores, sequences.labels
