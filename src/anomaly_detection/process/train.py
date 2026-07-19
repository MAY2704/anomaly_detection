"""End-to-end training pipeline: input -> process -> output."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from anomaly_detection.config import CFG, Config
from anomaly_detection.input.io import (
    DataQualityReport,
    load_and_prepare,
    prepare_input,
)
from anomaly_detection.input.simulate import generate_data
from anomaly_detection.output.alerts import build_alerts, write_alerts
from anomaly_detection.output.evaluate import (
    EvaluationResult,
    check_threshold_feasibility,
    evaluate_scores,
    sequence_errors,
    threshold_from_budget,
    threshold_from_errors,
)
from anomaly_detection.output.inference import METRICS_FILENAME, save_artifacts
from anomaly_detection.process.model import build_lstm_predictor
from anomaly_detection.process.preprocess import prepare_sequences, split_and_scale

if TYPE_CHECKING:
    import keras
    from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    """What a completed run produced.

    Attributes:
        model: The trained Keras model.
        scaler: The scaler fitted on training windows.
        threshold: Cutoff used to flag windows.
        result: Test-set metrics, or ``None`` when input was unlabelled.
        report: Data quality report for the input.
        alerts: Ranked alert table.
        output_dir: Where artifacts were written, or ``None`` if not saved.
    """

    model: keras.Model
    scaler: MinMaxScaler
    threshold: float
    result: EvaluationResult | None
    report: DataQualityReport
    alerts: pd.DataFrame | None = None
    output_dir: Path | None = None


def set_seeds(seed: int) -> None:
    """Seed Python, numpy, and TensorFlow for a reproducible run.

    Note: full determinism on GPU additionally requires
    ``TF_DETERMINISTIC_OPS=1`` and can cost significant throughput.
    """
    import tensorflow as tf

    random.seed(seed)
    # Legacy global seeding is deliberate: scikit-learn and Keras internals
    # read `np.random`'s global state, which a local Generator cannot set.
    # Our own simulation uses a seeded Generator and ignores this.
    np.random.seed(seed)  # noqa: NPY002
    tf.random.set_seed(seed)


def load_input(config: Config) -> tuple[pd.DataFrame, DataQualityReport]:
    """Obtain input data, from a file when configured or from the simulator.

    Args:
        config: Pipeline configuration.

    Returns:
        The prepared frame and its quality report.
    """
    if config.input_path:
        logger.info("Loading input from %s", config.input_path)
        return load_and_prepare(
            config.input_path,
            features=list(config.features),
            time_steps=config.time_steps,
            id_column=config.id_column,
            month_column=config.month_column,
            label_column=config.label_column,
            rate_of_change_from=config.rate_of_change_from,
            allow_gaps=config.allow_gaps,
        )

    logger.info(
        "Simulating %s companies over %s months",
        f"{config.n_companies:,}",
        config.n_months,
    )
    df = generate_data(
        config.n_companies,
        config.n_months,
        seed=config.seed,
        n_anomalous_companies=config.n_anomalous_companies,
        anomaly_offsets=config.anomaly_offsets,
        anomaly_probability=config.anomaly_probability,
    )
    return prepare_input(
        df,
        features=list(config.features),
        time_steps=config.time_steps,
        id_column=config.id_column,
        month_column=config.month_column,
        label_column=config.label_column,
        allow_gaps=config.allow_gaps,
    )


def run(config: Config = CFG, *, save: bool = True) -> TrainingArtifacts:
    """Run the full pipeline: load, window, split, train, score, report.

    With labels, the model trains on normal windows only and the run reports
    detection metrics. Without them the model trains on everything — anomalies
    are rare enough not to distort what normal looks like — and the run emits
    ranked alerts instead of metrics, since there is no ground truth to score
    against.

    Args:
        config: Pipeline configuration.
        save: Whether to persist artifacts.

    Returns:
        The trained model, scaler, threshold, and whatever output the labelling
        situation allows.

    Raises:
        ValueError: If the configuration yields an empty split.
    """
    set_seeds(config.seed)

    df, report = load_input(config)
    logger.info("Input\n%s", report.summary())

    sequences = prepare_sequences(
        df,
        list(config.features),
        config.time_steps,
        id_column=config.id_column,
        label_column=config.label_column,
        month_column=config.month_column,
    )
    logger.info(
        "Built %s windows (%s anomalous)",
        f"{len(sequences):,}",
        f"{sequences.n_anomalies:,}",
    )

    splits = split_and_scale(
        sequences,
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
    )

    # Labelled: withhold anomalies so they stay surprising to the model.
    # Unlabelled: nothing to withhold, and a low contamination rate leaves the
    # learned notion of "normal" essentially intact.
    if report.labelled:
        train_data = splits.train.normal_only()
        val_data = splits.val.normal_only()
    else:
        train_data, val_data = splits.train, splits.val
        logger.info("Unsupervised mode: training on all windows")

    if len(train_data) == 0:
        raise ValueError("no training sequences; check the split sizes")
    if len(val_data) == 0:
        raise ValueError("no validation sequences; check the split sizes")

    logger.info(
        "Split: train %s / val %s / test %s (%s anomalous)",
        f"{len(train_data):,}",
        f"{len(val_data):,}",
        f"{len(splits.test):,}",
        f"{splits.test.n_anomalies:,}",
    )

    model = build_lstm_predictor(
        config.time_steps,
        config.n_features,
        lstm_units=config.lstm_units,
        dense_units=config.dense_units,
        learning_rate=config.learning_rate,
    )

    model.fit(
        train_data.inputs,
        train_data.targets,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(val_data.inputs, val_data.targets),
        verbose=2,
    )

    val_errors = sequence_errors(
        val_data.targets, model.predict(val_data.inputs, verbose=0)
    )
    test_errors = sequence_errors(
        splits.test.targets, model.predict(splits.test.inputs, verbose=0)
    )

    threshold = _select_threshold(config, val_errors, test_errors, splits.test.labels)

    result: EvaluationResult | None = None
    if report.labelled:
        result = evaluate_scores(splits.test.labels, test_errors, threshold)
        logger.info("Evaluation\n%s", result.summary())
        for warning in result.warnings:
            logger.warning(warning)
    else:
        logger.info(
            "No labels available, so no metrics. Review the ranked alerts and "
            "tune the alert budget to match your review capacity."
        )

    alerts = build_alerts(
        splits.test.groups,
        test_errors,
        threshold=threshold,
        budget=config.alert_budget,
        dedupe=config.dedupe_alerts,
        labels=splits.test.labels if report.labelled else None,
    )
    logger.info("Top alerts\n%s", alerts.head(10).to_string(index=False))

    artifacts = TrainingArtifacts(
        model=model,
        scaler=splits.scaler,
        threshold=threshold,
        result=result,
        report=report,
        alerts=alerts,
    )

    if config.alerts_path:
        logger.info("Alerts written to %s", write_alerts(alerts, config.alerts_path))

    if save:
        artifacts.output_dir = _persist(artifacts, config)
        logger.info("Artifacts written to %s", artifacts.output_dir)

    return artifacts


def _select_threshold(
    config: Config,
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Pick a decision threshold from an alert budget or a percentile."""
    if config.alert_budget is not None:
        threshold = threshold_from_budget(test_errors, config.alert_budget)
        logger.info(
            "Threshold for an alert budget of %s: %.6g",
            f"{config.alert_budget:,}",
            threshold,
        )
        return threshold

    threshold = threshold_from_errors(val_errors, config.threshold_percentile)
    logger.info(
        "Threshold at p%g of held-out error: %.6g",
        config.threshold_percentile,
        threshold,
    )

    # Flag an unworkable percentile before the metrics land, so a poor result
    # is not mistaken for a poor model.
    infeasible = check_threshold_feasibility(
        n_normals=int((test_labels == 0).sum()),
        n_anomalies=int(test_labels.sum()),
        percentile=config.threshold_percentile,
    )
    if infeasible:
        logger.warning(infeasible)

    return threshold


def _persist(artifacts: TrainingArtifacts, config: Config) -> Path:
    """Write the scorable bundle plus a metrics report."""
    output_dir = save_artifacts(
        config.output_dir,
        model=artifacts.model,
        scaler=artifacts.scaler,
        threshold=artifacts.threshold,
        features=config.features,
        time_steps=config.time_steps,
        metadata={"seed": config.seed, "labelled": artifacts.report.labelled},
    )

    payload = {
        "config": asdict(config),
        "threshold": artifacts.threshold,
        "labelled": artifacts.report.labelled,
        "metrics": artifacts.result.to_dict() if artifacts.result else None,
    }
    (output_dir / METRICS_FILENAME).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    return output_dir


def main() -> None:
    """Entry point for ``python -m anomaly_detection.process.train``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    run(CFG)


if __name__ == "__main__":
    main()
