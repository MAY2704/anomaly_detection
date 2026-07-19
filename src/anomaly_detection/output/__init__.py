"""Output layer: scoring, metrics, and the alerts a human acts on."""

from anomaly_detection.output.alerts import (
    build_alerts,
    deduplicate_by_group,
    write_alerts,
)
from anomaly_detection.output.evaluate import (
    EvaluationResult,
    check_threshold_feasibility,
    evaluate_scores,
    precision_ceiling,
    sequence_errors,
    threshold_from_budget,
    threshold_from_errors,
)
from anomaly_detection.output.inference import (
    ScoringArtifacts,
    load_artifacts,
    save_artifacts,
    score_frame,
)

__all__ = [
    "EvaluationResult",
    "ScoringArtifacts",
    "build_alerts",
    "check_threshold_feasibility",
    "deduplicate_by_group",
    "evaluate_scores",
    "load_artifacts",
    "precision_ceiling",
    "save_artifacts",
    "score_frame",
    "sequence_errors",
    "threshold_from_budget",
    "threshold_from_errors",
    "write_alerts",
]
