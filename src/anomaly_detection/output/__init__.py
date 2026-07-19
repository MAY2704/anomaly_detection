"""Output layer: scoring, metrics, and the alerts a human acts on.

The inference exports are lazy for the same reason as in
:mod:`anomaly_detection.process`: loading a saved model needs Keras, but
thresholds, metrics, and alert ranking do not, and should stay importable
without it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from anomaly_detection.output.inference import (
        ScoringArtifacts,
        load_artifacts,
        save_artifacts,
        score_frame,
    )

_INFERENCE = "anomaly_detection.output.inference"
_LAZY_EXPORTS = {
    "ScoringArtifacts": _INFERENCE,
    "load_artifacts": _INFERENCE,
    "save_artifacts": _INFERENCE,
    "score_frame": _INFERENCE,
}


def __getattr__(name: str) -> Any:
    """Resolve TensorFlow-backed exports on first use (PEP 562)."""
    if name in _LAZY_EXPORTS:
        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
