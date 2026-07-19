"""Process layer: windowing, scaling, and model training.

`build_lstm_predictor` is exported lazily. Importing it eagerly would pull
TensorFlow into every consumer of this package — including `preprocess`, which
needs nothing more than numpy and scikit-learn. Keeping the import deferred is
what lets the test matrix run without a 600 MB dependency.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from anomaly_detection.process.preprocess import (
    Sequences,
    SplitData,
    apply_scaler,
    fit_scaler,
    prepare_sequences,
    split_and_scale,
    split_by_group,
)

if TYPE_CHECKING:
    from anomaly_detection.process.model import build_lstm_predictor

# Attribute -> module that defines it, resolved on first access.
_LAZY_EXPORTS = {"build_lstm_predictor": "anomaly_detection.process.model"}


def __getattr__(name: str) -> Any:
    """Resolve TensorFlow-backed exports on first use (PEP 562)."""
    if name in _LAZY_EXPORTS:
        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Sequences",
    "SplitData",
    "apply_scaler",
    "build_lstm_predictor",
    "fit_scaler",
    "prepare_sequences",
    "split_and_scale",
    "split_by_group",
]
