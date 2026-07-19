"""Input layer: getting data in and proving it is usable.

Two sources feed the pipeline — :mod:`~anomaly_detection.input.simulate` for
synthetic data and :mod:`~anomaly_detection.input.io` for real files. Both
produce the same long-format contract described in
:data:`~anomaly_detection.input.io.REQUIRED_COLUMNS`.
"""

from anomaly_detection.input.io import (
    DataQualityReport,
    derive_rate_of_change,
    load_table,
    prepare_input,
    validate_frame,
)
from anomaly_detection.input.simulate import generate_data

__all__ = [
    "DataQualityReport",
    "derive_rate_of_change",
    "generate_data",
    "load_table",
    "prepare_input",
    "validate_frame",
]
