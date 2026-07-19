"""The TensorFlow-free modules must stay TensorFlow-free.

Regression guard. A convenience re-export in a package `__init__` once pulled
Keras into `process.preprocess` and `output.alerts`, neither of which needs it.
Nothing failed locally — TensorFlow was installed — and it only surfaced in the
CI job that deliberately omits it.

These tests run in subprocesses with `keras` and `tensorflow` blocked at import
time, so they fail the same way whether or not TensorFlow is present.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

BLOCKER = """
import sys

class Blocked(Exception):
    pass

class _Blocker:
    BANNED = ("keras", "tensorflow")

    def find_module(self, fullname, path=None):
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.BANNED:
            raise Blocked(f"{fullname} must not be imported here")
        return None

sys.meta_path.insert(0, _Blocker())
"""

TF_FREE_IMPORTS = [
    "anomaly_detection",
    "anomaly_detection.config",
    "anomaly_detection.cli",
    "anomaly_detection.input",
    "anomaly_detection.input.io",
    "anomaly_detection.input.simulate",
    "anomaly_detection.process.preprocess",
    "anomaly_detection.output.alerts",
    "anomaly_detection.output.evaluate",
    "anomaly_detection.output.inference",
    "anomaly_detection.process",
    "anomaly_detection.output",
    # Importable without TensorFlow so input validation can fail fast; `run`
    # imports the model only once the data is known to be good.
    "anomaly_detection.process.train",
]


def run_isolated(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a subprocess where keras and tensorflow cannot import."""
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("module", TF_FREE_IMPORTS)
def test_module_imports_without_tensorflow(module):
    result = run_isolated(f"import {module}")
    assert result.returncode == 0, (
        f"{module} pulled in TensorFlow/Keras at import time:\n{result.stderr}"
    )


def test_the_blocker_actually_blocks():
    """Guard the guard: a test that cannot fail proves nothing."""
    result = run_isolated("import keras")
    assert result.returncode != 0
    assert "must not be imported here" in result.stderr


def test_cli_help_works_without_tensorflow():
    """`--help` must not pay for a TensorFlow import."""
    result = run_isolated(
        """
        import sys
        from anomaly_detection.cli import build_parser
        build_parser().parse_args(['train', '--help'])
        """
    )
    # argparse exits 0 after printing help.
    assert result.returncode == 0, result.stderr
    assert "--input-csv" in result.stdout


def test_lazy_export_is_still_reachable():
    """Laziness must not break the public API, only defer it."""
    keras = pytest.importorskip("keras", reason="TensorFlow/Keras not installed")
    assert keras is not None

    from anomaly_detection.output import score_frame
    from anomaly_detection.process import build_lstm_predictor

    assert callable(build_lstm_predictor)
    assert callable(score_frame)


def test_unknown_attribute_still_raises_attribute_error():
    import anomaly_detection.output as output
    import anomaly_detection.process as process

    with pytest.raises(AttributeError, match="no attribute"):
        _ = process.does_not_exist
    with pytest.raises(AttributeError, match="no attribute"):
        _ = output.does_not_exist
