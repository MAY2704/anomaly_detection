"""Tests for the Keras model builder.

Skipped when TensorFlow is unavailable so the fast suite still runs.
"""

from __future__ import annotations

import numpy as np
import pytest

keras = pytest.importorskip("keras", reason="TensorFlow/Keras not installed")

from anomaly_detection.process.model import build_lstm_predictor  # noqa: E402

pytestmark = pytest.mark.slow


class TestArchitecture:
    def test_output_shape_matches_input_shape(self):
        model = build_lstm_predictor(6, 3)
        assert model.output_shape == (None, 6, 3)

    def test_model_is_compiled(self):
        model = build_lstm_predictor(6, 3)
        assert model.optimizer is not None
        assert model.loss == "mse"

    def test_learning_rate_is_applied(self):
        model = build_lstm_predictor(6, 3, learning_rate=0.01)
        assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.01)

    def test_bottleneck_width_is_honoured(self):
        model = build_lstm_predictor(6, 3, dense_units=4)
        assert model.get_layer("bottleneck").units == 4

    def test_forward_pass_runs(self):
        model = build_lstm_predictor(4, 2)
        output = model.predict(np.zeros((5, 4, 2)), verbose=0)
        assert output.shape == (5, 4, 2)

    def test_one_training_step_reduces_nothing_catastrophically(self):
        rng = np.random.default_rng(0)
        x = rng.random((32, 4, 2))
        model = build_lstm_predictor(4, 2)
        history = model.fit(x, x, epochs=1, batch_size=16, verbose=0)
        assert np.isfinite(history.history["loss"][0])


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"time_steps": 0, "n_features": 3}, "time_steps"),
            ({"time_steps": 6, "n_features": 0}, "n_features"),
            ({"time_steps": 6, "n_features": 3, "lstm_units": 0}, "lstm_units"),
            ({"time_steps": 6, "n_features": 3, "dense_units": 0}, "dense_units"),
            ({"time_steps": 6, "n_features": 3, "learning_rate": 0}, "learning_rate"),
        ],
    )
    def test_invalid_dimensions_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            build_lstm_predictor(**kwargs)
