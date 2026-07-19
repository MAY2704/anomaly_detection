"""Tests for configuration validation."""

from __future__ import annotations

import dataclasses

import pytest

from anomaly_detection.config import CFG, Config


class TestDefaults:
    def test_defaults_are_valid(self):
        Config().validate()

    def test_derived_properties(self):
        config = Config(test_size=0.3, val_size=0.15)
        assert config.n_features == 3
        assert config.train_size == pytest.approx(0.55)

    def test_config_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            CFG.epochs = 10  # type: ignore[misc]

    def test_replace_produces_a_validated_copy(self):
        modified = dataclasses.replace(CFG, epochs=3)
        assert modified.epochs == 3
        assert CFG.epochs != 3

    def test_replace_rejects_invalid_values(self):
        with pytest.raises(ValueError, match="epochs"):
            dataclasses.replace(CFG, epochs=0)


class TestValidation:
    def test_window_longer_than_series_raises(self):
        with pytest.raises(ValueError, match="must exceed time_steps"):
            Config(n_months=6, time_steps=6)

    def test_splits_must_leave_training_data(self):
        with pytest.raises(ValueError, match="positive training fraction"):
            Config(test_size=0.7, val_size=0.3)

    def test_more_anomalous_than_total_companies_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            Config(n_companies=10, n_anomalous_companies=20)

    def test_alert_budget_defaults_to_unset(self):
        assert Config().alert_budget is None

    def test_valid_alert_budget_accepted(self):
        assert Config(alert_budget=50).alert_budget == 50

    @pytest.mark.parametrize("budget", [0, -1])
    def test_non_positive_alert_budget_raises(self, budget):
        with pytest.raises(ValueError, match="alert_budget"):
            Config(alert_budget=budget)

    def test_positive_offsets_raise(self):
        with pytest.raises(ValueError, match="must be negative"):
            Config(anomaly_offsets=(1,))

    def test_offset_beyond_series_raises(self):
        with pytest.raises(ValueError, match="outside"):
            Config(n_months=12, anomaly_offsets=(-99,))

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("n_companies", 0, "n_companies"),
            ("features", (), "features"),
            ("test_size", 0.0, "test_size"),
            ("test_size", 1.0, "test_size"),
            ("val_size", 0.0, "val_size"),
            ("threshold_percentile", 101.0, "threshold_percentile"),
            ("anomaly_probability", 1.5, "anomaly_probability"),
            ("epochs", 0, "epochs"),
            ("batch_size", 0, "batch_size"),
            ("learning_rate", 0.0, "learning_rate"),
        ],
    )
    def test_invalid_values_raise(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            Config(**{field: value})
