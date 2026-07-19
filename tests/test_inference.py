"""Tests for saving, loading, and applying a trained detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras", reason="TensorFlow/Keras not installed")

from anomaly_detection.input.io import prepare_input
from anomaly_detection.output.inference import (
    load_artifacts,
    save_artifacts,
    score_frame,
)
from anomaly_detection.process.model import build_lstm_predictor
from anomaly_detection.process.preprocess import (
    apply_scaler,
    fit_scaler,
    prepare_sequences,
)

pytestmark = pytest.mark.slow

FEATURES = ("TURNOVER", "ASSETS")
TIME_STEPS = 3


def make_frame(n_entities=6, n_months=10):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_entities):
        rows.append(
            pd.DataFrame(
                {
                    "CP_ID": f"E{i:02d}",
                    "MONTH": pd.date_range("2022-01-01", periods=n_months, freq="ME"),
                    "TURNOVER": rng.normal(100, 5, n_months),
                    "ASSETS": rng.normal(500, 10, n_months),
                }
            )
        )
    return pd.concat(rows).reset_index(drop=True)


@pytest.fixture
def bundle(tmp_path):
    """A saved bundle plus the frame it was fitted on."""
    df, _ = prepare_input(make_frame(), features=list(FEATURES), time_steps=TIME_STEPS)
    sequences = prepare_sequences(df, list(FEATURES), TIME_STEPS)
    scaler = fit_scaler(sequences)
    model = build_lstm_predictor(TIME_STEPS, len(FEATURES))

    # Fit briefly so the saved bundle matches what training actually produces:
    # an unfitted model has no optimizer variables and reloads with a warning.
    scaled = apply_scaler(sequences, scaler)
    model.fit(scaled.inputs, scaled.targets, epochs=1, batch_size=16, verbose=0)

    save_artifacts(
        tmp_path,
        model=model,
        scaler=scaler,
        threshold=0.25,
        features=FEATURES,
        time_steps=TIME_STEPS,
        metadata={"seed": 42},
    )
    return tmp_path, df


class TestRoundTrip:
    def test_saved_bundle_reloads_intact(self, bundle):
        directory, _ = bundle
        loaded = load_artifacts(directory)

        assert loaded.threshold == pytest.approx(0.25)
        assert loaded.features == FEATURES
        assert loaded.time_steps == TIME_STEPS
        assert loaded.metadata["seed"] == 42

    def test_reloaded_model_predicts_identically(self, bundle):
        """A bundle that scores differently after a reload is worthless."""
        directory, df = bundle
        loaded = load_artifacts(directory)

        first = score_frame(loaded, df)[1]
        second = score_frame(load_artifacts(directory), df)[1]
        assert np.allclose(first, second)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_artifacts(tmp_path / "absent")

    def test_incomplete_bundle_raises(self, bundle):
        directory, _ = bundle
        (directory / "scaler.joblib").unlink()

        with pytest.raises(FileNotFoundError, match="not a complete bundle"):
            load_artifacts(directory)


class TestScoreFrame:
    def test_returns_one_score_per_window(self, bundle):
        directory, df = bundle
        groups, scores, labels = score_frame(load_artifacts(directory), df)

        expected = 6 * (10 - TIME_STEPS)
        assert len(scores) == expected
        assert len(groups) == expected
        assert len(labels) == expected

    def test_scores_are_finite_and_non_negative(self, bundle):
        directory, df = bundle
        scores = score_frame(load_artifacts(directory), df)[1]
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)

    def test_an_injected_spike_outscores_normal_windows(self, bundle):
        """The end-to-end claim: an obvious anomaly ranks at the top."""
        directory, df = bundle
        spiked = df.copy()
        target = spiked.index[(spiked.CP_ID == "E03")][-1]
        spiked.loc[target, "TURNOVER"] *= 50

        groups, scores, _ = score_frame(load_artifacts(directory), spiked)
        assert groups[int(np.argmax(scores))] == "E03"

    def test_missing_feature_raises(self, bundle):
        directory, df = bundle
        with pytest.raises(KeyError, match="missing features"):
            score_frame(load_artifacts(directory), df.drop(columns=["ASSETS"]))

    def test_column_order_does_not_change_scores(self, bundle):
        """Feature order comes from the bundle, not the caller's columns."""
        directory, df = bundle
        loaded = load_artifacts(directory)

        reordered = df[["ASSETS", "TURNOVER", "IS_TRUE_ANOMALY", "CP_ID", "MONTH"]]
        assert np.allclose(
            score_frame(loaded, df)[1], score_frame(loaded, reordered)[1]
        )

    def test_frame_with_no_usable_history_raises(self, bundle):
        directory, _ = bundle
        short, _ = prepare_input(
            make_frame(n_entities=2, n_months=TIME_STEPS + 1),
            features=list(FEATURES),
            time_steps=TIME_STEPS,
        )
        # One window per entity exists; drop to below the window length.
        with pytest.raises((ValueError, KeyError)):
            score_frame(load_artifacts(directory), short.head(2))
