"""Tests for windowing, splitting, and scaling.

Several tests here are regressions for specific bugs; those are called out in
their docstrings so a future refactor does not quietly reintroduce them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.input.simulate import ID_COLUMN, LABEL_COLUMN, MONTH_COLUMN
from anomaly_detection.process.preprocess import (
    Sequences,
    apply_scaler,
    fit_scaler,
    prepare_sequences,
    split_and_scale,
    split_by_group,
)

FEATURES = ["TURNOVER", "ASSETS", "TURNOVER_ROC"]


def make_frame(n_companies: int, n_months: int, anomaly_month: int | None = None):
    """Build a deterministic frame, optionally with an anomaly at one month."""
    rows = []
    for c in range(n_companies):
        labels = np.zeros(n_months, dtype=int)
        if anomaly_month is not None:
            labels[anomaly_month] = 1
        rows.append(
            pd.DataFrame(
                {
                    ID_COLUMN: f"CP_{c:04d}",
                    MONTH_COLUMN: pd.date_range(
                        "2022-01-01", periods=n_months, freq="ME"
                    ),
                    "TURNOVER": np.arange(n_months, dtype=float) + c,
                    "ASSETS": np.arange(n_months, dtype=float) * 2.0,
                    "TURNOVER_ROC": np.zeros(n_months),
                    LABEL_COLUMN: labels,
                }
            )
        )
    return pd.concat(rows).reset_index(drop=True)


class TestPrepareSequences:
    def test_window_count_covers_every_target_month(self):
        """Regression: the loop used to stop one window early.

        With n_months months and a window of `time_steps`, there are exactly
        `n_months - time_steps` windows. The old bound of
        `len(vals) - ts - 1` produced one fewer, making the final month
        unreachable as a prediction target.
        """
        df = make_frame(n_companies=1, n_months=12)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        assert len(seq) == 12 - 3

    def test_anomaly_on_final_month_is_retained(self):
        """Regression: anomalies on the last month were silently dropped.

        `data.generate_data` injects at offset -1, i.e. the final month. If
        windowing cannot reach it, those anomalies exist in the ground truth
        but can never be detected — deflating recall for reasons invisible
        from the metrics alone.
        """
        n_months = 12
        df = make_frame(n_companies=1, n_months=n_months, anomaly_month=n_months - 1)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        assert seq.n_anomalies == 1, "final-month anomaly must survive windowing"

    def test_target_is_input_shifted_one_step(self):
        df = make_frame(n_companies=1, n_months=8)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        # TURNOVER is 0, 1, 2, ... so the shift is directly observable.
        assert seq.inputs[0, :, 0].tolist() == [0.0, 1.0, 2.0]
        assert seq.targets[0, :, 0].tolist() == [1.0, 2.0, 3.0]

    def test_windows_never_span_two_companies(self):
        df = make_frame(n_companies=3, n_months=6)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        assert len(seq) == 3 * (6 - 3)
        # Company id is baked into TURNOVER, so a straddling window would show
        # a discontinuity within its own values.
        for window in seq.inputs:
            deltas = np.diff(window[:, 0])
            assert np.allclose(deltas, 1.0)

    def test_shapes_are_consistent(self):
        df = make_frame(n_companies=4, n_months=10)
        seq = prepare_sequences(df, FEATURES, time_steps=4)
        assert seq.inputs.shape == (4 * 6, 4, 3)
        assert seq.targets.shape == seq.inputs.shape
        assert seq.labels.shape == (4 * 6,)
        assert seq.groups.shape == (4 * 6,)

    def test_unsorted_input_is_ordered_before_windowing(self):
        df = make_frame(n_companies=2, n_months=8).sample(frac=1.0, random_state=0)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        for window in seq.inputs:
            assert np.allclose(np.diff(window[:, 0]), 1.0)

    def test_series_shorter_than_window_yields_nothing(self):
        df = make_frame(n_companies=2, n_months=3)
        seq = prepare_sequences(df, FEATURES, time_steps=5)
        assert len(seq) == 0
        assert seq.inputs.shape == (0, 5, 3)

    def test_missing_column_raises(self):
        df = make_frame(n_companies=1, n_months=8).drop(columns=["ASSETS"])
        with pytest.raises(KeyError, match="ASSETS"):
            prepare_sequences(df, FEATURES, time_steps=3)

    def test_non_positive_time_steps_raises(self):
        df = make_frame(n_companies=1, n_months=8)
        with pytest.raises(ValueError, match="time_steps"):
            prepare_sequences(df, FEATURES, time_steps=0)


class TestSplitByGroup:
    def test_no_company_appears_in_two_splits(self):
        """Regression: splitting windows at random leaked across the boundary.

        Sliding windows overlap by `time_steps - 1` months, so random
        window-level splitting places near-duplicates in both train and test.
        Splits must partition companies, not windows.
        """
        df = make_frame(n_companies=30, n_months=10)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        train, val, test = split_by_group(seq, test_size=0.3, val_size=0.2, seed=0)

        train_g, val_g, test_g = (set(s.groups) for s in (train, val, test))
        assert not train_g & test_g
        assert not train_g & val_g
        assert not val_g & test_g

    def test_splits_cover_every_window(self):
        df = make_frame(n_companies=30, n_months=10)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        train, val, test = split_by_group(seq, test_size=0.3, val_size=0.2, seed=0)
        assert len(train) + len(val) + len(test) == len(seq)

    def test_rare_anomalies_reach_every_split(self):
        """Stratifying on per-company anomaly presence spreads rare positives."""
        df = make_frame(n_companies=40, n_months=10, anomaly_month=9)
        # Only the first 12 companies keep their anomaly.
        keep = {f"CP_{c:04d}" for c in range(12)}
        df.loc[~df[ID_COLUMN].isin(keep), LABEL_COLUMN] = 0

        seq = prepare_sequences(df, FEATURES, time_steps=3)
        train, val, test = split_by_group(seq, test_size=0.3, val_size=0.2, seed=0)
        assert train.n_anomalies > 0
        assert val.n_anomalies > 0
        assert test.n_anomalies > 0

    def test_split_is_deterministic_for_a_seed(self):
        df = make_frame(n_companies=30, n_months=10)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        first = split_by_group(seq, test_size=0.3, val_size=0.2, seed=7)
        second = split_by_group(seq, test_size=0.3, val_size=0.2, seed=7)
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a.groups, b.groups)

    def test_single_class_does_not_break_stratification(self):
        """One anomalous company is too few to stratify; fall back gracefully."""
        df = make_frame(n_companies=20, n_months=10, anomaly_month=9)
        df.loc[df[ID_COLUMN] != "CP_0000", LABEL_COLUMN] = 0
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        train, val, test = split_by_group(seq, test_size=0.3, val_size=0.2, seed=0)
        assert len(train) and len(val) and len(test)

    def test_too_few_companies_raises(self):
        df = make_frame(n_companies=2, n_months=10)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        with pytest.raises(ValueError, match="at least 3 companies"):
            split_by_group(seq, test_size=0.3, val_size=0.2, seed=0)

    @pytest.mark.parametrize(
        ("test_size", "val_size"),
        # (0.7, 0.3) sums to exactly 1.0 and must be rejected despite the
        # floating-point sum landing a hair below it.
        [(0.0, 0.2), (1.0, 0.2), (0.3, 0.0), (0.6, 0.5), (0.7, 0.3)],
    )
    def test_invalid_fractions_raise(self, test_size, val_size):
        df = make_frame(n_companies=20, n_months=10)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        with pytest.raises(ValueError):
            split_by_group(seq, test_size=test_size, val_size=val_size, seed=0)


class TestScaling:
    def test_scaler_fitted_on_normal_data_only(self):
        """Regression: fitting on anomalies let them define the feature range.

        An injected spike is orders of magnitude above normal. If it sets the
        scaler's maximum, normal variation compresses toward zero and the
        anomaly lands inside [0, 1] looking unremarkable.
        """
        df = make_frame(n_companies=10, n_months=10, anomaly_month=9)
        df.loc[df[LABEL_COLUMN] == 1, "TURNOVER"] = 1e9

        seq = prepare_sequences(df, FEATURES, time_steps=3)
        scaler = fit_scaler(seq.normal_only())
        assert scaler.data_max_[0] < 1e9

    def test_anomalies_scale_outside_unit_range(self):
        """Out-of-range values must stay out of range, not be clipped in."""
        df = make_frame(n_companies=10, n_months=10, anomaly_month=9)
        df.loc[df[LABEL_COLUMN] == 1, "TURNOVER"] = 1e9

        seq = prepare_sequences(df, FEATURES, time_steps=3)
        scaled = apply_scaler(seq, fit_scaler(seq.normal_only()))
        assert scaled.targets.max() > 1.0

    def test_normal_training_data_lands_in_unit_range(self):
        df = make_frame(n_companies=10, n_months=10)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        scaled = apply_scaler(seq, fit_scaler(seq))
        assert scaled.inputs.min() >= -1e-9
        assert scaled.inputs.max() <= 1.0 + 1e-9

    def test_fit_on_empty_sequences_raises(self):
        empty = Sequences(
            inputs=np.empty((0, 3, 3)),
            targets=np.empty((0, 3, 3)),
            labels=np.empty((0,), dtype=int),
            groups=np.empty((0,), dtype=object),
        )
        with pytest.raises(ValueError, match="zero sequences"):
            fit_scaler(empty)

    def test_split_and_scale_returns_consistent_bundle(self):
        df = make_frame(n_companies=30, n_months=10, anomaly_month=9)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        splits = split_and_scale(seq, test_size=0.3, val_size=0.2, seed=0)

        assert len(splits.train) + len(splits.val) + len(splits.test) == len(seq)
        assert not set(splits.train.groups) & set(splits.test.groups)
        assert splits.scaler.n_features_in_ == 3


class TestSequences:
    def test_normal_only_filters_anomalies(self):
        df = make_frame(n_companies=5, n_months=10, anomaly_month=9)
        seq = prepare_sequences(df, FEATURES, time_steps=3)
        assert seq.n_anomalies == 5
        assert seq.normal_only().n_anomalies == 0
        assert len(seq.normal_only()) == len(seq) - 5
