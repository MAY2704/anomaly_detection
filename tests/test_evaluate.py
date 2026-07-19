"""Tests for scoring, thresholding, and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.output.evaluate import (
    check_threshold_feasibility,
    evaluate_scores,
    precision_ceiling,
    sequence_errors,
    threshold_from_budget,
    threshold_from_errors,
)


class TestSequenceErrors:
    def test_perfect_prediction_scores_zero(self):
        targets = np.random.default_rng(0).normal(size=(10, 4, 3))
        assert np.allclose(sequence_errors(targets, targets.copy()), 0.0)

    def test_error_is_mean_over_steps_and_features(self):
        targets = np.zeros((1, 2, 2))
        predictions = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        assert sequence_errors(targets, predictions)[0] == pytest.approx(1.0)

    def test_one_error_per_sequence(self):
        targets = np.zeros((7, 4, 3))
        assert sequence_errors(targets, np.ones((7, 4, 3))).shape == (7,)

    def test_larger_deviation_scores_higher(self):
        targets = np.zeros((2, 3, 2))
        predictions = np.stack([np.full((3, 2), 0.1), np.full((3, 2), 5.0)])
        errors = sequence_errors(targets, predictions)
        assert errors[1] > errors[0]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            sequence_errors(np.zeros((4, 3, 2)), np.zeros((4, 3, 3)))

    def test_wrong_dimensionality_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            sequence_errors(np.zeros((4, 3)), np.zeros((4, 3)))


class TestThreshold:
    def test_percentile_is_respected(self):
        errors = np.arange(101, dtype=float)
        assert threshold_from_errors(errors, 95.0) == pytest.approx(95.0)

    def test_higher_percentile_gives_higher_threshold(self):
        errors = np.random.default_rng(0).exponential(size=1000)
        assert threshold_from_errors(errors, 99.0) > threshold_from_errors(errors, 90.0)

    def test_threshold_resists_a_single_outlier(self):
        """Regression: min-max normalising let one sample set the cutoff.

        The previous approach fitted a MinMaxScaler to validation error and
        thresholded at 0.5, i.e. halfway between the min and max. A single
        extreme validation sample therefore moved the decision boundary
        arbitrarily. A percentile barely notices it.
        """
        errors = np.random.default_rng(0).normal(1.0, 0.1, size=1000)
        baseline = threshold_from_errors(errors, 99.0)

        contaminated = np.append(errors, 1e6)
        assert threshold_from_errors(contaminated, 99.0) == pytest.approx(
            baseline, rel=0.1
        )

    def test_empty_errors_raise(self):
        with pytest.raises(ValueError, match="zero errors"):
            threshold_from_errors(np.array([]), 99.0)

    @pytest.mark.parametrize("percentile", [-1.0, 101.0])
    def test_out_of_range_percentile_raises(self, percentile):
        with pytest.raises(ValueError, match="percentile"):
            threshold_from_errors(np.arange(10, dtype=float), percentile)


class TestThresholdFromBudget:
    def test_flags_exactly_the_budget(self):
        scores = np.arange(100, dtype=float)
        threshold = threshold_from_budget(scores, budget=10)
        assert int((scores > threshold).sum()) == 10

    @pytest.mark.parametrize("budget", [1, 5, 27, 99])
    def test_budget_is_honoured_across_sizes(self, budget):
        scores = np.random.default_rng(0).normal(size=100)
        threshold = threshold_from_budget(scores, budget)
        assert int((scores > threshold).sum()) == budget

    def test_budget_bounds_false_positives_where_a_percentile_does_not(self):
        """The reason this exists: a percentile scales with the data volume.

        At a 1-in-2250 base rate, p99.9 admits ~27 false positives per 27,000
        windows, capping precision near 0.31 regardless of ranking quality. A
        budget fixes the count instead, so precision stops depending on how
        much normal data happens to be scored.
        """
        rng = np.random.default_rng(0)
        scores = np.concatenate([rng.normal(0, 1, 27_000), rng.normal(20, 1, 12)])
        labels = np.array([0] * 27_000 + [1] * 12)

        by_percentile = evaluate_scores(
            labels, scores, threshold_from_errors(scores[:27_000], 99.9)
        )
        by_budget = evaluate_scores(labels, scores, threshold_from_budget(scores, 12))

        assert by_percentile.false_positives > 20
        assert by_budget.false_positives == 0
        assert by_budget.precision > by_percentile.precision

    def test_ties_never_exceed_the_budget(self):
        scores = np.ones(50)
        threshold = threshold_from_budget(scores, budget=10)
        assert int((scores > threshold).sum()) <= 10

    def test_budget_larger_than_input_flags_everything(self):
        scores = np.arange(5, dtype=float)
        threshold = threshold_from_budget(scores, budget=99)
        assert int((scores > threshold).sum()) == 5

    def test_budget_selects_the_highest_scorers(self):
        scores = np.array([0.1, 9.0, 0.2, 8.0, 0.3])
        threshold = threshold_from_budget(scores, budget=2)
        assert set(np.flatnonzero(scores > threshold)) == {1, 3}

    def test_empty_scores_raise(self):
        with pytest.raises(ValueError, match="zero scores"):
            threshold_from_budget(np.array([]), budget=5)

    @pytest.mark.parametrize("budget", [0, -1])
    def test_non_positive_budget_raises(self, budget):
        with pytest.raises(ValueError, match="budget"):
            threshold_from_budget(np.arange(10, dtype=float), budget)


class TestPrecisionCeiling:
    def test_equal_counts_give_half(self):
        assert precision_ceiling(10, 10) == pytest.approx(0.5)

    def test_no_false_positives_gives_one(self):
        assert precision_ceiling(10, 0) == pytest.approx(1.0)

    def test_matches_the_measured_base_rate_case(self):
        # 12 anomalies, 27 expected false positives at a correct p99.9.
        assert precision_ceiling(12, 27) == pytest.approx(0.308, abs=1e-3)

    def test_empty_case_is_zero(self):
        assert precision_ceiling(0, 0) == 0.0


class TestThresholdFeasibility:
    def test_warns_when_percentile_swamps_the_anomalies(self):
        message = check_threshold_feasibility(
            n_normals=27_000, n_anomalies=12, percentile=99.9
        )
        assert message is not None
        assert "0.308" in message

    def test_silent_when_the_percentile_is_workable(self):
        assert (
            check_threshold_feasibility(
                n_normals=27_000, n_anomalies=12, percentile=99.99
            )
            is None
        )

    def test_silent_without_anomalies(self):
        assert (
            check_threshold_feasibility(n_normals=1000, n_anomalies=0, percentile=99.0)
            is None
        )

    def test_suggested_percentile_would_balance_the_counts(self):
        n_normals, n_anomalies = 27_000, 12
        message = check_threshold_feasibility(n_normals, n_anomalies, percentile=99.0)
        assert message is not None

        # The suggestion should itself be feasible.
        suggested = 100.0 * (1.0 - n_anomalies / n_normals)
        assert check_threshold_feasibility(n_normals, n_anomalies, suggested) is None


class TestEvaluateScores:
    def test_perfect_separation_scores_one(self):
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 5.0, 6.0])
        result = evaluate_scores(labels, scores, threshold=1.0)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.average_precision == pytest.approx(1.0)

    def test_confusion_matrix_counts(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 2.0, 0.0, 2.0])
        result = evaluate_scores(labels, scores, threshold=1.0)
        assert (result.true_negatives, result.false_positives) == (1, 1)
        assert (result.false_negatives, result.true_positives) == (1, 1)

    def test_threshold_is_exclusive(self):
        labels = np.array([0, 1])
        scores = np.array([1.0, 1.0])
        result = evaluate_scores(labels, scores, threshold=1.0)
        assert result.true_positives == 0

    def test_ranking_metrics_ignore_the_threshold(self):
        """PR-AUC reflects score quality even when the cutoff is badly chosen."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        good = evaluate_scores(labels, scores, threshold=0.5)
        bad = evaluate_scores(labels, scores, threshold=100.0)
        assert bad.recall == 0.0
        assert bad.average_precision == pytest.approx(good.average_precision)

    def test_absent_anomalies_warn_instead_of_crashing(self):
        labels = np.zeros(10, dtype=int)
        result = evaluate_scores(labels, np.random.default_rng(0).normal(size=10), 1.0)
        assert result.n_anomalies == 0
        assert result.average_precision == 0.0
        assert any("no anomalies" in w for w in result.warnings)

    def test_excess_false_positives_warn_about_the_precision_cap(self):
        labels = np.array([0] * 1000 + [1] * 2)
        scores = np.concatenate([np.zeros(1000), np.ones(2)])
        scores[:50] = 1.0  # 50 normals scoring as high as the anomalies

        result = evaluate_scores(labels, scores, threshold=0.5)
        assert result.false_positives == 50
        assert any("caps precision" in w for w in result.warnings)

    def test_no_cap_warning_when_false_positives_are_few(self):
        labels = np.array([0] * 100 + [1] * 10)
        scores = np.concatenate([np.zeros(100), np.ones(10)])
        result = evaluate_scores(labels, scores, threshold=0.5)
        assert not any("caps precision" in w for w in result.warnings)

    def test_few_anomalies_warn_about_instability(self):
        labels = np.array([0] * 100 + [1] * 3)
        scores = np.concatenate([np.zeros(100), np.ones(3)])
        result = evaluate_scores(labels, scores, threshold=0.5)
        assert any("unstable" in w for w in result.warnings)

    def test_anomaly_rate(self):
        labels = np.array([0] * 90 + [1] * 10)
        result = evaluate_scores(labels, labels.astype(float), threshold=0.5)
        assert result.anomaly_rate == pytest.approx(0.1)

    def test_result_serialises(self):
        labels = np.array([0, 1, 0, 1])
        result = evaluate_scores(labels, np.array([0.0, 1.0, 0.0, 1.0]), 0.5)
        payload = result.to_dict()
        assert payload["precision"] == 1.0
        assert isinstance(payload["warnings"], list)

    def test_summary_is_readable(self):
        labels = np.array([0, 1, 0, 1])
        summary = evaluate_scores(labels, np.array([0.0, 1.0, 0.0, 1.0]), 0.5).summary()
        assert "Precision" in summary
        assert "PR-AUC" in summary

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            evaluate_scores(np.array([0, 1]), np.array([0.5]), 0.5)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="zero samples"):
            evaluate_scores(np.array([]), np.array([]), 0.5)
