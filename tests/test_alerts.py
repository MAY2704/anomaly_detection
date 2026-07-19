"""Tests for ranking and deduplicating alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.output.alerts import (
    build_alerts,
    deduplicate_by_group,
    write_alerts,
)


class TestDeduplicate:
    def test_keeps_the_best_window_per_entity(self):
        alerts = pd.DataFrame({"group": ["A", "A", "B"], "score": [0.1, 0.9, 0.5]})
        out = deduplicate_by_group(alerts)
        assert len(out) == 2
        assert out.loc[out.group == "A", "score"].item() == pytest.approx(0.9)

    def test_output_is_ordered_by_score(self):
        alerts = pd.DataFrame({"group": list("ABC"), "score": [0.2, 0.9, 0.5]})
        out = deduplicate_by_group(alerts)
        assert out["group"].tolist() == ["B", "C", "A"]

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="score"):
            deduplicate_by_group(pd.DataFrame({"group": ["A"]}))


class TestBuildAlerts:
    def test_one_event_does_not_consume_the_budget(self):
        """The reason dedupe exists.

        A single anomalous month lands in many overlapping windows, so one
        real event produces a cluster of near-identical high scores. Without
        deduplication that cluster eats an alert budget: in the baseline run
        one company took 4 of 10 slots.
        """
        groups = np.array(["A"] * 8 + [f"B{i}" for i in range(8)])
        scores = np.concatenate([np.full(8, 9.0), np.linspace(1.0, 8.0, 8)])

        deduped = build_alerts(groups, scores, budget=5, dedupe=True)
        raw = build_alerts(groups, scores, budget=5, dedupe=False)

        assert deduped["group"].nunique() == 5
        assert set(raw["group"]) == {"A"}  # all five slots are the same event

    def test_budget_applies_after_dedupe(self):
        groups = np.array(["A", "A", "B", "C"])
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        assert len(build_alerts(groups, scores, budget=3, dedupe=True)) == 3

    def test_ranked_by_descending_score(self):
        groups = np.array(["A", "B", "C"])
        scores = np.array([0.5, 0.9, 0.1])
        alerts = build_alerts(groups, scores)
        assert alerts["group"].tolist() == ["B", "A", "C"]
        assert alerts["rank"].tolist() == [1, 2, 3]

    def test_threshold_sets_the_flagged_column(self):
        groups = np.array(["A", "B", "C"])
        scores = np.array([0.1, 0.5, 0.9])
        alerts = build_alerts(groups, scores, threshold=0.4)
        assert alerts["flagged"].sum() == 2

    def test_without_threshold_everything_is_flagged(self):
        alerts = build_alerts(np.array(["A", "B"]), np.array([0.1, 0.2]))
        assert alerts["flagged"].all()

    def test_labels_are_carried_through(self):
        groups = np.array(["A", "B"])
        scores = np.array([0.9, 0.1])
        alerts = build_alerts(groups, scores, labels=np.array([1, 0]))
        assert alerts.loc[alerts.group == "A", "is_true_anomaly"].item() == 1

    def test_labels_omitted_when_unlabelled(self):
        alerts = build_alerts(np.array(["A"]), np.array([0.9]))
        assert "is_true_anomaly" not in alerts.columns

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            build_alerts(np.array(["A", "B"]), np.array([0.1]))

    def test_label_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels"):
            build_alerts(
                np.array(["A", "B"]), np.array([0.1, 0.2]), labels=np.array([1])
            )

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError, match="budget"):
            build_alerts(np.array(["A"]), np.array([0.1]), budget=0)


class TestWriteAlerts:
    @pytest.mark.parametrize("suffix", [".csv", ".parquet"])
    def test_round_trips(self, tmp_path, suffix):
        alerts = build_alerts(np.array(["A", "B"]), np.array([0.9, 0.1]))
        path = write_alerts(alerts, tmp_path / f"alerts{suffix}")
        assert path.exists()

        reloaded = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
        assert reloaded["group"].tolist() == ["A", "B"]

    def test_creates_parent_directories(self, tmp_path):
        alerts = build_alerts(np.array(["A"]), np.array([0.9]))
        path = write_alerts(alerts, tmp_path / "nested" / "deep" / "alerts.csv")
        assert path.exists()

    def test_unsupported_format_raises(self, tmp_path):
        alerts = build_alerts(np.array(["A"]), np.array([0.9]))
        with pytest.raises(ValueError, match="unsupported alert format"):
            write_alerts(alerts, tmp_path / "alerts.txt")
