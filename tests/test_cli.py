"""Tests for the command-line interface.

Argument parsing runs without TensorFlow; end-to-end runs are marked slow.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.cli import build_parser, config_from_args, main


def make_csv(path, *, n_entities=40, n_months=12, labelled=False, gap=False, seed=0):
    """Write a small conforming input file, creating `path` if needed."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_entities):
        frame = pd.DataFrame(
            {
                "CP_ID": f"E{i:03d}",
                "MONTH": pd.date_range("2022-01-01", periods=n_months, freq="ME"),
                "TURNOVER": rng.normal(100, 5, n_months),
                "ASSETS": rng.normal(500, 10, n_months),
            }
        )
        if labelled:
            frame["IS_TRUE_ANOMALY"] = 0
        rows.append(frame)
    df = pd.concat(rows).reset_index(drop=True)
    if gap:
        df = df.drop(df.index[3:5]).reset_index(drop=True)

    target = path / "input.csv"
    df.to_csv(target, index=False)
    return target


class TestParsing:
    def test_train_defaults_match_config(self):
        config = config_from_args(build_parser().parse_args(["train"]))
        assert config.n_companies == 20_000
        assert config.input_path is None

    def test_input_csv_populates_config(self):
        args = build_parser().parse_args(["train", "--input-csv", "data.csv"])
        assert config_from_args(args).input_path == "data.csv"

    def test_column_overrides(self):
        args = build_parser().parse_args(
            ["train", "--id-column", "ENTITY", "--month-column", "PERIOD"]
        )
        config = config_from_args(args)
        assert (config.id_column, config.month_column) == ("ENTITY", "PERIOD")

    def test_alert_budget_override(self):
        args = build_parser().parse_args(["train", "--alert-budget", "50"])
        assert config_from_args(args).alert_budget == 50

    def test_dedupe_is_on_by_default(self):
        assert config_from_args(build_parser().parse_args(["train"])).dedupe_alerts

    def test_no_dedupe_disables_it(self):
        args = build_parser().parse_args(["train", "--no-dedupe"])
        assert not config_from_args(args).dedupe_alerts

    def test_score_requires_model_dir(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["score", "--input-csv", "x.csv"])

    def test_no_subcommand_prints_help(self, capsys):
        assert main([]) == 2
        assert "train" in capsys.readouterr().out

    def test_invalid_configuration_exits_with_code_two(self, capsys):
        assert main(["train", "--n-companies", "0"]) == 2
        assert "invalid configuration" in capsys.readouterr().err

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--help"])
        assert exc.value.code == 0


class TestInputErrors:
    def test_missing_file_exits_with_code_two(self, tmp_path, capsys):
        code = main(["train", "--input-csv", str(tmp_path / "absent.csv")])
        assert code == 2
        assert "not found" in capsys.readouterr().err

    def test_gapped_input_exits_with_code_two(self, tmp_path, capsys):
        path = make_csv(tmp_path, gap=True)
        code = main(["train", "--input-csv", str(path), "--time-steps", "3"])
        assert code == 2
        assert "gaps" in capsys.readouterr().err


@pytest.mark.slow
class TestEndToEnd:
    def test_simulated_run_writes_artifacts(self, tmp_path):
        pytest.importorskip("keras", reason="TensorFlow/Keras not installed")

        code = main(
            [
                "train",
                "--n-companies",
                "40",
                "--n-months",
                "10",
                "--n-anomalous-companies",
                "15",
                "--time-steps",
                "3",
                "--epochs",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert code == 0
        assert (tmp_path / "model.keras").exists()
        assert (tmp_path / "scaler.joblib").exists()
        assert (tmp_path / "metadata.json").exists()

        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["labelled"] is True
        assert "precision" in metrics["metrics"]

    def test_unlabelled_csv_trains_without_metrics(self, tmp_path):
        """The real-data case: no ground truth, so alerts instead of metrics."""
        pytest.importorskip("keras", reason="TensorFlow/Keras not installed")

        path = make_csv(tmp_path, labelled=False)
        out = tmp_path / "run"
        code = main(
            [
                "train",
                "--input-csv",
                str(path),
                "--time-steps",
                "3",
                "--epochs",
                "1",
                "--output-dir",
                str(out),
                "--alerts-path",
                str(tmp_path / "alerts.csv"),
            ]
        )
        assert code == 0

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["labelled"] is False
        assert metrics["metrics"] is None

        alerts = pd.read_csv(tmp_path / "alerts.csv")
        assert {"rank", "group", "score", "flagged"} <= set(alerts.columns)
        assert "is_true_anomaly" not in alerts.columns

    def test_train_then_score_round_trip(self, tmp_path):
        """Train once, then score fresh data with the saved bundle."""
        pytest.importorskip("keras", reason="TensorFlow/Keras not installed")

        train_csv = make_csv(tmp_path / "a", n_entities=40)
        model_dir = tmp_path / "bundle"

        assert (
            main(
                [
                    "train",
                    "--input-csv",
                    str(train_csv),
                    "--time-steps",
                    "3",
                    "--epochs",
                    "1",
                    "--output-dir",
                    str(model_dir),
                ]
            )
            == 0
        )

        # Fresh entities the model has never seen.
        score_csv = make_csv(tmp_path / "b", n_entities=10, seed=99)
        alerts_path = tmp_path / "scored.csv"

        assert (
            main(
                [
                    "score",
                    "--model-dir",
                    str(model_dir),
                    "--input-csv",
                    str(score_csv),
                    "--alerts-path",
                    str(alerts_path),
                ]
            )
            == 0
        )

        alerts = pd.read_csv(alerts_path)
        assert len(alerts) == 10  # deduped to one row per entity
        assert alerts["rank"].tolist() == sorted(alerts["rank"].tolist())
        assert alerts["score"].is_monotonic_decreasing

    def test_score_respects_alert_budget(self, tmp_path):
        pytest.importorskip("keras", reason="TensorFlow/Keras not installed")

        train_csv = make_csv(tmp_path / "a", n_entities=40)
        model_dir = tmp_path / "bundle"
        assert (
            main(
                [
                    "train",
                    "--input-csv",
                    str(train_csv),
                    "--time-steps",
                    "3",
                    "--epochs",
                    "1",
                    "--output-dir",
                    str(model_dir),
                ]
            )
            == 0
        )

        score_csv = make_csv(tmp_path / "b", n_entities=20, seed=99)
        alerts_path = tmp_path / "budgeted.csv"
        assert (
            main(
                [
                    "score",
                    "--model-dir",
                    str(model_dir),
                    "--input-csv",
                    str(score_csv),
                    "--alerts-path",
                    str(alerts_path),
                    "--alert-budget",
                    "5",
                ]
            )
            == 0
        )

        assert len(pd.read_csv(alerts_path)) == 5
