"""Tests for loading and validating real input files.

The four traps these guard against are all *silent* failures on real data —
they change what the model sees without raising anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.input.io import (
    DataQualityError,
    derive_rate_of_change,
    load_and_prepare,
    load_table,
    prepare_input,
    validate_frame,
)

FEATURES = ["TURNOVER", "ASSETS"]


def make_frame(specs, *, labelled=False, start="2022-01-01"):
    """Build a frame from (id, n_months) pairs."""
    rows = []
    for name, n in specs:
        frame = pd.DataFrame(
            {
                "CP_ID": name,
                "MONTH": pd.date_range(start, periods=n, freq="ME"),
                "TURNOVER": np.linspace(100.0, 200.0, n),
                "ASSETS": np.linspace(500.0, 900.0, n),
            }
        )
        if labelled:
            frame["IS_TRUE_ANOMALY"] = 0
        rows.append(frame)
    return pd.concat(rows).reset_index(drop=True)


def write(df, path, fmt="csv"):
    target = path / f"input.{fmt}"
    df.to_parquet(target, index=False) if fmt == "parquet" else df.to_csv(
        target, index=False
    )
    return target


class TestLoadTable:
    @pytest.mark.parametrize("fmt", ["csv", "parquet"])
    def test_round_trips_both_formats(self, tmp_path, fmt):
        df = make_frame([("A", 12)])
        loaded = load_table(write(df, tmp_path, fmt))
        assert len(loaded) == 12
        assert pd.api.types.is_datetime64_any_dtype(loaded["MONTH"])

    def test_unparseable_dates_raise_with_examples(self, tmp_path):
        """Trap: bad dates become NaT and silently drop rows much later."""
        df = make_frame([("A", 6)])
        df["MONTH"] = df["MONTH"].astype(str)
        df.loc[2, "MONTH"] = "not-a-date"

        with pytest.raises(ValueError, match="could not be parsed"):
            load_table(write(df, tmp_path))

    def test_error_names_the_offending_value(self, tmp_path):
        df = make_frame([("A", 6)])
        df["MONTH"] = df["MONTH"].astype(str)
        df.loc[1, "MONTH"] = "31/02/2022"

        with pytest.raises(ValueError, match="31/02/2022"):
            load_table(write(df, tmp_path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_table(tmp_path / "absent.csv")

    def test_unsupported_suffix_raises(self, tmp_path):
        path = tmp_path / "data.xlsx"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported file type"):
            load_table(path)

    def test_missing_month_column_raises(self, tmp_path):
        df = make_frame([("A", 6)]).drop(columns=["MONTH"])
        with pytest.raises(ValueError, match="month column"):
            load_table(write(df, tmp_path))


class TestValidateFrame:
    def test_reports_shape_and_labelling(self):
        report = validate_frame(
            make_frame([("A", 12), ("B", 12)], labelled=True),
            features=FEATURES,
            time_steps=6,
        )
        assert (report.n_rows, report.n_companies) == (24, 2)
        assert report.labelled

    def test_detects_unlabelled_input(self):
        report = validate_frame(
            make_frame([("A", 12)]), features=FEATURES, time_steps=6
        )
        assert not report.labelled

    def test_detects_short_series(self):
        """Trap: a company with too little history contributes zero windows."""
        report = validate_frame(
            make_frame([("LONG", 12), ("SHORT", 4)]), features=FEATURES, time_steps=6
        )
        assert report.short_companies == ["SHORT"]

    def test_series_exactly_time_steps_long_is_short(self):
        # A window needs time_steps inputs plus one shifted target month.
        report = validate_frame(
            make_frame([("EXACT", 6)]), features=FEATURES, time_steps=6
        )
        assert report.short_companies == ["EXACT"]

    def test_detects_calendar_gaps(self):
        """Trap: a window spanning a gap treats non-adjacent months as adjacent."""
        df = make_frame([("GAPPED", 12)])
        df = df.drop(df.index[4:7]).reset_index(drop=True)

        report = validate_frame(df, features=FEATURES, time_steps=3)
        assert report.gapped_companies == ["GAPPED"]

    def test_contiguous_months_are_not_flagged(self):
        report = validate_frame(
            make_frame([("A", 24)]), features=FEATURES, time_steps=6
        )
        assert report.gapped_companies == []

    def test_detects_duplicate_periods(self):
        df = pd.concat([make_frame([("A", 6)]), make_frame([("A", 6)])])
        report = validate_frame(df, features=FEATURES, time_steps=3)
        assert report.duplicate_periods == 6

    def test_missing_feature_raises(self):
        df = make_frame([("A", 12)]).drop(columns=["ASSETS"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_frame(df, features=FEATURES, time_steps=6)

    def test_non_numeric_feature_raises(self):
        df = make_frame([("A", 12)])
        df["TURNOVER"] = "lots"
        with pytest.raises(ValueError, match="expected numeric"):
            validate_frame(df, features=FEATURES, time_steps=6)

    def test_null_feature_raises(self):
        df = make_frame([("A", 12)])
        df.loc[3, "TURNOVER"] = np.nan
        with pytest.raises(ValueError, match="nulls"):
            validate_frame(df, features=FEATURES, time_steps=6)

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_frame(pd.DataFrame(), features=FEATURES, time_steps=6)

    def test_summary_is_readable(self):
        report = validate_frame(
            make_frame([("A", 12), ("S", 3)]), features=FEATURES, time_steps=6
        )
        text = report.summary()
        assert "Companies" in text
        assert "unsupervised" in text


class TestPrepareInput:
    def test_adds_label_column_when_absent(self):
        out, report = prepare_input(
            make_frame([("A", 12)]), features=FEATURES, time_steps=6
        )
        assert not report.labelled
        assert (out["IS_TRUE_ANOMALY"] == 0).all()

    def test_preserves_existing_labels(self):
        df = make_frame([("A", 12)], labelled=True)
        df.loc[5, "IS_TRUE_ANOMALY"] = 1
        out, report = prepare_input(df, features=FEATURES, time_steps=6)
        assert report.labelled
        assert out["IS_TRUE_ANOMALY"].sum() == 1

    def test_drops_short_companies_and_reports_it(self):
        out, report = prepare_input(
            make_frame([("LONG", 12), ("SHORT", 3)]), features=FEATURES, time_steps=6
        )
        assert set(out["CP_ID"]) == {"LONG"}
        assert report.short_companies == ["SHORT"]
        assert report.n_dropped == 3

    def test_short_companies_can_be_made_fatal(self):
        with pytest.raises(DataQualityError, match="cannot form a window"):
            prepare_input(
                make_frame([("LONG", 12), ("SHORT", 3)]),
                features=FEATURES,
                time_steps=6,
                drop_short=False,
            )

    def test_gaps_raise_by_default(self):
        df = make_frame([("GAPPED", 12)])
        df = df.drop(df.index[4:7]).reset_index(drop=True)

        with pytest.raises(DataQualityError, match="gaps"):
            prepare_input(df, features=FEATURES, time_steps=3)

    def test_gaps_can_be_allowed_explicitly(self):
        df = make_frame([("GAPPED", 12)])
        df = df.drop(df.index[4:7]).reset_index(drop=True)

        out, report = prepare_input(
            df, features=FEATURES, time_steps=3, allow_gaps=True
        )
        assert len(out) == 9
        assert report.gapped_companies == ["GAPPED"]

    def test_duplicates_always_raise(self):
        df = pd.concat([make_frame([("A", 12)]), make_frame([("A", 12)])])
        with pytest.raises(DataQualityError, match="duplicate"):
            prepare_input(df, features=FEATURES, time_steps=3)

    def test_output_is_sorted_by_entity_then_period(self):
        df = make_frame([("B", 8), ("A", 8)]).sample(frac=1.0, random_state=0)
        out, _ = prepare_input(df, features=FEATURES, time_steps=3)
        assert out["CP_ID"].is_monotonic_increasing
        for _, group in out.groupby("CP_ID"):
            assert group["MONTH"].is_monotonic_increasing

    def test_all_companies_too_short_raises(self):
        with pytest.raises(DataQualityError, match="nothing to score"):
            prepare_input(
                make_frame([("A", 3), ("B", 2)]), features=FEATURES, time_steps=6
            )


class TestDeriveRateOfChange:
    def test_first_period_is_zero(self):
        out = derive_rate_of_change(
            make_frame([("A", 6)]), source="TURNOVER", target="ROC"
        )
        assert out.groupby("CP_ID")["ROC"].first().eq(0).all()

    def test_does_not_span_entities(self):
        out = derive_rate_of_change(
            make_frame([("A", 6), ("B", 6)]), source="TURNOVER", target="ROC"
        )
        assert out.groupby("CP_ID")["ROC"].first().eq(0).all()

    def test_clipping_bounds_extremes(self):
        df = make_frame([("A", 4)])
        df.loc[0, "TURNOVER"] = 1e-9  # next period is a vast relative jump
        out = derive_rate_of_change(df, source="TURNOVER", target="ROC", clip=5.0)
        assert out["ROC"].between(-5, 5).all()

    def test_missing_source_raises(self):
        with pytest.raises(KeyError, match="ABSENT"):
            derive_rate_of_change(make_frame([("A", 6)]), source="ABSENT", target="ROC")


class TestLoadAndPrepare:
    def test_derives_missing_roc_feature(self, tmp_path):
        path = write(make_frame([("A", 12), ("B", 12)]), tmp_path)
        out, report = load_and_prepare(
            path,
            features=["TURNOVER", "ASSETS", "TURNOVER_ROC"],
            time_steps=6,
            rate_of_change_from="TURNOVER",
        )
        assert "TURNOVER_ROC" in out.columns
        assert report.n_companies == 2

    def test_unlabelled_file_loads_in_unsupervised_mode(self, tmp_path):
        path = write(make_frame([("A", 12)]), tmp_path)
        _, report = load_and_prepare(path, features=FEATURES, time_steps=6)
        assert not report.labelled

    def test_gap_in_file_raises(self, tmp_path):
        df = make_frame([("A", 12)])
        df = df.drop(df.index[5:8]).reset_index(drop=True)
        path = write(df, tmp_path)

        with pytest.raises(DataQualityError, match="gaps"):
            load_and_prepare(path, features=FEATURES, time_steps=3)
