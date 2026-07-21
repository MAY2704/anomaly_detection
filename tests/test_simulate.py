"""Tests for data simulation and anomaly injection."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.input.simulate import ID_COLUMN, LABEL_COLUMN, generate_data

EXPECTED_COLUMNS = {
    ID_COLUMN,
    "MONTH",
    "TURNOVER",
    "ASSETS",
    "TURNOVER_ROC",
    LABEL_COLUMN,
}


class TestShape:
    def test_row_count_is_companies_times_months(self):
        df = generate_data(25, 12, seed=0, n_anomalous_companies=5)
        assert len(df) == 25 * 12

    def test_expected_columns_present(self):
        df = generate_data(10, 12, seed=0, n_anomalous_companies=2)
        assert set(df.columns) == EXPECTED_COLUMNS

    def test_every_company_has_full_history(self):
        df = generate_data(10, 12, seed=0, n_anomalous_companies=2)
        assert (df.groupby(ID_COLUMN).size() == 12).all()

    def test_ids_are_zero_padded_and_unique(self):
        df = generate_data(10, 6, seed=0, n_anomalous_companies=1)
        ids = df[ID_COLUMN].unique()
        assert len(ids) == 10
        assert all(i.startswith("CP_") for i in ids)


class TestReproducibility:
    def test_same_seed_gives_identical_frames(self):
        """Regression: injection used global numpy state, not the seeded rng.

        `generate_data(seed=...)` previously controlled only the base series.
        Anomaly placement came from `np.random.*`, so results depended on
        whatever global seed the caller happened to have set.
        """
        a = generate_data(20, 12, seed=123, n_anomalous_companies=5)
        b = generate_data(20, 12, seed=123, n_anomalous_companies=5)
        assert a.equals(b)

    def test_seed_controls_injection_independently_of_global_state(self):
        # Deliberately polluting global state: the point is that generate_data
        # ignores it entirely.
        np.random.seed(1)  # noqa: NPY002
        a = generate_data(20, 12, seed=99, n_anomalous_companies=5)
        np.random.seed(2)  # noqa: NPY002
        b = generate_data(20, 12, seed=99, n_anomalous_companies=5)
        assert a[LABEL_COLUMN].tolist() == b[LABEL_COLUMN].tolist()

    def test_different_seeds_give_different_data(self):
        a = generate_data(20, 12, seed=1, n_anomalous_companies=5)
        b = generate_data(20, 12, seed=2, n_anomalous_companies=5)
        assert not a["TURNOVER"].equals(b["TURNOVER"])


class TestAnomalyInjection:
    def test_anomalies_are_injected(self):
        df = generate_data(
            100, 12, seed=0, n_anomalous_companies=40, anomaly_probability=1.0
        )
        assert df[LABEL_COLUMN].sum() > 0

    def test_injection_confined_to_configured_offsets(self):
        n_months = 12
        offsets = (-1, -2)
        df = generate_data(
            100,
            n_months,
            seed=0,
            n_anomalous_companies=40,
            anomaly_offsets=offsets,
            anomaly_probability=1.0,
        )
        positions = df.groupby(ID_COLUMN).cumcount()[df[LABEL_COLUMN] == 1]
        allowed = {n_months + off for off in offsets}
        assert set(positions.unique()) <= allowed

    def test_anomaly_count_bounded_by_eligible_companies(self):
        df = generate_data(
            100, 12, seed=0, n_anomalous_companies=10, anomaly_probability=1.0
        )
        affected = df.loc[df[LABEL_COLUMN] == 1, ID_COLUMN].nunique()
        assert affected <= 10

    def test_zero_probability_injects_nothing(self):
        df = generate_data(
            50, 12, seed=0, n_anomalous_companies=20, anomaly_probability=0.0
        )
        assert df[LABEL_COLUMN].sum() == 0

    def test_anomalous_turnover_deviates_sharply(self):
        df = generate_data(
            200, 12, seed=0, n_anomalous_companies=60, anomaly_probability=1.0
        )
        anomalous = df.loc[df[LABEL_COLUMN] == 1]
        if anomalous.empty:
            pytest.skip("no anomalies injected at this seed")

        # Each anomaly is a spike or collapse relative to the prior month.
        prev = df["TURNOVER"].shift(1).loc[anomalous.index]
        ratio = anomalous["TURNOVER"].to_numpy() / prev.to_numpy()
        assert ((ratio > 1.5) | (ratio < 0.35)).all()

    def test_turnover_never_negative(self):
        df = generate_data(
            100, 12, seed=0, n_anomalous_companies=40, anomaly_probability=1.0
        )
        assert (df["TURNOVER"] >= 0).all()


class TestDerivedFeatures:
    def test_roc_is_clipped(self):
        df = generate_data(
            100, 12, seed=0, n_anomalous_companies=40, anomaly_probability=1.0
        )
        assert df["TURNOVER_ROC"].between(-5, 5).all()

    def test_roc_first_month_is_zero(self):
        df = generate_data(10, 12, seed=0, n_anomalous_companies=2)
        first = df.groupby(ID_COLUMN).head(1)
        assert (first["TURNOVER_ROC"] == 0).all()

    def test_no_nulls_anywhere(self):
        df = generate_data(20, 12, seed=0, n_anomalous_companies=5)
        assert not df.isnull().to_numpy().any()


class TestValidation:
    def test_too_many_anomalous_companies_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            generate_data(5, 12, seed=0, n_anomalous_companies=10)

    @pytest.mark.parametrize("offset", [0, 1, -99])
    def test_invalid_offsets_raise(self, offset):
        with pytest.raises(ValueError, match="offset"):
            generate_data(
                10, 12, seed=0, n_anomalous_companies=2, anomaly_offsets=(offset,)
            )

    def test_unknown_anomaly_kind_raises(self):
        with pytest.raises(ValueError, match="anomaly_kind"):
            generate_data(10, 12, seed=0, n_anomalous_companies=2, anomaly_kind="bogus")


class TestMultivariateAnomaly:
    """The multivariate mode injects a broken lead-lag between the features.

    Its defining property is that the anomaly is invisible in any single
    series and only shows up in how turnover and assets move together — the
    reason a joint model is needed at all.
    """

    def _generate(self, **kwargs):
        params = {
            "n_companies": 200,
            "n_months": 24,
            "seed": 0,
            "n_anomalous_companies": 60,
            "anomaly_probability": 1.0,
            "anomaly_offsets": (-1, -2),
            "anomaly_kind": "multivariate",
        }
        params.update(kwargs)
        return generate_data(**params)

    def test_same_columns_as_univariate(self):
        assert set(self._generate().columns) == EXPECTED_COLUMNS

    def test_anomalies_are_injected(self):
        assert self._generate()[LABEL_COLUMN].sum() > 0

    def test_reproducible_for_a_seed(self):
        assert self._generate(seed=7).equals(self._generate(seed=7))

    def _within_series_z(self, df, column):
        """|z| of each anomalous value against its own company's history."""
        anomalies = df[df[LABEL_COLUMN] == 1]
        stats = df.groupby(ID_COLUMN)[column].agg(["mean", "std"])

        def z(row):
            mean = stats.loc[row[ID_COLUMN], "mean"]
            std = stats.loc[row[ID_COLUMN], "std"]
            return abs(row[column] - mean) / std

        return anomalies.apply(z, axis=1)

    def test_anomalies_are_marginally_invisible(self):
        """Each anomalous value is unremarkable within its own series.

        This is the whole point: a per-series detector cannot see these. The
        anomalous values sit within a few standard deviations of each
        company's own history for both features — nowhere near the blatant
        break a univariate spike would produce.
        """
        df = self._generate()
        assert (df[LABEL_COLUMN] == 1).any()
        for column in ("TURNOVER", "ASSETS"):
            # A univariate spike lands well past 3 sigma; these stay modest.
            assert self._within_series_z(df, column).median() < 3.0

    def test_turnover_is_not_itself_spiked(self):
        """Turnover stays i.i.d.-normal; the anomaly is in the relationship."""
        assert self._within_series_z(self._generate(), "TURNOVER").max() < 4.0

    def test_no_nulls(self):
        assert not self._generate().isnull().to_numpy().any()


class TestAnomalyKindsAreDistinct:
    def test_kinds_produce_different_data(self):
        shared = {"seed": 1, "n_anomalous_companies": 20}
        uni = generate_data(50, 20, anomaly_kind="univariate", **shared)
        multi = generate_data(50, 20, anomaly_kind="multivariate", **shared)
        assert not uni["ASSETS"].equals(multi["ASSETS"])

    def test_univariate_is_the_default(self):
        shared = {"seed": 3, "n_anomalous_companies": 10}
        explicit = generate_data(30, 15, anomaly_kind="univariate", **shared)
        default = generate_data(30, 15, **shared)
        assert default.equals(explicit)
