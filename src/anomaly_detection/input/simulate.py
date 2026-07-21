"""Simulation of corporate monthly time series with injected anomalies.

The generator produces a tidy long-format frame: one row per company-month,
with a ground-truth label marking the months that were tampered with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly_detection.input.io import (
    DEFAULT_ID_COLUMN,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MONTH_COLUMN,
)

# Simulated data conforms to the same contract as real input, so the two
# sources are interchangeable downstream.
ID_COLUMN = DEFAULT_ID_COLUMN
"""Column identifying a company (the grouping key for splits and windowing)."""

LABEL_COLUMN = DEFAULT_LABEL_COLUMN
"""Ground-truth column: 1 where an anomaly was injected, else 0."""

MONTH_COLUMN = DEFAULT_MONTH_COLUMN

# Multiplicative factors applied to the previous month's turnover. A spike
# inflates it; a collapse guts it. Both are far outside normal variation.
SPIKE_RANGE = (1.7, 3.0)
COLLAPSE_RANGE = (0.01, 0.30)

# Anomalies are injected only where the preceding month is already large, so
# the tampered value is a visible break rather than noise in the low range.
INJECTION_QUANTILE = 0.90

# --- Multivariate ("lead-lag") anomaly mode ---------------------------------
# Assets normally track LAST month's turnover deviation (a lead-lag link the
# next-step model can see in its input window). The anomaly breaks that link
# while keeping every value within its own normal range, so it is invisible in
# any single series and only shows up in how the two move together.
ASSET_MULTIPLIER = 5.0  # assets sit at ~5x turnover, matching the univariate scale
MULTIVARIATE_TURNOVER_NOISE = 0.15  # marginal spread of the driver
LEADLAG_BETA = 0.9  # fraction of last month's turnover deviation assets follow
LEADLAG_COND_NOISE = 0.02  # tight conditional noise -> a break is huge conditionally
LEADLAG_DRIVER_KICK = 1.5  # driver move (in turnover sigmas) at an anomaly


def generate_data(
    n_companies: int,
    n_months: int,
    *,
    seed: int = 42,
    n_anomalous_companies: int = 40,
    anomaly_offsets: tuple[int, ...] = (-1, -2, -3),
    anomaly_probability: float = 0.5,
    anomaly_kind: str = "univariate",
) -> pd.DataFrame:
    """Simulate company time series and inject anomalies.

    Every source of randomness draws from a single seeded generator, so the
    same ``seed`` always yields the same frame regardless of global numpy
    state.

    Two anomaly kinds are supported. ``"univariate"`` injects a turnover spike
    or collapse — visible in a single variable. ``"multivariate"`` breaks a
    lead-lag relationship between turnover and assets while keeping every value
    within its own normal range, so the anomaly is invisible in any single
    series and only detectable jointly (see the module constants).

    Args:
        n_companies: Number of companies to simulate.
        n_months: Months of history per company.
        seed: Seed for the internal random generator.
        n_anomalous_companies: How many companies are eligible for injection.
        anomaly_offsets: Negative offsets from the end of each series marking
            injectable months.
        anomaly_probability: Chance an eligible slot is actually tampered with.
        anomaly_kind: ``"univariate"`` or ``"multivariate"``.

    Returns:
        Long-format frame with columns ``CP_ID``, ``MONTH``, ``TURNOVER``,
        ``ASSETS``, ``TURNOVER_ROC``, and ``IS_TRUE_ANOMALY``, sorted by
        company then month.

    Raises:
        ValueError: If ``n_anomalous_companies`` exceeds ``n_companies``, an
            offset falls outside the series, or ``anomaly_kind`` is unknown.
    """
    if n_anomalous_companies > n_companies:
        raise ValueError(
            f"n_anomalous_companies ({n_anomalous_companies}) exceeds "
            f"n_companies ({n_companies})"
        )
    for off in anomaly_offsets:
        if off >= 0 or abs(off) > n_months:
            raise ValueError(
                f"offset {off} must be negative and within {n_months} months"
            )
    if anomaly_kind not in ("univariate", "multivariate"):
        raise ValueError(
            f"anomaly_kind must be 'univariate' or 'multivariate', got {anomaly_kind!r}"
        )

    rng = np.random.default_rng(seed)
    base = np.linspace(1e7, 2e7, n_months)

    simulate = (
        _simulate_univariate if anomaly_kind == "univariate" else _simulate_multivariate
    )
    turnover, assets, labels = simulate(
        rng,
        base,
        n_companies=n_companies,
        n_months=n_months,
        n_anomalous_companies=n_anomalous_companies,
        anomaly_offsets=anomaly_offsets,
        anomaly_probability=anomaly_probability,
    )

    # `ME` (month end) — `M` was removed in pandas 3.0.
    months = pd.date_range("2022-01-01", periods=n_months, freq="ME")
    width = max(4, len(str(n_companies - 1)))
    company_ids = np.array([f"CP_{i:0{width}d}" for i in range(n_companies)])

    df = pd.DataFrame(
        {
            ID_COLUMN: np.repeat(company_ids, n_months),
            MONTH_COLUMN: np.tile(months, n_companies),
            "TURNOVER": turnover.ravel(),
            "ASSETS": assets.ravel(),
            LABEL_COLUMN: labels.ravel(),
        }
    )

    # Rate of change is derived after injection so anomalies show up in it.
    df["TURNOVER_ROC"] = (
        df.groupby(ID_COLUMN)["TURNOVER"].pct_change().fillna(0.0).clip(-5, 5)
    )

    return df


def _simulate_univariate(
    rng: np.random.Generator,
    base: np.ndarray,
    *,
    n_companies: int,
    n_months: int,
    n_anomalous_companies: int,
    anomaly_offsets: tuple[int, ...],
    anomaly_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turnover spikes/collapses — an anomaly visible in a single variable."""
    # Vectorised simulation: draw every company at once rather than looping.
    turnover = base * rng.normal(1.0, 0.05, size=(n_companies, n_months))
    assets = turnover * rng.normal(5.0, 0.1, size=(n_companies, n_months))

    labels = np.zeros((n_companies, n_months), dtype=np.int8)

    # Threshold computed before injection, so tampered values cannot shift it.
    turnover_threshold = np.quantile(turnover, INJECTION_QUANTILE)

    n_eligible = min(n_anomalous_companies, n_companies)
    candidates = rng.choice(n_companies, size=n_eligible, replace=False)

    for company in candidates:
        for off in anomaly_offsets:
            month = n_months + off  # offset is negative
            prev = month - 1
            if prev < 0:
                continue
            # Only tamper when the preceding month is already in the top decile.
            if turnover[company, prev] <= turnover_threshold:
                continue
            if rng.random() >= anomaly_probability:
                continue

            low, high = SPIKE_RANGE if rng.random() < 0.5 else COLLAPSE_RANGE
            factor = rng.uniform(low, high)
            turnover[company, month] = max(0.0, turnover[company, prev] * factor)
            labels[company, month] = 1

    return turnover, assets, labels


def _simulate_multivariate(
    rng: np.random.Generator,
    base: np.ndarray,
    *,
    n_companies: int,
    n_months: int,
    n_anomalous_companies: int,
    anomaly_offsets: tuple[int, ...],
    anomaly_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A broken lead-lag link — invisible in any single series, only jointly.

    Turnover is i.i.d. monthly noise, so it carries no self-structure to
    exploit. Assets normally track *last* month's turnover deviation, tightly
    (conditional noise ``LEADLAG_COND_NOISE``) inside a wide marginal spread
    (``MULTIVARIATE_TURNOVER_NOISE`` scaled by ``LEADLAG_BETA``). An anomaly
    flips that response: the driver moves a within-normal amount last month and
    assets react the wrong way this month, so both values stay marginally
    ordinary while their relationship is broken.
    """
    turnover = base * rng.normal(
        1.0, MULTIVARIATE_TURNOVER_NOISE, size=(n_companies, n_months)
    )
    relative_deviation = turnover / base - 1.0

    assets = np.empty_like(turnover)
    assets[:, 0] = (
        ASSET_MULTIPLIER
        * base[0]
        * (1.0 + rng.normal(0.0, LEADLAG_COND_NOISE, n_companies))
    )
    for month in range(1, n_months):
        assets[:, month] = (
            ASSET_MULTIPLIER
            * base[month]
            * (1.0 + LEADLAG_BETA * relative_deviation[:, month - 1])
            * (1.0 + rng.normal(0.0, LEADLAG_COND_NOISE, n_companies))
        )

    labels = np.zeros((n_companies, n_months), dtype=np.int8)

    n_eligible = min(n_anomalous_companies, n_companies)
    candidates = rng.choice(n_companies, size=n_eligible, replace=False)

    for company in candidates:
        for off in anomaly_offsets:
            month = n_months + off  # offset is negative
            prev = month - 1
            if prev < 0:
                continue
            if rng.random() >= anomaly_probability:
                continue

            direction = 1.0 if rng.random() < 0.5 else -1.0
            kick = direction * LEADLAG_DRIVER_KICK * MULTIVARIATE_TURNOVER_NOISE
            # A within-normal driver move last month...
            turnover[company, prev] = base[prev] * (1.0 + kick)
            # ...to which assets react the wrong way, staying in their own range.
            assets[company, month] = (
                ASSET_MULTIPLIER
                * base[month]
                * (1.0 - LEADLAG_BETA * kick)
                * (1.0 + rng.normal(0.0, LEADLAG_COND_NOISE))
            )
            labels[company, month] = 1

    return turnover, assets, labels
