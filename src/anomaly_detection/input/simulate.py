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


def generate_data(
    n_companies: int,
    n_months: int,
    *,
    seed: int = 42,
    n_anomalous_companies: int = 40,
    anomaly_offsets: tuple[int, ...] = (-1, -2, -3),
    anomaly_probability: float = 0.5,
) -> pd.DataFrame:
    """Simulate company time series and inject turnover anomalies.

    Every source of randomness draws from a single seeded generator, so the
    same ``seed`` always yields the same frame regardless of global numpy
    state.

    Args:
        n_companies: Number of companies to simulate.
        n_months: Months of history per company.
        seed: Seed for the internal random generator.
        n_anomalous_companies: How many companies are eligible for injection.
        anomaly_offsets: Negative offsets from the end of each series marking
            injectable months.
        anomaly_probability: Chance an eligible slot is actually tampered with.

    Returns:
        Long-format frame with columns ``CP_ID``, ``MONTH``, ``TURNOVER``,
        ``ASSETS``, ``TURNOVER_ROC``, and ``IS_TRUE_ANOMALY``, sorted by
        company then month.

    Raises:
        ValueError: If ``n_anomalous_companies`` exceeds ``n_companies`` or an
            offset falls outside the series.
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

    rng = np.random.default_rng(seed)

    # Vectorised simulation: draw every company at once rather than looping.
    base = np.linspace(1e7, 2e7, n_months)
    turnover = base * rng.normal(1.0, 0.05, size=(n_companies, n_months))
    assets = turnover * rng.normal(5.0, 0.1, size=(n_companies, n_months))

    # `ME` (month end) — `M` was removed in pandas 3.0.
    months = pd.date_range("2022-01-01", periods=n_months, freq="ME")
    width = max(4, len(str(n_companies - 1)))
    company_ids = np.array([f"CP_{i:0{width}d}" for i in range(n_companies)])

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
