"""Turning window scores into a ranked list a human can work through."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALERT_COLUMNS = ("rank", "group", "score", "flagged")


def deduplicate_by_group(
    alerts: pd.DataFrame, *, group_column: str = "group"
) -> pd.DataFrame:
    """Keep only the highest-scoring window per company.

    A single anomalous month appears in up to ``2 * time_steps`` overlapping
    windows, so one real event generates a cluster of near-identical alerts.
    Left in, that cluster consumes an alert budget without adding information:
    in the baseline run one company took 4 of 10 slots. Collapsing to the best
    window per company makes the budget mean "investigate N companies".

    Args:
        alerts: Frame with a group column and a ``score`` column.
        group_column: Column identifying the company.

    Returns:
        One row per group, the highest-scoring, ordered by descending score.

    Raises:
        KeyError: If a required column is missing.
    """
    for column in (group_column, "score"):
        if column not in alerts.columns:
            raise KeyError(f"column '{column}' not found in alerts")

    best = (
        alerts.sort_values("score", ascending=False, kind="stable")
        .drop_duplicates(subset=[group_column], keep="first")
        .reset_index(drop=True)
    )
    return best


def build_alerts(
    groups: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None = None,
    budget: int | None = None,
    dedupe: bool = True,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Rank scored windows into an alert table.

    When `dedupe` is set, deduplication happens *before* the budget is applied,
    so a budget of N yields N distinct companies rather than N windows that
    might all describe the same event.

    Args:
        groups: ``(n,)`` company id per window.
        scores: ``(n,)`` anomaly score per window, higher being more anomalous.
        threshold: Flag windows scoring strictly above this.
        budget: Keep only the top `budget` rows after deduplication. Applied
            in addition to `threshold` when both are given.
        dedupe: Collapse to the best window per company.
        labels: Optional ground truth, included as an ``is_true_anomaly``
            column when present.

    Returns:
        A frame ordered by descending score with ``rank``, ``group``,
        ``score``, and ``flagged`` columns.

    Raises:
        ValueError: If `groups` and `scores` differ in length, or `budget` is
            below 1.
    """
    if len(groups) != len(scores):
        raise ValueError(
            f"length mismatch: groups {len(groups)} vs scores {len(scores)}"
        )
    if budget is not None and budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    alerts = pd.DataFrame({"group": groups, "score": scores})
    if labels is not None:
        if len(labels) != len(scores):
            raise ValueError(
                f"length mismatch: labels {len(labels)} vs scores {len(scores)}"
            )
        alerts["is_true_anomaly"] = labels.astype(int)

    alerts["flagged"] = alerts["score"] > threshold if threshold is not None else True

    if dedupe:
        n_before = len(alerts)
        alerts = deduplicate_by_group(alerts)
        logger.debug("deduplicated %d windows to %d companies", n_before, len(alerts))
    else:
        alerts = alerts.sort_values(
            "score", ascending=False, kind="stable"
        ).reset_index(drop=True)

    if budget is not None:
        alerts = alerts.head(budget)

    alerts.insert(0, "rank", np.arange(1, len(alerts) + 1))
    return alerts.reset_index(drop=True)


def write_alerts(alerts: pd.DataFrame, path: str | Path) -> Path:
    """Write an alert table to CSV or Parquet, chosen by suffix.

    Args:
        alerts: Table from :func:`build_alerts`.
        path: Destination. Parent directories are created.

    Returns:
        The path written.

    Raises:
        ValueError: If the suffix is not supported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        alerts.to_parquet(path, index=False)
    elif suffix == ".csv":
        alerts.to_csv(path, index=False)
    else:
        raise ValueError(
            f"unsupported alert format '{suffix}'; expected .csv or .parquet"
        )

    return path
