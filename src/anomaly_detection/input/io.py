"""Loading and validating real input files.

Real tables break the pipeline's assumptions in ways that are silent rather
than loud: a company with too little history contributes no windows, a gap in
the calendar gets spanned as if contiguous. Both quietly change what the model
sees. Everything here exists to turn those into visible failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ID_COLUMN = "CP_ID"
DEFAULT_MONTH_COLUMN = "MONTH"
DEFAULT_LABEL_COLUMN = "IS_TRUE_ANOMALY"

REQUIRED_COLUMNS = (DEFAULT_ID_COLUMN, DEFAULT_MONTH_COLUMN)
"""Columns every input must carry, on top of the configured features.

The label column is optional: real data is usually unlabelled.
"""

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq"}

# Monthly observations land 28-31 days apart. Anything beyond this means a
# missing period, not month-length variation.
MAX_NORMAL_GAP_DAYS = 32


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """Format a count with the right noun form, for readable error messages."""
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


@dataclass(frozen=True)
class DataQualityReport:
    """What validation found.

    Attributes:
        n_rows: Rows in the frame.
        n_companies: Distinct ids.
        labelled: Whether a usable label column was present.
        short_companies: Ids with too little history to form a single window.
        gapped_companies: Ids whose observations are not consecutive months.
        duplicate_periods: Count of (id, month) pairs appearing more than once.
        n_dropped: Rows removed during preparation.
    """

    n_rows: int
    n_companies: int
    labelled: bool
    short_companies: list[str] = field(default_factory=list)
    gapped_companies: list[str] = field(default_factory=list)
    duplicate_periods: int = 0
    n_dropped: int = 0

    @property
    def has_problems(self) -> bool:
        """Whether anything needs the caller's attention."""
        return bool(
            self.short_companies or self.gapped_companies or self.duplicate_periods
        )

    def summary(self) -> str:
        """Return a human-readable report."""
        lines = [
            f"Rows:      {self.n_rows:,}",
            f"Companies: {self.n_companies:,}",
            f"Labelled:  {'yes' if self.labelled else 'no (unsupervised mode)'}",
        ]
        if self.n_dropped:
            lines.append(f"Dropped:   {self.n_dropped:,} rows")
        if self.short_companies:
            lines.append(
                f"Too short: {len(self.short_companies):,} companies "
                f"(e.g. {', '.join(self.short_companies[:3])})"
            )
        if self.gapped_companies:
            lines.append(
                f"Gaps:      {len(self.gapped_companies):,} companies "
                f"(e.g. {', '.join(self.gapped_companies[:3])})"
            )
        if self.duplicate_periods:
            lines.append(f"Duplicates: {self.duplicate_periods:,} repeated periods")
        return "\n".join(lines)


class DataQualityError(ValueError):
    """Input violates an assumption that would silently corrupt results."""


def load_table(
    path: str | Path, *, month_column: str = DEFAULT_MONTH_COLUMN
) -> pd.DataFrame:
    """Read a CSV or Parquet file and parse its month column.

    Args:
        path: File to read. Format is taken from the suffix.
        month_column: Name of the period column to parse as dates.

    Returns:
        The loaded frame, with `month_column` as datetime64.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the suffix is unsupported, the month column is absent,
            or its values cannot be parsed as dates.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported file type '{suffix}'; expected one of "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )

    df = pd.read_parquet(path) if suffix in {".parquet", ".pq"} else pd.read_csv(path)

    if month_column not in df.columns:
        raise ValueError(
            f"month column '{month_column}' not found; available columns: "
            f"{list(df.columns)}"
        )

    # Parse explicitly so bad dates fail here, with the offending values named,
    # rather than becoming NaT and silently dropping rows much later.
    parsed = pd.to_datetime(df[month_column], errors="coerce")
    unparseable = parsed.isna() & df[month_column].notna()
    if unparseable.any():
        examples = df.loc[unparseable, month_column].astype(str).unique()[:5]
        raise ValueError(
            f"{int(unparseable.sum()):,} values in '{month_column}' could not be "
            f"parsed as dates, e.g. {list(examples)}"
        )

    df[month_column] = parsed
    return df


def derive_rate_of_change(
    df: pd.DataFrame,
    *,
    source: str,
    target: str,
    id_column: str = DEFAULT_ID_COLUMN,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Add a within-company rate-of-change column.

    Clipping bounds the ratio: a near-zero denominator otherwise produces
    values large enough to dominate the scaler and drown every other feature.

    Args:
        df: Frame sorted by id then period.
        source: Column to differentiate.
        target: Name for the new column.
        id_column: Grouping key.
        clip: Symmetric bound on the result.

    Returns:
        A copy with `target` added.

    Raises:
        KeyError: If `source` or `id_column` is missing.
    """
    for column in (source, id_column):
        if column not in df.columns:
            raise KeyError(f"column '{column}' not found in input")

    out = df.copy()
    out[target] = (
        out.groupby(id_column)[source].pct_change().fillna(0.0).clip(-clip, clip)
    )
    return out


def validate_frame(
    df: pd.DataFrame,
    *,
    features: list[str],
    time_steps: int,
    id_column: str = DEFAULT_ID_COLUMN,
    month_column: str = DEFAULT_MONTH_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> DataQualityReport:
    """Check an input frame against the pipeline's assumptions.

    Args:
        df: Frame to inspect.
        features: Feature columns that must be present and numeric.
        time_steps: Window length, which sets the minimum history per company.
        id_column: Company identifier column.
        month_column: Period column.
        label_column: Optional ground-truth column.

    Returns:
        A :class:`DataQualityReport`. Structural faults raise instead.

    Raises:
        ValueError: If the frame is empty, required columns are missing, a
            feature is non-numeric, or a feature contains nulls.
    """
    if df.empty:
        raise ValueError("input frame is empty")

    missing = [c for c in [id_column, month_column, *features] if c not in df.columns]
    if missing:
        raise ValueError(
            f"missing required columns: {missing}; available: {list(df.columns)}"
        )

    for feature in features:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(
                f"feature '{feature}' is {df[feature].dtype}, expected numeric"
            )
        n_null = int(df[feature].isna().sum())
        if n_null:
            raise ValueError(
                f"feature '{feature}' contains {n_null:,} nulls; impute or drop "
                "them before scoring"
            )

    labelled = label_column in df.columns

    # A window needs time_steps inputs plus one shifted target month.
    sizes = df.groupby(id_column).size()
    short = sorted(str(c) for c in sizes[sizes <= time_steps].index)

    duplicates = int(df.duplicated(subset=[id_column, month_column]).sum())

    gapped = sorted(
        str(company)
        for company, group in df.groupby(id_column)
        if _has_calendar_gap(group[month_column])
    )

    return DataQualityReport(
        n_rows=len(df),
        n_companies=int(df[id_column].nunique()),
        labelled=labelled,
        short_companies=short,
        gapped_companies=gapped,
        duplicate_periods=duplicates,
    )


def _has_calendar_gap(months: pd.Series) -> bool:
    """Whether a company's observations skip any period."""
    ordered = months.sort_values()
    if len(ordered) < 2:
        return False
    return bool((ordered.diff().dt.days.dropna() > MAX_NORMAL_GAP_DAYS).any())


def prepare_input(
    df: pd.DataFrame,
    *,
    features: list[str],
    time_steps: int,
    id_column: str = DEFAULT_ID_COLUMN,
    month_column: str = DEFAULT_MONTH_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    drop_short: bool = True,
    allow_gaps: bool = False,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate, clean, and sort a frame ready for windowing.

    Companies with too little history are dropped (they contribute no windows
    either way, but dropping them silently is how you end up puzzled about a
    row count). Calendar gaps raise by default, because a window spanning one
    presents non-consecutive months to the model as though they were adjacent
    — the model reads a discontinuity that never happened.

    Args:
        df: Loaded input frame.
        features: Feature columns, in model order.
        time_steps: Window length.
        id_column: Company identifier column.
        month_column: Period column.
        label_column: Optional ground-truth column. Added as all-zero when
            absent, which puts the pipeline in unsupervised mode.
        drop_short: Drop companies with `time_steps` or fewer periods rather
            than raising.
        allow_gaps: Permit non-consecutive months instead of raising.

    Returns:
        The cleaned frame and its quality report.

    Raises:
        DataQualityError: On duplicate periods, or on gaps when `allow_gaps`
            is false, or on short series when `drop_short` is false.
        ValueError: On structural faults, via :func:`validate_frame`.
    """
    report = validate_frame(
        df,
        features=features,
        time_steps=time_steps,
        id_column=id_column,
        month_column=month_column,
        label_column=label_column,
    )

    if report.duplicate_periods:
        raise DataQualityError(
            f"{_count(report.duplicate_periods, 'duplicate')} "
            f"({id_column}, {month_column}) pair(s); each entity-period must "
            "appear exactly once"
        )

    if report.gapped_companies and not allow_gaps:
        raise DataQualityError(
            f"{_count(len(report.gapped_companies), 'entity', 'entities')} have "
            f"gaps in their monthly history "
            f"(e.g. {', '.join(report.gapped_companies[:3])}). "
            "A window spanning a gap treats non-consecutive months as adjacent. "
            "Reindex to a complete monthly grid, or pass allow_gaps to accept "
            "this."
        )

    if report.short_companies and not drop_short:
        raise DataQualityError(
            f"{_count(len(report.short_companies), 'entity', 'entities')} have "
            f"{time_steps} or fewer periods and cannot form a window "
            f"(e.g. {', '.join(report.short_companies[:3])})"
        )

    out = df.copy()

    if not report.labelled:
        # Unsupervised mode: no ground truth, so nothing is known to be bad.
        logger.info(
            "no '%s' column found; running unsupervised (no metrics available)",
            label_column,
        )
        out[label_column] = 0

    n_before = len(out)
    if report.short_companies:
        out = out[~out[id_column].isin(report.short_companies)]
        logger.warning(
            "dropped %s companies with <= %s periods (too short to window)",
            f"{len(report.short_companies):,}",
            time_steps,
        )

    out = out.sort_values([id_column, month_column], kind="stable").reset_index(
        drop=True
    )

    if out.empty:
        raise DataQualityError(
            f"no companies have more than {time_steps} periods; nothing to score"
        )

    final = DataQualityReport(
        n_rows=len(out),
        n_companies=int(out[id_column].nunique()),
        labelled=report.labelled,
        short_companies=report.short_companies,
        gapped_companies=report.gapped_companies,
        duplicate_periods=report.duplicate_periods,
        n_dropped=n_before - len(out),
    )
    return out, final


def load_and_prepare(
    path: str | Path,
    *,
    features: list[str],
    time_steps: int,
    id_column: str = DEFAULT_ID_COLUMN,
    month_column: str = DEFAULT_MONTH_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    rate_of_change_from: str | None = None,
    drop_short: bool = True,
    allow_gaps: bool = False,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load a file and prepare it for the pipeline in one step.

    Args:
        path: CSV or Parquet file.
        features: Feature columns, in model order.
        time_steps: Window length.
        id_column: Company identifier column.
        month_column: Period column.
        label_column: Optional ground-truth column.
        rate_of_change_from: If a feature is a rate of change absent from the
            file, derive it from this source column first.
        drop_short: Drop companies with too little history.
        allow_gaps: Permit non-consecutive months.

    Returns:
        The cleaned frame and its quality report.
    """
    df = load_table(path, month_column=month_column)

    derived = [f for f in features if f not in df.columns and f.endswith("_ROC")]
    if derived and rate_of_change_from:
        df = df.sort_values([id_column, month_column], kind="stable")
        for feature in derived:
            df = derive_rate_of_change(
                df, source=rate_of_change_from, target=feature, id_column=id_column
            )
            logger.info("derived '%s' from '%s'", feature, rate_of_change_from)

    return prepare_input(
        df,
        features=features,
        time_steps=time_steps,
        id_column=id_column,
        month_column=month_column,
        label_column=label_column,
        drop_short=drop_short,
        allow_gaps=allow_gaps,
    )


def frame_from_arrays(
    ids: np.ndarray, months: np.ndarray, **columns: np.ndarray
) -> pd.DataFrame:
    """Build a conforming frame from raw arrays, for tests and small scripts."""
    return pd.DataFrame(
        {DEFAULT_ID_COLUMN: ids, DEFAULT_MONTH_COLUMN: months, **columns}
    )
