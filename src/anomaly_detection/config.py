"""Configuration for the anomaly detection pipeline.

All tunable behaviour lives here. :class:`Config` is frozen so a run cannot
mutate its own settings partway through; build a modified copy with
:func:`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_TRAIN_FRACTION = 1e-6
"""Smallest usable training fraction.

Compared against a tolerance rather than zero: ``1.0 - 0.7 - 0.3`` evaluates
to ``4.16e-17`` in floating point, so an exact ``> 0.0`` test would accept a
split that leaves no training data at all.
"""


@dataclass(frozen=True)
class Config:
    """Hyperparameters and pipeline settings.

    Attributes are grouped by pipeline stage. :meth:`validate` runs
    automatically on construction and raises on incoherent combinations.
    """

    # --- Reproducibility -------------------------------------------------
    seed: int = 42
    """Master seed. Seeds numpy, TensorFlow, and data simulation alike."""

    # --- Input -----------------------------------------------------------
    input_path: str | None = None
    """Read a CSV or Parquet file instead of simulating data.

    When unset the pipeline simulates its own data. When set, the file must
    carry the id and period columns plus every entry in `features`; a label
    column is optional, and its absence puts the run in unsupervised mode.
    """

    id_column: str = "CP_ID"
    """Column identifying one entity's series — the grouping key for splits."""

    month_column: str = "MONTH"
    """Period column, used for chronological ordering."""

    label_column: str = "IS_TRUE_ANOMALY"
    """Ground-truth column. Optional for real input."""

    rate_of_change_from: str | None = "TURNOVER"
    """Derive any missing ``*_ROC`` feature from this column at load time."""

    allow_gaps: bool = False
    """Accept non-consecutive periods instead of raising.

    A window spanning a gap presents non-adjacent months to the model as
    though they were adjacent, so this is off by default.
    """

    # --- Data simulation -------------------------------------------------
    n_companies: int = 20_000
    """Number of simulated companies (each contributes one time series)."""

    n_months: int = 36
    """Months of history per company."""

    n_anomalous_companies: int = 40
    """How many companies are eligible to receive injected anomalies."""

    anomaly_offsets: tuple[int, ...] = (-1, -2, -3)
    """Month offsets (from the end of each series) eligible for injection."""

    anomaly_probability: float = 0.5
    """Chance an eligible slot actually receives an anomaly."""

    anomaly_kind: str = "univariate"
    """Which kind of anomaly the simulator injects.

    - ``"univariate"``: a turnover spike or collapse — visible in one variable.
    - ``"multivariate"``: a broken lead-lag relationship. Assets normally track
      last month's turnover; the anomaly makes them move the wrong way while
      every value stays within its own normal range, so the anomaly is
      invisible in any single series and only shows up in how the variables
      move together. Best paired with ``features=("TURNOVER", "ASSETS")``.
    """

    features: tuple[str, ...] = ("TURNOVER", "ASSETS", "TURNOVER_ROC")
    """Feature columns fed to the model, in order."""

    # --- Windowing -------------------------------------------------------
    time_steps: int = 6
    """Sliding-window length. The model sees `time_steps` months at a time."""

    # --- Splitting -------------------------------------------------------
    test_size: float = 0.3
    """Fraction of *companies* held out for testing."""

    val_size: float = 0.15
    """Fraction of *companies* held out for validation and thresholding."""

    # --- Model / training ------------------------------------------------
    lstm_units: int = 32
    dense_units: int = 8
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 64

    # --- Anomaly decision ------------------------------------------------
    threshold_percentile: float = 99.9
    """Percentile of held-out *normal* validation error used as the cutoff.

    Sequences scoring above this error are flagged. Deriving it from a
    held-out split rather than fixing it in config lets it adapt to the error
    scale the trained model actually produces.

    The percentile is the expected false-positive rate on normal data, so it
    should be chosen against the base rate. Anomalies here are rarer than
    1 in 2,000; at p99 the ~1% of normal windows flagged swamp the true
    positives. Measured at 3,000 companies, moving p99 → p99.9 raised
    precision from 0.038 to 0.121 at unchanged recall.
    """

    alert_budget: int | None = None
    """Flag this many highest-scoring windows instead of using a percentile.

    An alert budget fixes the *count* of flagged items rather than the
    fraction, which is what actually bounds false positives when anomalies are
    rare: at a base rate of 1 in 2,250, even a correct p99.9 admits ~27 false
    positives per 27,000 windows and caps precision near 0.31. Set this to the
    number of alerts you can realistically investigate.

    Takes precedence over `threshold_percentile` when set.
    """

    # --- Output ----------------------------------------------------------
    output_dir: str = "runs"
    """Directory for the saved model, scaler, and metrics."""

    alerts_path: str | None = None
    """Write a ranked alert table here. Format follows the suffix."""

    dedupe_alerts: bool = True
    """Collapse alerts to the highest-scoring window per entity.

    One anomalous month lands in many overlapping windows, so without this a
    single event can consume much of an alert budget.
    """

    def __post_init__(self) -> None:
        """Validate on construction so bad settings fail fast."""
        self.validate()

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return len(self.features)

    @property
    def train_size(self) -> float:
        """Fraction of companies remaining for training."""
        return 1.0 - self.test_size - self.val_size

    def validate(self) -> None:
        """Raise :class:`ValueError` if the settings cannot produce a run."""
        if self.n_companies < 1:
            raise ValueError(f"n_companies must be >= 1, got {self.n_companies}")

        # A window needs `time_steps` inputs plus one shifted target month.
        if self.n_months <= self.time_steps:
            raise ValueError(
                f"n_months ({self.n_months}) must exceed time_steps "
                f"({self.time_steps}) or no windows can be built"
            )

        if not self.features:
            raise ValueError("features must not be empty")

        if self.anomaly_kind not in ("univariate", "multivariate"):
            raise ValueError(
                "anomaly_kind must be 'univariate' or 'multivariate', got "
                f"{self.anomaly_kind!r}"
            )

        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")

        if not 0.0 < self.val_size < 1.0:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}")

        if self.train_size < MIN_TRAIN_FRACTION:
            raise ValueError(
                f"test_size + val_size ({self.test_size + self.val_size}) "
                "must leave a positive training fraction"
            )

        if self.n_anomalous_companies > self.n_companies:
            raise ValueError(
                f"n_anomalous_companies ({self.n_anomalous_companies}) exceeds "
                f"n_companies ({self.n_companies})"
            )

        if not 0.0 <= self.anomaly_probability <= 1.0:
            raise ValueError(
                f"anomaly_probability must be in [0, 1], got {self.anomaly_probability}"
            )

        # Offsets index backwards from the end of a series.
        for off in self.anomaly_offsets:
            if off >= 0:
                raise ValueError(f"anomaly_offsets must be negative, got {off}")
            if abs(off) > self.n_months:
                raise ValueError(
                    f"anomaly offset {off} is outside a {self.n_months}-month series"
                )

        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ValueError(
                f"threshold_percentile must be in [0, 100], got "
                f"{self.threshold_percentile}"
            )

        if self.alert_budget is not None and self.alert_budget < 1:
            raise ValueError(f"alert_budget must be >= 1, got {self.alert_budget}")

        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")


CFG = Config()
"""Default configuration instance, for convenience in scripts and notebooks."""
