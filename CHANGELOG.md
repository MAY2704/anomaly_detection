# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Multivariate anomaly mode** for the simulator
  (`generate_data(..., anomaly_kind="multivariate")`,
  `Config.anomaly_kind`, `anomaly-detection train --anomaly-kind ...`). It
  breaks a lead-lag relationship between turnover and assets while keeping every
  value within its own normal range, so the anomaly is invisible in any single
  series and only detectable jointly. The default remains `"univariate"`, and
  that path is byte-for-byte unchanged.
- `docs/baseline-comparison.md` — a head-to-head of the LSTM against ARIMA and
  matrix-profile baselines on both anomaly kinds, with methodology, numbers, and
  a reproduction recipe. The baseline code and its heavy dependencies are
  intentionally kept out of the repository; the document is the durable record.

## [0.2.0]

Real input files, saved detectors, and a three-layer package structure.

### Added

- **`input/io.py`** — load CSV or Parquet and validate it. Real tables break
  the pipeline's assumptions *silently*, so loading now refuses instead:
  unparseable dates (which otherwise become `NaT` and drop rows much later),
  entities with too little history to form a window (dropped with a warning),
  calendar gaps (a window spanning one presents non-adjacent months as
  adjacent), duplicate periods, and missing or non-numeric columns.
  `--allow-gaps` overrides the gap check.
- **`output/inference.py` and `anomaly-detection score`** — save a
  self-contained bundle (model, scaler, threshold, feature order, window
  length) and apply it to new data without retraining. Feature order comes
  from the bundle, so a differently-ordered input file cannot silently produce
  wrong scores.
- **`output/alerts.py`** — ranked alert tables with per-entity deduplication.
  One anomalous month lands in many overlapping windows, so a single event
  otherwise consumes much of an alert budget; deduplication runs *before* the
  budget, making a budget of N mean N distinct entities. `--no-dedupe` opts
  out.
- **Unsupervised mode.** A label column is now optional. Without one the model
  trains on everything — a low contamination rate leaves the learned notion of
  normal intact — and the run emits ranked alerts in place of metrics.
- `--input-csv` on `train`, configurable `--id-column`, `--month-column`, and
  `--label-column`, and automatic derivation of missing `*_ROC` features.
- `tests/test_import_isolation.py`, which imports every TensorFlow-free module
  in a subprocess with `keras` and `tensorflow` blocked, including a test that
  the blocker itself works so the guard cannot become a silent no-op.

### Fixed

- **Convenience re-exports pulled TensorFlow into modules that do not need
  it.** `process/__init__.py` eagerly imported `model.py`, so importing
  `process.preprocess` — numpy and scikit-learn only — loaded Keras, and
  `output.alerts` did too via the import chain. This passed everywhere
  TensorFlow happened to be installed and only failed in the CI job that
  deliberately omits it. The TensorFlow-backed names are now exported lazily
  (PEP 562), leaving the public API unchanged.
- **Input validation ran after TensorFlow was imported.** A bad path or a
  malformed file now fails in milliseconds instead of after a multi-second
  import, because `run` loads and validates input before importing the model
  or seeding TensorFlow.
- `write_alerts` was missing from `output.__all__`.

### Changed

- **Package restructured into `input/`, `process/`, and `output/` layers.**
  Imports move accordingly: `anomaly_detection.data` →
  `anomaly_detection.input.simulate`, `anomaly_detection.preprocess` →
  `anomaly_detection.process.preprocess`, `anomaly_detection.evaluate` →
  `anomaly_detection.output.evaluate`, and so on.
- **The CLI now requires a subcommand**: `anomaly-detection train` or
  `anomaly-detection score`. A bare invocation prints help and exits 2.
- `prepare_sequences` accepts `id_column`, `label_column`, and `month_column`
  instead of hardcoding them, so the process layer no longer depends on the
  input layer's naming.
- Artifacts gained `metadata.json`; `metrics.json` now records whether the run
  was labelled, and carries `null` metrics when it was not.
- Exit codes distinguish causes: `2` for bad input or configuration, `1` for
  unexpected failure.

## [0.1.0]

First packaged release. The pipeline was reorganised into an installable
package and several correctness bugs affecting reported metrics were fixed.

### Fixed

- **Final-month anomalies were unreachable.** Window generation looped to
  `len(vals) - time_steps - 1`, one short, so the last month of every series
  could never be a prediction target. Anomalies injected at offset `-1`
  existed in the ground truth but could not be detected, depressing recall
  invisibly. Roughly a third of injected anomalies were affected.
- **Train/test leakage.** Splits were taken over individual windows. Because
  sliding windows overlap by `time_steps - 1` months, near-duplicate rows
  landed on both sides of the split and test metrics were optimistic. Splits
  are now taken over companies, stratified on whether a company contains any
  anomaly.
- **Validation set was training data.** The threshold was derived from the
  first 10% of the training sequences, which the model had already fitted.
  In-sample error understates normal error, pushing the cutoff too low.
  Validation is now a genuinely held-out company split.
- **Threshold was set by a single outlier.** Test errors were min-max
  normalised against validation error and cut at 0.5 — halfway between the
  validation min and max — so one extreme validation sample moved the decision
  boundary arbitrarily. The threshold is now a configurable percentile
  (default: 99.9th) of held-out normal error.
- **`seed` did not control anomaly injection.** `generate_data` seeded a local
  generator for the base series but used global `np.random` calls for
  injection, so results depended on the caller's global state.
- **Scaler was fitted on anomalous data.** Injected extremes defined the
  feature range, compressing normal variation and making anomalies look
  ordinary. It is now fitted on normal training windows only.
- **`pandas` deprecation.** `freq="M"` became `freq="ME"`; the old alias was
  removed in pandas 3.0.
- **Mixed Keras namespaces.** `from keras import layers` was combined with
  `tf.keras.Model`, which can yield incompatible class hierarchies under
  Keras 3. Standardised on the `keras` namespace.
- **Duplicated README.** The file contained its entire contents twice.

### Added

- `src/` layout, `pyproject.toml`, and an installable `anomaly-detection`
  console script.
- `anomaly_detection.evaluate` with per-sequence scoring, percentile
  thresholding, and metrics including average precision (PR-AUC) — the
  appropriate headline metric at this class imbalance.
- **Alert budgets** (`--alert-budget N`, `Config.alert_budget`). Flags a fixed
  *count* of highest-scoring windows instead of a fixed fraction. A percentile
  admits false positives in proportion to how much normal data is scored,
  which caps precision at a low base rate; a budget bounds them directly. On
  the baseline run this raised precision from 0.121 to 0.600 and F1 from 0.216
  to 0.750 at unchanged recall, from the same model and the same scores.
  Uses no labels, so it leaks no ground truth.
- **Threshold feasibility warnings.** Before evaluating, the run reports when
  the configured percentile is incompatible with the base rate, quoting the
  resulting precision ceiling and a percentile that would balance the counts.
  After evaluating, it reports when observed false positives exceed the
  anomaly count, and points at `average_precision` to distinguish a ranking
  problem from a threshold problem.
- Artifact persistence: trained model, fitted scaler, config, and metrics are
  written to `--output-dir`.
- Warnings for degenerate evaluations (no anomalies present, too few anomalies
  for stable metrics).
- A `Results` section in the README with measured baseline numbers.
- Configuration validation with actionable error messages.
- A pytest suite, including regression tests for each bug above.
- GitHub Actions CI: lint, type check, a TensorFlow-free matrix across three
  operating systems and three Python versions, a full test job, and a build
  check.
- `LICENSE` (MIT), `CONTRIBUTING.md`, issue and pull request templates.

### Changed

- Config fields renamed from `SCREAMING_CASE` to `snake_case` and given
  documentation and validation. `THRESHOLD` is replaced by
  `threshold_percentile`; `VAL_FRAC` by `val_size` (now a company fraction).
- Default `test_size` reduced from 0.6 to 0.3, with 0.15 for validation,
  leaving 0.55 for training.
- Default `threshold_percentile` raised from 99.0 to 99.9. At the base rate
  this data produces, p99 caps precision at 0.043 by arithmetic alone.
- Data simulation vectorised rather than looping per company.
- `prepare_sequences` returns a `Sequences` dataclass carrying group ids,
  rather than a bare tuple.

[Unreleased]: https://github.com/MAY2704/anomaly_detection/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/MAY2704/anomaly_detection/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MAY2704/anomaly_detection/releases/tag/v0.1.0
