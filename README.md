# anomaly_detection

[![CI](https://github.com/MAY2704/anomaly_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/MAY2704/anomaly_detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Unsupervised anomaly detection on simulated corporate time series, using an
LSTM encoder-decoder trained only on normal behaviour.

The model learns to predict the next month of a company's financials. Trained
exclusively on normal sequences, it predicts ordinary continuations well and
anomalous ones badly — so prediction error becomes the anomaly score.

## How it works

```
simulate  →  window  →  split by company  →  train on normal  →  score  →  threshold
20k series   6-month     55/15/30              LSTM enc-dec      per-seq   p99 of
× 36 months  windows     train/val/test        (normal only)     MSE       held-out
```

1. **Simulate.** 20,000 companies × 36 months of turnover and assets, with
   sharp spikes or collapses injected into a small number of them.
2. **Window.** Sliding 6-month windows. Each window's target is itself shifted
   forward one month, making this next-step prediction rather than
   reconstruction.
3. **Split by company.** Whole companies go to train, validation, or test —
   never split individual windows (see [Methodology](#methodology)).
4. **Train on normal only.** Anomalies are withheld so they stay surprising.
5. **Score and threshold.** Per-sequence MSE is the anomaly score. The cutoff
   is a high percentile of error on *held-out normal* data.

## Results

A run at 3,000 companies for 3 epochs (`seed=42`), giving 27,000 test windows
containing 12 anomalies — a base rate of 1 in 2,250:

| Threshold | Precision | Recall | F1 | False positives |
| --- | --- | --- | --- | --- |
| p99 | 0.038 | 1.000 | 0.073 | 307 |
| p99.9 (default) | 0.121 | 1.000 | 0.216 | 87 |
| **`--alert-budget 20`** | **0.600** | **1.000** | **0.750** | **8** |

Every row scores **PR-AUC 0.641** and **ROC AUC 0.9998**. Same model, same
scores, same ranking — only the cutoff moved, and precision moved 5×.

Two things follow.

**ROC AUC is not informative here.** 0.9998 suggests a near-perfect detector
while precision says 88% of its flags are wrong. Average precision is not
fooled, which is why it leads the summary.

**Precision was never limited by the model.** The scores support precision
0.667 at full recall (best F1 0.800). A percentile flags a fixed *fraction* of
normal data — at p99.9 that is ~27 false positives per 27,000 windows against
12 real anomalies, capping precision at 0.308 before the model does anything.
An alert budget fixes the *count* instead, which is what actually bounds false
positives at a low base rate.

The run tells you when this is happening rather than making you work it out:

```
WARNING  p99.9 flags ~27 of 26,988 normal items but only 12 anomalies exist,
         so precision cannot exceed 0.308 however well the model ranks.
         Raise the percentile (~p99.956 balances them) or set an alert budget.
```

### Choosing a threshold

| Use | When |
| --- | --- |
| `--alert-budget N` | You have a fixed investigation capacity. Bounds false positives directly and is insensitive to how much normal data is scored. |
| `--threshold-percentile P` | You want a fixed decision rule that transfers to new data unchanged. Pick `P` against the base rate, not out of habit. |

The budget reads the scores it thresholds, so it uses **no labels** and leaks
no ground truth — but the cutoff depends on the batch, so it is not a portable
decision rule. The percentile is portable but must be matched to the base
rate.

A caveat on very high percentiles: the estimate comes from held-out normal
error, and at p99.9 only ~14 of 13,525 validation windows sit above the
cutoff. Tail quantiles from that few points are noisy — the measured
false-positive rate came out at 0.32% against a nominal 0.1%. Past roughly
p99.9 you need a much larger validation split for the number to mean anything,
which is a further argument for the budget.

## Install

```bash
git clone https://github.com/MAY2704/anomaly_detection.git
cd anomaly_detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

For development tooling, `pip install -e ".[dev]"`.

## Usage

Two verbs: `train` fits a detector and saves a reusable bundle, `score` applies
that bundle to new data.

```bash
# train on simulated data
anomaly-detection train --n-companies 500 --epochs 2

# train on your own data (no labels required)
anomaly-detection train --input-csv data.csv --output-dir bundle/

# score new data with the saved bundle
anomaly-detection score --model-dir bundle/ --input-csv new.csv \
    --alerts-path alerts.csv --alert-budget 50
```

`train --output-dir` receives a self-contained bundle:

| File | Contents |
| --- | --- |
| `model.keras` | Trained model |
| `scaler.joblib` | Fitted feature scaler |
| `metadata.json` | Threshold, feature order, window length |
| `metrics.json` | Config and metrics (metrics are `null` when unlabelled) |

The feature order lives in the bundle, so scoring cannot silently mismatch the
column order of an input file.

## Using your own data

Your file needs an id column, a period column, and your feature columns. **A
label column is optional** — without one the run goes unsupervised: it trains
on everything, skips metrics, and emits ranked alerts.

```csv
CP_ID,MONTH,TURNOVER,ASSETS
ACME_0001,2023-01-31,1043204.7,5007382.1
ACME_0001,2023-02-28,1071553.2,5142201.9
```

```bash
anomaly-detection train --input-csv data.csv --output-dir bundle/
anomaly-detection score --model-dir bundle/ --input-csv new.csv \
    --alerts-path alerts.csv --alert-budget 20
```

```
3 of 5 alerts above threshold
 rank    group     score  flagged
    1 NEW_0042 13.122672     True
    2 NEW_0005 12.950127     True
    3 NEW_0017 11.286215     True
    4 NEW_0027  0.037036    False
    5 NEW_0004  0.031617    False
```

Column names are configurable with `--id-column`, `--month-column`, and
`--label-column`. Any `*_ROC` feature absent from the file is derived
automatically.

### Input is validated, loudly

Real tables break the pipeline's assumptions in ways that are *silent* rather
than loud — the results still look plausible. Loading refuses instead:

| Problem | Why it matters | Behaviour |
| --- | --- | --- |
| Missing or non-numeric column | — | Error naming the column |
| Unparseable dates | Become `NaT` and drop rows much later | Error quoting the bad values |
| Entity with ≤ `time_steps` periods | Contributes zero windows | Dropped, with a warning |
| Gap in the monthly history | A window spans it, presenting non-adjacent months as adjacent | Error; `--allow-gaps` to override |
| Duplicate (id, period) rows | — | Error |

```
error: 1 entity has gaps in their monthly history (e.g. NEW_0000). A window
spanning a gap treats non-consecutive months as adjacent. Reindex to a
complete monthly grid, or pass allow_gaps to accept this.
```

Exit codes: `0` success, `2` bad input or configuration, `1` unexpected failure.

### As a library

```python
from anomaly_detection import Config
from anomaly_detection.process.train import run

artifacts = run(Config(n_companies=1_000, epochs=3), save=False)
print(artifacts.result.summary())
```

Scoring with a saved bundle:

```python
from anomaly_detection.input import load_and_prepare
from anomaly_detection.output import build_alerts, load_artifacts, score_frame

bundle = load_artifacts("bundle/")
df, report = load_and_prepare(
    "new.csv", features=list(bundle.features), time_steps=bundle.time_steps
)
groups, scores, _ = score_frame(bundle, df)
alerts = build_alerts(groups, scores, threshold=bundle.threshold, budget=20)
```

## Configuration

Every setting lives in [`Config`](src/anomaly_detection/config.py), is
validated on construction, and is exposed as a CLI flag. The ones that matter
most:

| Setting | Default | Notes |
| --- | --- | --- |
| `n_companies` | `20000` | Lower it for fast iteration |
| `time_steps` | `6` | Window length in months |
| `test_size` / `val_size` | `0.3` / `0.15` | Fractions of *companies*, not windows |
| `threshold_percentile` | `99.9` | Expected false-positive rate; higher → fewer false positives |
| `alert_budget` | `None` | Flag a fixed *count* instead. Overrides the percentile |
| `n_anomalous_companies` | `40` | An absolute count, **not** a proportion of `n_companies` |
| `dense_units` | `8` | Bottleneck width; narrower sharpens separation |
| `epochs` | `5` | |

## Methodology

Four properties of this pipeline are load-bearing. Each addresses a mistake
that produces good-looking but meaningless numbers.

**Splits partition companies, not windows.** Sliding windows overlap by
`time_steps - 1` months. Splitting windows at random puts near-duplicates on
both sides of the boundary, and the test set effectively grades the model on
data it has already seen. Splits are also stratified on whether a company
contains any anomaly, so rare positives reach all three sets.

**The threshold comes from held-out data.** Error measured on data the model
trained on is in-sample and understates normal error, dragging the cutoff down
and flooding the results with false positives. Validation is a separate
company split.

**A percentile, not a normalised midpoint.** Min-max normalising test error
against validation error and cutting at the midpoint lets a single extreme
validation sample define the decision boundary. A percentile is stable, and it
has a direct interpretation: p99 means roughly 1% of normal windows are
expected to be flagged.

**The threshold is chosen against the base rate, not by convention.** A
percentile that ignores how rare anomalies are caps precision by arithmetic
before the model contributes anything; see [Results](#results). The run warns
when the configured percentile is incompatible with the anomalies present, and
`--alert-budget` bounds the false-positive count directly.

**PR-AUC is the headline metric.** With a handful of anomalies among hundreds
of thousands of windows, a model that flags nothing scores over 99.9%
accuracy, and as the [results](#results) show, ROC AUC reads 0.9998 where
precision is 0.12. Average precision is not fooled, so it is what the summary
leads with and what pull requests should compare.

The run also warns when an evaluation is degenerate: no anomalies present in
the test split, or too few for metrics to be stable across seeds.

## Development

```bash
pip install -e ".[dev]"
pre-commit install

pytest -m "not slow"    # fast suite, no TensorFlow required
pytest                  # everything
ruff check . && mypy    # lint and type check
```

The data, windowing, splitting, and metrics modules import no TensorFlow, so
the fast suite runs in seconds and CI can test them across three operating
systems without a 600 MB install. Please keep it that way.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Project layout

Three layers, matching the three stages of a run:

```
src/anomaly_detection/
├── config.py          # validated settings
├── cli.py             # train / score subcommands
├── input/             # getting data in, and proving it is usable
│   ├── io.py          #   load CSV/Parquet, validate, derive features
│   └── simulate.py    #   synthetic data with injected anomalies
├── process/           # turning data into a trained model
│   ├── preprocess.py  #   windowing, entity-wise splitting, scaling
│   ├── model.py       #   LSTM encoder-decoder
│   └── train.py       #   end-to-end pipeline
└── output/            # turning scores into decisions
    ├── evaluate.py    #   thresholds, metrics, feasibility checks
    ├── inference.py   #   save/load a bundle, score new data
    └── alerts.py      #   ranked alerts with per-entity dedupe
```

Both input sources produce the same frame contract, so the simulator and a
real file are interchangeable downstream.

## Limitations

- The bundled dataset is **simulated**. Anomalies are injected with a known
  generative process, which is considerably easier than the real thing; the
  quoted metrics describe that simulation, not your data.
- Splits partition *entities*, which answers "does this generalise to new
  entities". For a production question — "will this work next month" — split
  by time instead: train on the past, score the future.
- The scaler is fitted globally, so an entity an order of magnitude smaller
  than its peers is squeezed into a narrow sub-range. Real financials may need
  per-entity normalisation or a `RobustScaler`.
- Anomalies are only injected in the last three months of a series, so the
  detector is never tested on anomalies early in a history.
- `ASSETS` is generated before injection, so it does not reflect an anomalous
  turnover reading. Only `TURNOVER` and its rate of change carry the signal.
- `n_anomalous_companies` is an absolute count, so raising `n_companies` alone
  makes anomalies **rarer**, not more plentiful. Scale both together, or the
  problem quietly gets harder as you scale up.
- With few positives, threshold-dependent metrics vary noticeably across
  seeds. Compare PR-AUC, and average over seeds when it matters. The run warns
  when the test split holds fewer than 30 anomalies.
- TensorFlow dropped native Windows GPU support after 2.10, so Windows runs
  are CPU-only. Use WSL2 for GPU.

## License

[MIT](LICENSE)
