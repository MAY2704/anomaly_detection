# anomaly_detection

[![CI](https://github.com/MAY2704/anomaly_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/MAY2704/anomaly_detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Find unusual months in time-series data — a turnover that suddenly triples or
collapses, a figure that breaks a company's own pattern.

You give it a table of monthly numbers per entity. It learns what each entity's
normal looks like, then hands you a ranked list of the months that don't fit.
**No labelled examples of "bad" needed** — it learns from normal behaviour
alone.

## What you do with it

1. **Train** a detector on your history (or on built-in demo data).
2. **Score** new data — get a ranked list of the most unusual entities.
3. Investigate the top of the list. Set how many alerts you want and it hands
   you exactly that many.

## Quickstart

```bash
git clone https://github.com/MAY2704/anomaly_detection.git
cd anomaly_detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Try it on the built-in demo data:

```bash
anomaly-detection train --n-companies 500 --epochs 2
```

## Use your own data

Your file (CSV or Parquet) needs an **id** column, a **month** column, and one
or more **numeric** columns to watch. That's it — no labels required.

```csv
CP_ID,MONTH,TURNOVER,ASSETS
ACME_0001,2023-01-31,1043204.7,5007382.1
ACME_0001,2023-02-28,1071553.2,5142201.9
```

Train a detector, then score new data with it:

```bash
# 1. learn what normal looks like, save the detector to bundle/
anomaly-detection train --input-csv history.csv --output-dir bundle/

# 2. point it at new data and ask for the 20 most unusual entities
anomaly-detection score --model-dir bundle/ --input-csv new.csv \
    --alerts-path alerts.csv --alert-budget 20
```

You get a ranked table — highest score is most unusual:

```
3 of 5 alerts above threshold
 rank    group     score  flagged
    1 NEW_0042 13.122672     True
    2 NEW_0005 12.950127     True
    3 NEW_0017 11.286215     True
    4 NEW_0027  0.037036    False
    5 NEW_0004  0.031617    False
```

Different column names? Use `--id-column`, `--month-column`. Training separately
from scoring means you train once and reuse the saved `bundle/` on new data as
often as you like.

## How many alerts do you want?

The single most useful knob. Two ways to decide what gets flagged:

| Option | Use when |
| --- | --- |
| `--alert-budget 20` | You can investigate ~20 cases. Flags the 20 most unusual, full stop. **Start here.** |
| `--threshold-percentile 99.9` | You want a fixed rule that behaves the same on every future batch. |

Why a budget is usually the right choice: real anomalies are rare, often fewer
than 1 in 1,000. A percentile flags a fixed *fraction* of everything, so on a
big dataset it drowns a handful of real problems in hundreds of false alarms. A
budget flags a fixed *count*, so your alert list stays the size you can
actually work through. On the demo data, switching from a percentile to a
budget of 20 took precision from 0.12 to 0.60 — same model, five times fewer
false alarms.

You don't have to work this out yourself — the run warns you when a percentile
would flood you:

```
WARNING  p99.9 flags ~27 of 26,988 normal items but only 12 anomalies exist,
         so precision cannot exceed 0.308 however well the model ranks.
         Raise the percentile (~p99.956 balances them) or set an alert budget.
```

## Bad data is caught before it wastes your time

Messy input usually fails *silently* — the run finishes and the numbers look
fine but aren't. This refuses instead, and tells you what to fix:

| Problem | What happens |
| --- | --- |
| Missing or non-numeric column | Error naming the column |
| Dates it can't read | Error quoting the bad values |
| Entity with too little history | Dropped, with a warning |
| A gap in the monthly history | Error (use `--allow-gaps` to accept it) |
| Duplicate (id, month) rows | Error |

```
error: 1 entity has gaps in their monthly history (e.g. NEW_0000). A window
spanning a gap treats non-consecutive months as adjacent. Reindex to a
complete monthly grid, or pass allow_gaps to accept this.
```

Exit codes for scripting: `0` success, `2` bad input or settings, `1`
unexpected failure.

## What a training run saves

`train --output-dir bundle/` writes a self-contained detector you can reuse:

| File | What it is |
| --- | --- |
| `model.keras` | The trained model |
| `scaler.joblib` | How features were scaled |
| `metadata.json` | Threshold, feature order, window length |
| `metrics.json` | Settings and (if labelled) how well it scored |

The feature order travels with the bundle, so scoring can't silently go wrong
if your new file has columns in a different order.

## From Python

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

## Common settings

Everything is a CLI flag and lives in [`Config`](src/anomaly_detection/config.py).
The ones you'll actually touch:

| Setting | Default | What it does |
| --- | --- | --- |
| `--alert-budget` | off | Flag a fixed number of entities. The knob to reach for first |
| `--time-steps` | `6` | How many months of context the model looks at |
| `--epochs` | `5` | Longer training; raise if scores look noisy |
| `--input-csv` | — | Your data file; omit to use demo data |
| `--allow-gaps` | off | Accept months that aren't consecutive |

---

## How it works

The model is trained to predict each entity's next month from the previous few.
Trained only on ordinary history, it predicts normal continuations well and
unusual ones badly — so **how wrong the prediction is** becomes the anomaly
score.

```
your data  →  6-month windows  →  train on normal  →  score each window  →  rank
```

An entity whose recent months are hard to predict rises to the top of the list.

## Why it's built the way it is

A rare-event detector is easy to get subtly wrong in ways that still produce
good-looking numbers. A few choices here exist specifically to avoid that, and
are worth knowing before you change them:

- **It judges entities it has never seen.** Train, validation, and test split by
  *entity*, not by row, so the reported quality reflects new entities rather
  than ones it already memorised.
- **The alert threshold comes from held-out data**, never from the data the
  model trained on — otherwise it looks more confident than it is.
- **It leads with the right score.** At this rarity, "accuracy" and ROC AUC both
  look near-perfect while the alert list is mostly wrong. The summary leads with
  average precision (PR-AUC), which isn't fooled, and warns when too few
  anomalies are present for any number to be stable.

If you're extending the code, [CONTRIBUTING.md](CONTRIBUTING.md) has the full
reasoning and the rules that keep these properties intact.

## Is an LSTM worth it?

Not always. We ran it head-to-head against classical methods (ARIMA, matrix
profile) on two kinds of anomaly. On a single-variable spike, a training-free
matrix profile nearly ties it. On an anomaly hidden in how variables move
*together*, the LSTM is the only method that works and the baselines drop to
near-random. The full write-up, numbers, and reproduction recipe are in
[docs/baseline-comparison.md](docs/baseline-comparison.md). The simulator can
generate both kinds via `--anomaly-kind {univariate,multivariate}`.

## Good to know

- The built-in dataset is **simulated**, so its headline numbers describe a toy
  problem, not yours. Judge real performance on your own data.
- Splits are by entity ("does this generalise to new entities?"). For a
  strictly "will this catch next month's problem?" question, split your data by
  time instead — train on the past, score the future.
- Very different entity sizes can wash out the small ones, because scaling is
  global. Large real datasets may want per-entity scaling.
- On Windows, training runs on CPU (TensorFlow dropped native Windows GPU
  support after 2.10); use WSL2 for GPU.

## Project layout

Three stages, three folders:

```
src/anomaly_detection/
├── input/     # load and validate data (real files or simulated)
├── process/   # window, split, and train the model
└── output/    # score, rank alerts, save/load a detector
```

## For contributors

```bash
pip install -e ".[dev]"
pytest -m "not slow"    # fast checks, no TensorFlow needed
pytest                  # everything
ruff check . && mypy    # lint and types
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
