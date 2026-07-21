# Baseline comparison: is the LSTM worth its weight?

An LSTM is heavier and slower than classical time-series methods. This note
records a head-to-head we ran to answer a simple question — does it earn that
cost? — and the honest answer: **it depends entirely on the kind of anomaly.**

The comparison code and its dependencies are deliberately **not** part of this
repository (they pull in heavy libraries only a one-off study needs). This
document is the durable record of what we did and what we found; a reproduction
recipe is at the end.

## What we compared

| Method | Idea | Sees temporal structure? | Sees features jointly? |
| --- | --- | --- | --- |
| **LSTM** (this repo) | Predict next month from the past window; large error = anomaly | Yes | **Yes** |
| **ARIMA** | Same predict-and-measure-the-residual idea, one series at a time | Yes | No |
| **Matrix profile** | Flag a subsequence unlike anything else in the series' own history | Yes | No |

Two methods we considered and left out:

- **Isolation forest** scores each record on its own. It has no memory, so it
  cannot tell that a value is only strange *given what came before it* — the
  whole point of time series. Ruled out.
- **Prophet** is built for a single series with strong seasonality, which
  monthly financials rarely have, and it is a heavy, fragile dependency. Left
  as possible future work.

Matrix profile uses the **non-normalized** distance (`stumpy.aamp`), not the
canonical z-normalized `stumpy.stump`. The z-normalized version compares
subsequence *shape* and is blind to scale, so it scores a pure amplitude spike
near random. Using it would have been a straw man; the non-normalized distance
gives matrix profile its best honest shot.

## How we measured

Everything was scored on the **same windows, the same entity-wise train/
validation/test split, and the same labels**, so any difference is the method,
not the harness.

- **PR-AUC** (area under the precision-recall curve) is the headline. It is
  threshold-free and, unlike accuracy or ROC AUC, is not flattered by the
  extreme class imbalance of rare-event detection.
- **Precision and recall** are reported at a shared **alert budget** — each
  method may flag the same fixed number of highest-scoring windows — which is
  the fairest fixed operating point because it holds the alert count constant
  across methods.
- Results are **averaged over 3 seeds**. Single-seed numbers swing noticeably
  at these anomaly counts.

Run configuration: 1,500 companies × 40 months, features `TURNOVER` and
`ASSETS`, window length 4, 10 training epochs, alert budget 50. The two anomaly
kinds below are the simulator's [`anomaly_kind`](../src/anomaly_detection/config.py)
setting — the *only* thing that changed between the two tables.

## Result 1 — a single-variable anomaly (`anomaly_kind="univariate"`)

A turnover figure suddenly spikes or collapses. The break is visible in one
number.

| Method | PR-AUC | Precision@50 | Recall@50 |
| --- | --- | --- | --- |
| **LSTM** | **0.97** | 0.71 | 1.00 |
| Matrix profile | 0.89 | 0.62 | 0.88 |
| ARIMA | 0.60 | 0.45 | 0.64 |

All three detect it. The LSTM leads, but a **training-free matrix profile is
close behind** — and how large the LSTM's edge is depends on how long you train
and how many nuisance features you feed it (with fewer epochs and a noisier
feature set we have seen the two draw level). For single-variable anomalies,
the simple method is a defensible choice; the deep model is not buying much.

## Result 2 — a relationship anomaly (`anomaly_kind="multivariate"`)

Assets normally track *last* month's turnover. In the anomaly they move the
wrong way, while every individual value stays within its own normal range.
Turnover alone: nothing. Assets alone: nothing. Only jointly, over time, is
anything wrong. (The anomalous values sit ~1.2–1.5σ within each series' own
history — a per-series detector has nothing to flag.)

| Method | PR-AUC | Precision@50 | Recall@50 |
| --- | --- | --- | --- |
| **LSTM** | **0.69** | 0.71 | 0.66 |
| Matrix profile | 0.03 | 0.03 | 0.03 |
| ARIMA | 0.02 | 0.03 | 0.03 |

The single-series methods **collapse to near-random** — they never reliably
flag a real case, because there is nothing to see in the one series each of
them watches. The LSTM catches roughly two-thirds of anomalies that are, by
construction, invisible to any single-variable method.

Why the LSTM can and the others cannot: because it predicts each month from a
window containing *both* series, it learns that this month's assets should
follow last month's turnover. The conditional relationship is tight (small
conditional noise) inside a wide marginal spread, so the broken value is a huge
*conditional* surprise while remaining an unremarkable *marginal* one. ARIMA
and matrix profile only ever see one series' marginal behaviour, so the break
is invisible to them.

## The takeaway

The LSTM is not better in general. On easy, single-variable anomalies a matrix
profile ties or nearly ties it for a fraction of the cost. Its value is
specific and real: it is the only method here that can see anomalies hiding in
how several variables move together — and on those, the simpler tools are not
merely worse, they are blind. **Choose the tool to match the anomaly you are
worried about.**

A caveat worth keeping in view: these anomalies are *simulated*, with a known
generative process. The multivariate result is robust across seeds and
configurations; the size of the LSTM's univariate edge is not. Real data is the
proper next test.

## Reproducing this

The baselines are not in the repo. To reproduce:

```bash
pip install -e .            # this package
pip install statsmodels stumpy   # the baselines (heavy; not a project dependency)
```

Then, for each `anomaly_kind`:

1. Generate data with
   [`generate_data(..., anomaly_kind=...)`](../src/anomaly_detection/input/simulate.py),
   or `anomaly-detection train --anomaly-kind {univariate,multivariate}`.
2. Window and split with `prepare_sequences` and `split_and_scale`, train the
   LSTM (`build_lstm_predictor`), and score the test set with `sequence_errors`.
3. Score the same test windows with an ARIMA one-step residual and a
   non-normalized matrix-profile distance per feature, aligned to the same
   windows.
4. Compare with `evaluate_scores` at a shared alert budget, and with
   `average_precision_score` for PR-AUC.
