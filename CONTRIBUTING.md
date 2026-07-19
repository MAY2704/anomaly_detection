# Contributing

Thanks for taking the time to contribute.

## Getting set up

```bash
git clone https://github.com/MAY2704/anomaly_detection.git
cd anomaly_detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

## Running the checks

```bash
pytest -m "not slow"    # fast suite, no TensorFlow needed
pytest                  # everything, including model and end-to-end tests
ruff check .            # lint
ruff format .           # format
mypy                    # type check
```

CI runs all of the above. The fast suite deliberately avoids importing
TensorFlow so the cross-platform matrix stays quick — please keep the `input/`
and `output/evaluate.py` and `output/alerts.py` modules free of TensorFlow
imports. Only `process/model.py`, `process/train.py`, and
`output/inference.py` may need it.

## Package structure

Three layers, and the dependency direction matters:

- **`input/`** — obtaining data and proving it is usable. Knows nothing about
  models.
- **`process/`** — windowing, scaling, training. Takes column names as
  parameters rather than importing them, so it does not depend on how input
  happens to be named.
- **`output/`** — thresholds, metrics, alerts, inference.

New code should sit in whichever layer owns its concern, and should not make
a lower layer import a higher one.

## Making changes

1. Branch off `main`.
2. Make your change, with tests.
3. Make sure the checks above pass.
4. Add a `CHANGELOG.md` entry under `## [Unreleased]`.
5. Open a pull request describing what changed and why.

## Testing expectations

- New behaviour needs a test.
- Bug fixes need a **regression test** whose docstring states what broke and
  why it mattered. Several existing tests follow this pattern — see
  `tests/test_preprocess.py`. The point is that a future refactor cannot
  quietly undo the fix.
- Mark anything needing TensorFlow or a training run with `@pytest.mark.slow`.

## Methodology notes

This project detects rare events, and a few properties are load-bearing.
Please don't undo them without a good reason:

- **Split by company, never by window.** Sliding windows overlap, so
  window-level splits leak near-duplicate rows across the boundary.
- **Derive the threshold from held-out data.** A threshold computed on
  training data understates normal error and floods results with false
  positives.
- **Fit the scaler on normal data only.** Anomalies are extreme by
  construction; letting them set the feature range hides them. (Unsupervised
  runs cannot do this, and accept the contamination knowingly.)
- **Fail loudly on bad input.** Every check in `input/io.py` guards a failure
  that is otherwise *silent* — the run completes and the numbers look fine.
  If you add an input assumption, add the check with it.
- **Report PR-AUC.** At roughly 1-in-2000 positives, accuracy is meaningless
  and ROC AUC is misleadingly flattering — the baseline run scores ROC AUC
  0.9998 at precision 0.121.
- **Judge the threshold separately from the model.** A percentile flags a
  fixed fraction of normal data, so at a low base rate it admits more false
  positives than there are anomalies and caps precision regardless of ranking
  quality. If precision looks bad, check `average_precision` first: if it is
  healthy, the cutoff is the problem, not the model. `--alert-budget` bounds
  the flag count directly.

## Style

- Google-style docstrings on public functions.
- Type hints throughout `src/`.
- Comments explain *why*, not *what*.

## Reporting bugs

Open an issue with the bug report template. For anything affecting detection
quality, please include the config you ran and the contents of
`runs/metrics.json`.
