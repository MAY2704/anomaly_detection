# anomaly_detection

Simple LSTM-based time-series anomaly detection example.

## Overview
- Simulates corporate time series (TURNOVER, ASSETS, TURNOVER_ROC) with injected anomalies.
- Prepares sliding sequences, trains an LSTM sequence-to-sequence predictor on normal sequences.
- Uses reconstruction MSE as anomaly score and evaluates detection performance.

## Files
- `config.py` - hyperparameters and configuration (seeds, model, training, threshold).
- `data.py` - data simulation and anomaly injection.
- `preprocess.py` - sequence creation, scaling, train/test split.
- `model.py` - LSTM encoder-decoder builder.
- `train.py` - main training/evaluation script.
- `requirements.txt` - Python dependencies.

## Quickstart (dev container / Ubuntu 24.04)
1. Create and activate venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r /workspaces/anomaly_detection/requirements.txt
```

3. Run training (preferred as package):
```bash
python -m anomaly_detection.train
```
Alternative (if running `train.py` directly): ensure `/workspaces/anomaly_detection` is on `PYTHONPATH` or run from the package root.

## Development tips
- Reduce `N_CP` in `config.py` for faster iteration.
- Use the validation MSE distribution (e.g., 95th percentile) to derive a threshold instead of a fixed value in `config.py`.
- Save model and scaler after training for inference.

## Notes
- The workspace includes a package marker (`__init__.py`) so modules can be run via `python -m anomaly_detection.train`.
- TensorFlow will print device/GPU info; GPU is used only if CUDA drivers are present.

```# filepath: /workspaces/anomaly_detection/README.md
# anomaly_detection

Simple LSTM-based time-series anomaly detection example.

## Overview
- Simulates corporate time series (TURNOVER, ASSETS, TURNOVER_ROC) with injected anomalies.
- Prepares sliding sequences, trains an LSTM sequence-to-sequence predictor on normal sequences.
- Uses reconstruction MSE as anomaly score and evaluates detection performance.

## Files
- `config.py` - hyperparameters and configuration (seeds, model, training, threshold).
- `data.py` - data simulation and anomaly injection.
- `preprocess.py` - sequence creation, scaling, train/test split.
- `model.py` - LSTM encoder-decoder builder.
- `train.py` - main training/evaluation script.
- `requirements.txt` - Python dependencies.

## Quickstart (dev container / Ubuntu 24.04)
1. Create and activate venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r /workspaces/anomaly_detection/requirements.txt
```

3. Run training (preferred as package):
```bash
python -m anomaly_detection.train
```
Alternative (if running `train.py` directly): ensure `/workspaces/anomaly_detection` is on `PYTHONPATH` or run from the package root.

## Development tips
- Reduce `N_CP` in `config.py` for faster iteration.
- Use the validation MSE distribution (e.g., 95th percentile) to derive a threshold instead of a fixed value in `config.py`.
- Save model and scaler after training for inference.

## Notes
- The workspace includes a package marker (`__init__.py`) so modules can be run via `python -m anomaly_detection.train`.
- TensorFlow will print device/GPU info; GPU is used only if CUDA drivers are present.
