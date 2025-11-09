import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Random seeds
    NP_SEED: int = 42
    TF_SEED: int = 42

    # Data
    N_CP: int = 20000
    N_MONTHS: int = 36
    FEATURES: tuple = ("TURNOVER", "ASSETS", "TURNOVER_ROC")
    TIME_STEP: int = 6

    # Model / Training
    LSTM_UNITS: int = 32
    DENSE_UNITS: int = 8
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 5
    BATCH_SIZE: int = 64

    # Anomaly decision (0..1). Original code used 1.05 which is incorrect for normalized scores.
    THRESHOLD: float = 0.5

    # Misc
    TEST_SIZE: float = 0.6
    VAL_FRAC: float = 0.1

CFG = Config()