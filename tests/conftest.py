"""Shared fixtures."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from anomaly_detection.config import Config
from anomaly_detection.input.simulate import generate_data


@pytest.fixture
def small_config() -> Config:
    """A configuration small enough to run in seconds."""
    return Config(
        n_companies=60,
        n_months=12,
        n_anomalous_companies=20,
        anomaly_probability=1.0,
        time_steps=3,
        epochs=1,
        batch_size=32,
    )


@pytest.fixture
def small_frame(small_config: Config) -> pd.DataFrame:
    """A simulated frame matching `small_config`."""
    return generate_data(
        small_config.n_companies,
        small_config.n_months,
        seed=small_config.seed,
        n_anomalous_companies=small_config.n_anomalous_companies,
        anomaly_probability=small_config.anomaly_probability,
    )


@pytest.fixture
def tiny_config(small_config: Config) -> Config:
    """The smallest configuration that still trains end to end."""
    return dataclasses.replace(small_config, n_companies=30, n_months=8)
