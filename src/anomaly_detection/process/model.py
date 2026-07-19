"""LSTM encoder-decoder for next-step sequence prediction."""

from __future__ import annotations

import keras
from keras import layers


def build_lstm_predictor(
    time_steps: int,
    n_features: int,
    *,
    lstm_units: int = 32,
    dense_units: int = 8,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile an LSTM encoder-decoder.

    The encoder LSTM collapses the input window to a single vector, a dense
    bottleneck squeezes it further, and the decoder LSTM expands it back into
    a full-length sequence. Trained on normal data only, the model learns to
    predict ordinary continuations; anomalous windows then produce large
    prediction error, which is what the detector thresholds.

    Args:
        time_steps: Window length.
        n_features: Number of input features.
        lstm_units: Units in the encoder and decoder LSTMs.
        dense_units: Width of the bottleneck. Narrower forces more compression
            and sharper separation, at the cost of reconstruction fidelity.
        learning_rate: Adam learning rate.

    Returns:
        A compiled model mapping ``(batch, time_steps, n_features)`` to a
        tensor of the same shape.

    Raises:
        ValueError: If any dimension is not positive.
    """
    if time_steps < 1:
        raise ValueError(f"time_steps must be >= 1, got {time_steps}")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    if lstm_units < 1:
        raise ValueError(f"lstm_units must be >= 1, got {lstm_units}")
    if dense_units < 1:
        raise ValueError(f"dense_units must be >= 1, got {dense_units}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    inputs = keras.Input(shape=(time_steps, n_features), name="window")
    encoded = layers.LSTM(lstm_units, activation="tanh", name="encoder")(inputs)
    bottleneck = layers.Dense(dense_units, activation="relu", name="bottleneck")(
        encoded
    )
    repeated = layers.RepeatVector(time_steps, name="repeat")(bottleneck)
    decoded = layers.LSTM(
        lstm_units, activation="tanh", return_sequences=True, name="decoder"
    )(repeated)
    outputs = layers.TimeDistributed(
        layers.Dense(n_features, activation="linear"), name="projection"
    )(decoded)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_predictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    return model
