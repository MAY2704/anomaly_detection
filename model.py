from typing import Any
import tensorflow as tf
from keras import layers

def build_lstm_predictor(ts: int, nfeat: int, lstm_units: int = 32, dense_units: int = 8,
                         learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build a simple LSTM sequence-to-sequence predictor.

    Args:
        ts: time steps (sequence length).
        nfeat: number of features.
        lstm_units: units in LSTM encoder/decoder.
        dense_units: units in bottleneck dense layer.
        learning_rate: optimizer learning rate.

    Returns:
        Compiled tf.keras.Model.
    """
    inp = layers.Input((ts, nfeat))
    e = layers.LSTM(lstm_units, activation="tanh")(inp)
    b = layers.Dense(dense_units, activation="relu")(e)
    r = layers.RepeatVector(ts)(b)
    d = layers.LSTM(lstm_units, activation="tanh", return_sequences=True)(r)
    out = layers.TimeDistributed(layers.Dense(nfeat, activation="linear"))(d)

    model = tf.keras.Model(inputs=inp, outputs=out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model