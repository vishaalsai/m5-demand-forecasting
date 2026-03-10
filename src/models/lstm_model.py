"""
lstm_model.py
-------------
LSTM sequence model for CA_1 daily demand forecasting using TensorFlow/Keras.
"""

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    raise ImportError("Install tensorflow: pip install tensorflow")


def create_sequences(series: np.ndarray, look_back: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D time series into (X, y) supervised learning arrays.

    Parameters
    ----------
    series    : 1-D numpy array of scaled sales values
    look_back : number of past time steps used as input features

    Returns
    -------
    X : shape (n_samples, look_back, 1)
    y : shape (n_samples,)
    """
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back: i])
        y.append(series[i])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)


def build_lstm_model(look_back: int = 28, units: int = 64, dropout: float = 0.2) -> "tf.keras.Model":
    """
    Build a two-layer LSTM model.

    Parameters
    ----------
    look_back : input sequence length
    units     : LSTM hidden units per layer
    dropout   : dropout rate between layers

    Returns
    -------
    Compiled Keras model (not yet trained)
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    print(model.summary())
    return model


def train_lstm(
    model: "tf.keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> "tf.keras.callbacks.History":
    """
    Train the LSTM model with early stopping.

    Returns
    -------
    Keras History object
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


def predict_lstm(model: "tf.keras.Model", X_test: np.ndarray, scaler=None) -> np.ndarray:
    """
    Generate predictions and optionally inverse-transform scaling.

    Parameters
    ----------
    model   : trained Keras model
    X_test  : shaped (n_samples, look_back, 1)
    scaler  : sklearn MinMaxScaler or None

    Returns
    -------
    1-D numpy array of predicted values in original scale
    """
    preds = model.predict(X_test).flatten()
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    return np.maximum(preds, 0)
