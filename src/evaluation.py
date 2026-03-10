"""
evaluation.py
-------------
Forecasting evaluation metrics for the M5 project.
"""

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (as a percentage)."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (as a percentage, bounded 0–200%)."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model") -> pd.DataFrame:
    """
    Compute all metrics and return a single-row summary DataFrame.

    Parameters
    ----------
    y_true, y_pred : array-like
    model_name     : label for the model column

    Returns
    -------
    pd.DataFrame with columns: model, RMSE, MAE, MAPE, SMAPE
    """
    return pd.DataFrame([{
        "model": model_name,
        "RMSE":  rmse(y_true, y_pred),
        "MAE":   mae(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }])
