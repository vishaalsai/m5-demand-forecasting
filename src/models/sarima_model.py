"""
sarima_model.py
---------------
SARIMA model wrapper for CA_1 daily demand forecasting.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")


def train_sarima(
    train_series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7),
) -> object:
    """
    Fit a SARIMA model on the training series.

    Parameters
    ----------
    train_series    : pd.Series of daily sales (indexed by date or integer)
    order           : (p, d, q) — non-seasonal AR, differencing, MA orders
    seasonal_order  : (P, D, Q, m) — seasonal orders; m=7 for weekly seasonality

    Returns
    -------
    Fitted SARIMAXResults object
    """
    print(f"Fitting SARIMA{order}x{seasonal_order} ...")
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    print(f"AIC: {result.aic:.2f}  |  BIC: {result.bic:.2f}")
    return result


def predict_sarima(fitted_model, steps: int) -> np.ndarray:
    """
    Generate out-of-sample point forecasts.

    Parameters
    ----------
    fitted_model : SARIMAXResults from train_sarima()
    steps        : number of future periods to forecast

    Returns
    -------
    np.ndarray of predicted values
    """
    forecast = fitted_model.forecast(steps=steps)
    return np.maximum(forecast.values, 0)  # clip negatives


def sarima_summary(fitted_model) -> str:
    """Return the SARIMA model summary as a string."""
    return str(fitted_model.summary())
