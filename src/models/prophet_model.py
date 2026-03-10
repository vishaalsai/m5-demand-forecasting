"""
prophet_model.py
----------------
Meta Prophet model wrapper for CA_1 daily demand forecasting.
"""

import pandas as pd
import numpy as np

try:
    from prophet import Prophet
except ImportError:
    raise ImportError("Install prophet: pip install prophet")


def build_prophet_df(df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales") -> pd.DataFrame:
    """
    Convert the project DataFrame to Prophet's required format (ds, y).

    Also adds SNAP days and holidays as regressors if present.
    """
    prophet_df = df[[date_col, sales_col]].rename(columns={date_col: "ds", sales_col: "y"}).copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    if "snap_day" in df.columns:
        prophet_df["snap_day"] = df["snap_day"].astype(int).values

    return prophet_df


def train_prophet(
    train_df: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    add_snap: bool = True,
) -> "Prophet":
    """
    Fit a Prophet model on the training DataFrame (must have ds, y columns).

    Parameters
    ----------
    train_df             : DataFrame with ds, y — output of build_prophet_df()
    yearly_seasonality   : include yearly Fourier seasonality
    weekly_seasonality   : include weekly Fourier seasonality
    add_snap             : add snap_day as an additional regressor if present

    Returns
    -------
    Fitted Prophet model
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=0.05,
    )

    if add_snap and "snap_day" in train_df.columns:
        model.add_regressor("snap_day")

    model.fit(train_df)
    print("Prophet model fitted.")
    return model


def predict_prophet(model: "Prophet", periods: int, freq: str = "D") -> pd.DataFrame:
    """
    Generate a future forecast DataFrame.

    Parameters
    ----------
    model   : fitted Prophet model
    periods : number of future days to forecast
    freq    : pandas frequency string

    Returns
    -------
    Prophet forecast DataFrame (includes yhat, yhat_lower, yhat_upper)
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)

    # Fill regressor for future dates (assume no SNAP for simplicity)
    if "snap_day" in model.extra_regressors:
        future["snap_day"] = 0

    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    return forecast


def extract_forecast_values(forecast: pd.DataFrame, n_test: int) -> np.ndarray:
    """Return the last n_test predicted values from a Prophet forecast DataFrame."""
    return forecast["yhat"].values[-n_test:]
