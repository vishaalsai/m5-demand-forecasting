"""
eda.py
------
Helper functions for exploratory data analysis on the CA_1 daily sales DataFrame.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the sales column."""
    return df["sales"].describe().to_frame()


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of nulls per column."""
    total = df.isnull().sum()
    pct = (total / len(df) * 100).round(2)
    return pd.DataFrame({"missing": total, "pct": pct})[total > 0]


# ---------------------------------------------------------------------------
# Trend & decomposition
# ---------------------------------------------------------------------------

def plot_sales_over_time(df: pd.DataFrame, rolling_window: int = 28, save_path: str = None):
    """Plot raw daily sales with an optional rolling mean overlay."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["date"], df["sales"], alpha=0.4, linewidth=0.8, label="Daily Sales")
    ax.plot(
        df["date"],
        df["sales"].rolling(rolling_window, center=True).mean(),
        linewidth=2,
        color="crimson",
        label=f"{rolling_window}-day Rolling Mean",
    )
    ax.set_title("CA_1 Daily Sales Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------

def plot_weekly_seasonality(df: pd.DataFrame, save_path: str = None):
    """Box plot of sales by day of week."""
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="sales", by="day_of_week", ax=ax)
    ax.set_xticklabels(day_labels)
    ax.set_title("Sales Distribution by Day of Week — CA_1")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Units Sold")
    plt.suptitle("")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_monthly_seasonality(df: pd.DataFrame, save_path: str = None):
    """Box plot of sales by calendar month."""
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(12, 5))
    df.boxplot(column="sales", by="month", ax=ax)
    ax.set_xticklabels(month_labels)
    ax.set_title("Sales Distribution by Month — CA_1")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units Sold")
    plt.suptitle("")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_yearly_trend(df: pd.DataFrame, save_path: str = None):
    """Bar chart of total annual sales."""
    annual = df.groupby("year")["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(annual["year"], annual["sales"], color="steelblue", edgecolor="white")
    ax.set_title("Annual Total Sales — CA_1")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Units Sold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Flag days whose sales z-score exceeds the threshold."""
    mean = df["sales"].mean()
    std = df["sales"].std()
    df = df.copy()
    df["z_score"] = (df["sales"] - mean) / std
    return df[df["z_score"].abs() > threshold].sort_values("z_score", ascending=False)


def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame, save_path: str = None):
    """Overlay detected anomaly points on the sales time series."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["date"], df["sales"], alpha=0.5, linewidth=0.8, label="Daily Sales")
    ax.scatter(
        anomalies["date"],
        anomalies["sales"],
        color="red",
        zorder=5,
        s=50,
        label="Anomaly",
    )
    ax.set_title("Sales Anomalies (Z-score > 3σ) — CA_1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

def run_adf_test(series: pd.Series, verbose: bool = True) -> dict:
    """
    Run the Augmented Dickey-Fuller stationarity test.

    Returns a dict with statistic, p_value, and is_stationary (p < 0.05).
    """
    result = adfuller(series.dropna())
    output = {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }
    if verbose:
        print(f"ADF Statistic : {output['adf_statistic']:.4f}")
        print(f"p-value       : {output['p_value']:.4f}")
        print(f"Stationary    : {output['is_stationary']}")
        for k, v in output["critical_values"].items():
            print(f"  Critical ({k}): {v:.4f}")
    return output
