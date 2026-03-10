"""
visualizations.py
-----------------
Reusable plotting utilities for model forecasts and business impact analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    forecast: pd.Series,
    model_name: str = "Model",
    date_col: str = "date",
    sales_col: str = "sales",
    save_path: str = None,
):
    """
    Plot train actuals, test actuals, and model forecast on one axis.

    Parameters
    ----------
    train, test : DataFrames with date_col and sales_col
    forecast    : Series of predicted values aligned to test index
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(train[date_col], train[sales_col], label="Train Actuals", color="steelblue", linewidth=1)
    ax.plot(test[date_col], test[sales_col], label="Test Actuals", color="black", linewidth=1.5)
    ax.plot(test[date_col], forecast.values, label=f"{model_name} Forecast", color="crimson",
            linestyle="--", linewidth=1.5)
    ax.set_title(f"{model_name} — Forecast vs Actuals (CA_1)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_residuals(y_true: pd.Series, y_pred: pd.Series, model_name: str = "Model", save_path: str = None):
    """Plot residuals over time and a residual distribution histogram."""
    residuals = pd.Series(y_true.values) - pd.Series(y_pred.values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(residuals.values, linewidth=0.8, color="darkorange")
    axes[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_title(f"{model_name} — Residuals Over Time")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].set_title(f"{model_name} — Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame, save_path: str = None):
    """
    Bar chart comparing RMSE, MAE, MAPE across models.

    Parameters
    ----------
    metrics_df : DataFrame with columns [model, RMSE, MAE, MAPE, SMAPE]
    """
    metric_cols = ["RMSE", "MAE", "MAPE"]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(14, 5))
    for ax, metric in zip(axes, metric_cols):
        ax.bar(metrics_df["model"], metrics_df[metric], color=["steelblue", "seagreen", "crimson"])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
    fig.suptitle("Model Comparison — CA_1 Forecast", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
