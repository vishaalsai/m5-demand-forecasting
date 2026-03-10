"""
streamlit_app.py
----------------
Interactive demand forecasting dashboard for CA_1 (M5 Walmart dataset).

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="M5 Demand Forecast — CA_1",
    page_icon="📦",
    layout="wide",
)

st.title("📦 M5 Walmart Demand Forecasting Dashboard")
st.caption("Store: CA_1  |  Scope: Daily aggregated sales  |  Models: SARIMA · Prophet · LSTM")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["SARIMA", "Prophet", "LSTM"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=28)

st.sidebar.markdown("---")
st.sidebar.info(
    "Place M5 CSV files in `data/raw/` and run `python src/data_loader.py` to verify data loading."
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading CA_1 sales data ...")
def get_data():
    try:
        from src.data_loader import load_ca1_daily
        return load_ca1_daily()
    except FileNotFoundError:
        return None


df = get_data()

if df is None:
    st.warning(
        "Data files not found in `data/raw/`. "
        "Please download the M5 dataset from Kaggle and place the CSV files there."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Days", f"{len(df):,}")
col2.metric("Avg Daily Sales", f"{df['sales'].mean():,.0f}")
col3.metric("Peak Day Sales", f"{df['sales'].max():,}")
col4.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")

# ---------------------------------------------------------------------------
# Sales overview chart
# ---------------------------------------------------------------------------
st.subheader("Historical Sales — CA_1")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["date"], df["sales"], linewidth=0.7, alpha=0.6, label="Daily Sales")
ax.plot(
    df["date"],
    df["sales"].rolling(28, center=True).mean(),
    color="crimson",
    linewidth=2,
    label="28-day Rolling Mean",
)
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------------------------
# Seasonality heatmap
# ---------------------------------------------------------------------------
st.subheader("Weekly × Monthly Seasonality Heatmap")
pivot = df.pivot_table(index="month", columns="day_of_week", values="sales", aggfunc="mean")
pivot.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot)]
pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot.columns)]

fig2, ax2 = plt.subplots(figsize=(10, 5))
import seaborn as sns
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", ax=ax2)
ax2.set_title("Average Daily Sales by Month × Day of Week")
st.pyplot(fig2)

# ---------------------------------------------------------------------------
# Forecast placeholder
# ---------------------------------------------------------------------------
st.subheader(f"{model_choice} Forecast — Next {forecast_horizon} Days")
st.info(
    f"Train your {model_choice} model in `notebooks/02_models.ipynb` and save predictions to "
    f"`outputs/metrics/` to visualise them here."
)

# Placeholder random forecast for UI demonstration
last_date = df["date"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
baseline = df["sales"].tail(28).mean()
noise = np.random.normal(0, baseline * 0.05, forecast_horizon)
demo_forecast = np.maximum(baseline + noise, 0)

fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(df["date"].tail(90), df["sales"].tail(90), label="Historical (last 90 days)", color="steelblue")
ax3.plot(future_dates, demo_forecast, linestyle="--", color="crimson", label=f"{model_choice} Forecast (demo)")
ax3.fill_between(future_dates, demo_forecast * 0.9, demo_forecast * 1.1, alpha=0.2, color="crimson", label="±10% Band")
ax3.axvline(last_date, color="gray", linestyle=":")
ax3.set_xlabel("Date")
ax3.set_ylabel("Units Sold")
ax3.legend()
ax3.set_title(f"{model_choice} — {forecast_horizon}-Day Forecast (placeholder)")
st.pyplot(fig3)

st.caption("Replace the demo forecast with actual model predictions by loading saved outputs.")
