"""
streamlit_app.py
----------------
4-page Streamlit dashboard for M5 CA_1 demand forecasting.

Pages:
  1. Overview        – raw data, key metrics, rolling trend
  2. EDA & Patterns  – seasonality, heatmap, anomalies
  3. Model Results   – forecast accuracy, SARIMA vs LSTM vs Prophet
  4. Business Impact – cost analysis, safety stock calculator

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── resolve project root & src/ ───────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
PLOT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="M5 Demand Forecasting — CA_1",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📦 M5 Demand Forecasting")
st.sidebar.markdown("**Store:** Walmart CA_1 · California")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["1 · Overview", "2 · EDA & Patterns", "3 · Model Results", "4 · Business Impact"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** M5 Forecasting Competition  \n"
    "**Period:** Jan 2011 – May 2016  \n"
    "**Models:** SARIMA · LSTM · Naive  \n"
    "**Best MAPE:** 5.61% (SARIMA)"
)

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading M5 data …")
def load_data() -> pd.DataFrame:
    from data_loader import load_m5_data
    return load_m5_data()


def show_png(filename: str, caption: str = "") -> None:
    path = os.path.join(PLOT_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not yet generated: `{filename}`  \n"
                f"Run `python notebooks/run_phase3.py` to produce it.")


def load_csv(name: str) -> pd.DataFrame | None:
    path = os.path.join(METRICS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


# ── data load ─────────────────────────────────────────────────────────────────
# Raw M5 CSVs are not committed (too large). On Streamlit Cloud the app
# falls back to pre-generated plots and metric CSVs from outputs/.
try:
    df = load_data()
    data_ok = True
except Exception:
    data_ok = False

# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
if page == "1 · Overview":
    st.title("Sales Overview — Walmart CA_1")
    st.markdown(
        "Daily aggregated sales for Walmart store **CA_1** (California), "
        "covering **Jan 2011 – May 2016** from the M5 Forecasting Competition."
    )

    if not data_ok:
        st.info(
            "Raw M5 data files are not bundled with this deployment (files are ~1 GB). "
            "Interactive charts below are replaced by pre-generated plots. "
            "All forecast outputs and business metrics are fully available."
        )

    if data_ok:
        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Days",        f"{len(df):,}")
        c2.metric("Date Range",        f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}")
        c3.metric("Avg Daily Sales",   f"{df['sales'].mean():,.0f}")
        c4.metric("Peak Daily Sales",  f"{df['sales'].max():,.0f}")
        c5.metric("Total Units Sold",  f"{df['sales'].sum():,.0f}")

        st.markdown("---")

        # Full time-series
        st.subheader("Daily Sales — Full 5-Year History")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df["date"], df["sales"], color="#1565C0", lw=0.7, alpha=0.7, label="Daily sales")
        ax.plot(df["date"], df["sales"].rolling(30, center=True).mean(),
                color="#E53935", lw=2.2, label="30-day rolling mean")
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Year-over-year summary
        st.subheader("Year-over-Year Summary")
        yoy = df.copy()
        yoy["year"] = yoy["date"].dt.year
        yoy_agg = (yoy.groupby("year")["sales"]
                   .agg(Total="sum", Mean="mean", Max="max", Min="min")
                   .reset_index())
        yoy_agg.columns = ["Year", "Total Units", "Avg/Day", "Peak/Day", "Low/Day"]
        yoy_agg["Total Units"] = yoy_agg["Total Units"].map("{:,.0f}".format)
        yoy_agg["Avg/Day"]     = yoy_agg["Avg/Day"].map("{:,.0f}".format)
        yoy_agg["Peak/Day"]    = yoy_agg["Peak/Day"].map("{:,.0f}".format)
        yoy_agg["Low/Day"]     = yoy_agg["Low/Day"].map("{:,.0f}".format)
        st.dataframe(yoy_agg, use_container_width=True, hide_index=True)

        with st.expander("Show raw data (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)
    else:
        show_png("01_timeseries_overview.png", "CA_1 Daily Sales 2011–2016")
        # Show hardcoded KPIs from known dataset stats
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Days",       "1,913")
        c2.metric("Date Range",       "Jan 2011 – May 2016")
        c3.metric("Avg Daily Sales",  "4,894")
        c4.metric("Peak Daily Sales", "8,185")
        c5.metric("Total Units Sold", "9,364,498")

# =============================================================================
# PAGE 2 — EDA & PATTERNS
# =============================================================================
elif page == "2 · EDA & Patterns":
    st.title("Exploratory Data Analysis — Seasonality & Patterns")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Decomposition", "Weekly", "Monthly", "SNAP Impact", "Anomalies"]
    )

    with tab1:
        st.subheader("Seasonal-Trend Decomposition (period=365)")
        show_png("02_decomposition.png",
                 "Additive decomposition: Trend + Seasonal + Residual")
        st.markdown(
            "The decomposition reveals a **slight upward trend** through 2015, "
            "strong weekly seasonality, and mild annual seasonality. "
            "The residuals show occasional spikes corresponding to holidays and SNAP days."
        )

    with tab2:
        st.subheader("Weekly Seasonality")
        show_png("03_weekly_seasonality.png", "Average sales by day of week")
        if data_ok:
            st.subheader("Interactive Year × Weekday Heatmap")
            df2 = df.copy()
            df2["year"]     = df2["date"].dt.year
            df2["day_name"] = df2["date"].dt.day_name()
            pivot = (df2.groupby(["year", "day_name"])["sales"]
                     .mean().unstack())
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            pivot = pivot[[d for d in day_order if d in pivot.columns]]
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, label="Avg Daily Sales")
            ax.set_title("Average Daily Sales by Year × Weekday")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.subheader("Monthly Seasonality")
        show_png("04_monthly_seasonality.png", "Average sales by month of year")
        if data_ok:
            df3 = df.copy()
            df3["month_name"] = df3["date"].dt.strftime("%b")
            df3["month_num"]  = df3["date"].dt.month
            monthly = (df3.groupby(["month_num","month_name"])["sales"]
                       .mean().reset_index().sort_values("month_num"))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(monthly["month_name"], monthly["sales"],
                   color=plt.cm.tab20.colors[:12], edgecolor="white")
            ax.set_title("Average Daily Sales by Month")
            ax.set_ylabel("Average Units Sold")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab4:
        st.subheader("SNAP Day Sales Impact")
        show_png("05_snap_impact.png", "Sales on SNAP vs non-SNAP days")
        if data_ok:
            snap_stats = df.groupby("snap_day")["sales"].agg(["mean","median","std"]).round(1)
            snap_stats.index = ["Non-SNAP", "SNAP"]
            snap_stats.columns = ["Mean", "Median", "Std Dev"]
            st.dataframe(snap_stats, use_container_width=True)
            snap_lift = (snap_stats.loc["SNAP","Mean"] / snap_stats.loc["Non-SNAP","Mean"] - 1) * 100
            st.info(f"SNAP days show a **{snap_lift:.1f}% average sales lift** vs non-SNAP days.")

    with tab5:
        st.subheader("Anomaly Detection")
        show_png("06_anomalies.png", "Detected outlier days")
        if data_ok:
            mu, sigma = df["sales"].mean(), df["sales"].std()
            outliers = df[np.abs(df["sales"] - mu) > 2.5 * sigma].copy()
            outliers["z_score"] = ((outliers["sales"] - mu) / sigma).round(2)
            st.markdown(f"**{len(outliers)} anomalous days** detected (|z| > 2.5σ):")
            st.dataframe(outliers[["date","sales","z_score","is_holiday","snap_day"]],
                         use_container_width=True, hide_index=True)

# =============================================================================
# PAGE 3 — MODEL RESULTS
# =============================================================================
elif page == "3 · Model Results":
    st.title("Forecast Model Results — 28-Day Test Window")
    st.markdown(
        "Three models were trained on the first **1,885 days** (Jan 2011 – Mar 2016) "
        "and evaluated on the final **28 days** (Mar 28 – Apr 24, 2016)."
    )

    # Metrics table
    comp = load_csv("model_comparison.csv")
    if comp is not None:
        st.subheader("Model Accuracy Metrics")
        display_comp = comp.copy()
        display_comp["MAPE"] = display_comp["MAPE"].apply(
            lambda x: f"{float(x):.2f}%" if pd.notna(x) else "—"
        )
        display_comp["SMAPE"] = display_comp["SMAPE"].apply(
            lambda x: f"{float(x):.2f}%" if pd.notna(x) else "—"
        )
        display_comp["RMSE"] = display_comp["RMSE"].apply(
            lambda x: f"{float(x):,.1f}" if pd.notna(x) else "—"
        )
        display_comp["MAE"] = display_comp["MAE"].apply(
            lambda x: f"{float(x):,.1f}" if pd.notna(x) else "—"
        )
        display_comp.columns = ["Model", "RMSE", "MAE", "MAPE", "SMAPE"]
        st.dataframe(display_comp, use_container_width=True, hide_index=True)

        valid = comp.dropna(subset=["MAPE"])
        if not valid.empty:
            best = valid.loc[valid["MAPE"].idxmin(), "model"]
            best_mape = valid["MAPE"].min()
            st.success(f"**Best model: {best}** with MAPE = {best_mape:.2f}%")
    else:
        st.info("Run `python run_models.py` to generate model metrics.")
        st.dataframe(pd.DataFrame({
            "Model": ["SARIMA","LSTM","Naive"],
            "RMSE":  ["359.0","959.4","437.7"],
            "MAPE":  ["5.61%","17.76%","6.96%"],
        }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Forecast plots in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SARIMA(1,1,1)(1,1,1,7)")
        show_png("08_sarima_forecast.png", "SARIMA — 28-Day Forecast with 95% CI")
        st.caption("MAPE: 5.61% | RMSE: 359  \nCaptures weekly seasonality via seasonal differencing.")
    with col2:
        st.subheader("LSTM (Bi-LSTM, 30-day window)")
        show_png("12_lstm_forecast.png", "LSTM — 28-Day Recursive Forecast")
        st.caption("MAPE: 17.76% | RMSE: 959  \nRecursive forecast prone to error accumulation.")

    st.subheader("All Models Overlay")
    show_png("14_model_overlay.png", "Actual vs All Model Forecasts")

    # MAPE bar chart inline
    if comp is not None:
        valid2 = comp.dropna(subset=["MAPE"])
        if len(valid2) > 0:
            st.subheader("MAPE Comparison")
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#1E88E5" if m == "SARIMA" else
                      "#E53935" if m == "LSTM" else "#FB8C00"
                      for m in valid2["model"]]
            bars = ax.bar(valid2["model"], valid2["MAPE"], color=colors,
                          edgecolor="white", width=0.5)
            for bar, val in zip(bars, valid2["MAPE"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{val:.2f}%", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
            ax.set_ylabel("MAPE (%)")
            ax.set_title("MAPE by Model (lower is better)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(0, valid2["MAPE"].max() * 1.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # LSTM training loss
    with st.expander("LSTM Training Curve"):
        show_png("11_lstm_training_loss.png", "LSTM Training vs Validation Loss")

# =============================================================================
# PAGE 4 — BUSINESS IMPACT
# =============================================================================
elif page == "4 · Business Impact":
    st.title("Business Impact Analysis")
    st.markdown(
        "Translating forecast accuracy into **inventory cost terms** for CA_1 "
        "using the following parameters:"
    )

    # Cost parameters display
    UNIT_COST         = 8.50
    HOLDING_COST_RATE = 0.25
    STOCKOUT_PENALTY  = 15.00
    TEST_DAYS         = 28

    p1, p2, p3 = st.columns(3)
    p1.metric("Unit Cost",          f"${UNIT_COST:.2f}")
    p2.metric("Holding Cost Rate",  f"{HOLDING_COST_RATE*100:.0f}% / year")
    p3.metric("Stockout Penalty",   f"${STOCKOUT_PENALTY:.2f} / unit")

    st.markdown("---")

    # --- Section 4A: Annual cost comparison ---
    st.subheader("Annual Inventory Cost by Model")
    biz = load_csv("business_impact.csv")
    if biz is not None:
        # KPI tiles — highlight SARIMA
        sarima_row = biz[biz["Model"] == "SARIMA"].iloc[0]
        lstm_row   = biz[biz["Model"] == "LSTM"].iloc[0]
        naive_row  = biz[biz["Model"] == "Naive"].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("SARIMA Annual Cost",
                  f"${sarima_row['Total_Cost_Year']:,.0f}",
                  delta=f"${sarima_row['Savings_vs_Naive']:,.0f} vs Naive",
                  delta_color="inverse")
        c2.metric("LSTM Annual Cost",
                  f"${lstm_row['Total_Cost_Year']:,.0f}",
                  delta=f"${lstm_row['Savings_vs_Naive']:,.0f} vs Naive",
                  delta_color="inverse")
        c3.metric("Naive Baseline",
                  f"${naive_row['Total_Cost_Year']:,.0f}",
                  delta="baseline")

        show_png("15_business_impact.png",
                 "Annual inventory costs: Overstock + Stockout per model")

        with st.expander("Cost breakdown table"):
            biz_disp = biz.copy()
            for col in ["Overstock_Cost_Year","Stockout_Cost_Year","Total_Cost_Year","Savings_vs_Naive"]:
                biz_disp[col] = biz_disp[col].apply(lambda x: f"${float(x):,.0f}")
            biz_disp.columns = ["Model","Overstock/Year","Stockout/Year","Total/Year","Savings vs Naive"]
            st.dataframe(biz_disp, use_container_width=True, hide_index=True)

        # Insight callout
        sarima_save = sarima_row["Savings_vs_Naive"]
        st.info(
            f"**Key insight:** With STOCKOUT_PENALTY=${STOCKOUT_PENALTY:.0f}/unit, "
            f"stockout costs dominate (>99% of total). "
            f"SARIMA's lower MAPE ({5.61:.2f}%) translates to "
            f"{'lower' if sarima_save > 0 else 'higher'} total costs than Naive "
            f"by ${abs(sarima_save):,.0f}/year — reflecting that systematic "
            f"under-prediction drives stockout penalties more than MAPE alone suggests."
        )
    else:
        st.info("Run `python -X utf8 notebooks/run_phase3.py` to generate business impact data.")
        show_png("15_business_impact.png")

    st.markdown("---")

    # --- Section 4B: Executive dashboard ---
    st.subheader("Executive Dashboard")
    show_png("16_executive_dashboard.png",
             "2×2 dashboard: MAPE, Annual Costs, SARIMA Forecast, Safety Stock")

    st.markdown("---")

    # --- Section 4C: Safety Stock ---
    st.subheader("Safety Stock Analysis")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        ss_df = load_csv("safety_stock_analysis.csv")
        if ss_df is not None:
            st.markdown("**Pre-computed table** (based on SARIMA forecast error σ = 311 units/day)")
            for col in ["annual_holding_cost", "inventory_value_usd"]:
                ss_df[col] = ss_df[col].apply(lambda x: f"${float(x):,.0f}")
            ss_df.columns = [
                "SL %", "Z", "Lead (days)",
                "Safety Stock (units)", "Ann. Holding Cost", "Inventory Value"
            ]
            st.dataframe(ss_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run Phase 3 script to generate safety_stock_analysis.csv.")

    with col_right:
        st.markdown("**Interactive Calculator**")
        from scipy import stats as sp_stats

        sigma_input = st.number_input(
            "Forecast error σ (units/day)", 1.0, 3000.0, 311.0, 10.0,
            help="Use SARIMA σ=311 or enter a custom value"
        )
        svc_level = st.slider("Service Level (%)", 80, 99, 95)
        lead_time = st.number_input("Lead Time (days)", 1, 30, 3)
        unit_cost = st.number_input("Unit Cost ($)", 0.5, 200.0, UNIT_COST, 0.50)
        hold_rate = st.number_input("Holding Cost Rate", 0.05, 0.50, HOLDING_COST_RATE, 0.01,
                                    format="%.2f")

        z         = sp_stats.norm.ppf(svc_level / 100)
        ss_units  = z * sigma_input * np.sqrt(lead_time)
        ann_hold  = ss_units * unit_cost * hold_rate
        inv_val   = ss_units * unit_cost

        r1, r2, r3 = st.columns(3)
        r1.metric("Z-score",             f"{z:.3f}")
        r2.metric("Safety Stock",        f"{ss_units:,.0f} units")
        r3.metric("Inventory Value",     f"${inv_val:,.0f}")
        st.metric("Annual Holding Cost", f"${ann_hold:,.0f}")
        st.caption("Formula: SS = z × σ × √(lead_time)")
