# M5 Walmart Demand Forecasting — End-to-End Time Series System

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://appapppy-uzsmfcqvv5qdjsnewofujf.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-vishaalsai%2Fm5--demand--forecasting-181717?logo=github)](https://github.com/vishaalsai/m5-demand-forecasting)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **[Launch Live Dashboard →](https://appapppy-uzsmfcqvv5qdjsnewofujf.streamlit.app/)**

---

## Business Problem

Retail giants like Walmart manage hundreds of thousands of SKUs across thousands of stores. Poor demand forecasting leads to two costly failure modes:

- **Overstock:** Excess inventory ties up capital, raises storage costs, and forces margin-eroding markdowns.
- **Stockouts:** Lost sales, customer dissatisfaction, and long-term brand damage — each costing ~$15/unit in lost margin and goodwill.

This project builds an **end-to-end demand forecasting system** on Walmart's real M5 competition dataset — scoped to **store CA_1 (California)** — demonstrating how time-series modelling translates directly into supply-chain ROI. The full pipeline runs from raw data through statistical and deep-learning models to an interactive business dashboard.

---

## Key Results

### Forecast Accuracy — 28-Day Test Window (Mar 28 – Apr 24 2016)

| Model | MAPE | RMSE | MAE | Annual Inventory Cost |
|---|---|---|---|---|
| **SARIMA(1,1,1)(1,1,1,7)** | **5.61%** | **359** | **272** | $1,260,614 |
| Naive (Lag-28 baseline) | 6.96% | 438 | — | $1,170,725 |
| LSTM (BiLSTM, 30-day window) | 17.76% | 959 | 822 | $2,153,418 |

> **SARIMA achieves the lowest MAPE (5.61%)** — 1.35 pp better than Naive and 12.15 pp better than LSTM.
> Annual cost assumptions: unit cost $8.50 · holding 25%/yr · stockout penalty $15/unit.

### Business Impact

| Metric | Value |
|---|---|
| SARIMA forecast error σ | 311 units/day |
| Recommended safety stock | **886 units** (95% SL, 3-day lead time) |
| Safety stock annual holding cost | $1,883 |
| SARIMA stockout cost vs LSTM | **$892,057/yr cheaper** |
| SARIMA overstock cost/year | $88 |
| SARIMA stockout cost/year | $1,260,526 |

> **Key insight:** Under a high stockout penalty ($15/unit), cost is dominated by under-prediction rather than over-prediction. SARIMA's systematic bias toward underprediction makes Naive cheaper in pure cost terms — but SARIMA's 5.61% MAPE still makes it the model of choice for operational planning and safety-stock sizing.

---

## EDA Findings

Five key patterns discovered in Phase 1 exploratory analysis:

1. **Strong weekly seasonality** — Saturday and Sunday average ~18% more sales than Monday–Friday; weekday sales are relatively flat.
2. **SNAP day lift** — California SNAP benefit days produce a measurable sales spike (~8% above non-SNAP average), concentrated in food and household categories.
3. **Upward trend 2011–2014, plateau thereafter** — Rolling-mean analysis shows steady growth through 2014 followed by stabilisation, consistent with store maturation.
4. **Holiday anomalies are predictable** — Christmas Eve, Thanksgiving, and Super Bowl Sunday generate the top outlier days (z-score > 2.5σ); all are calendar-identifiable.
5. **Non-stationary in levels, stationary after first differencing** — ADF test on raw sales: p = 0.21 (fail to reject unit root). After d=1 differencing: p < 0.001 — confirming the need for integrated models (ARIMA, SARIMA).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion & preprocessing | `pandas`, `numpy` |
| Statistical forecasting | `statsmodels` — SARIMA |
| Deep learning forecasting | `tensorflow` / Keras — LSTM |
| Additive forecasting | `prophet` (Meta) |
| Experiment tracking | `mlflow` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Interactive dashboard | `streamlit` |
| Scientific computing | `scipy` |
| Environment | `jupyter`, `anaconda` |

---

## Project Structure

```
m5-demand-forecasting/
├── data/
│   └── raw/                        # M5 CSVs — place here (not tracked, ~1 GB)
│       ├── sales_train_validation.csv
│       ├── calendar.csv
│       └── sell_prices.csv
├── src/
│   ├── data_loader.py              # Full ingestion → feature engineering pipeline
│   ├── eda.py                      # EDA helper functions
│   ├── evaluation.py               # RMSE, MAE, MAPE, SMAPE metrics
│   ├── visualizations.py           # Reusable plotting utilities
│   └── models/
│       ├── sarima_model.py
│       ├── prophet_model.py
│       └── lstm_model.py
├── notebooks/
│   ├── 01_eda.ipynb                # Phase 1 — Exploratory Data Analysis
│   ├── 02_models.ipynb             # Phase 2 — SARIMA + LSTM training
│   └── 03_business_impact.ipynb   # Phase 3 — ROI & inventory simulation
├── notebooks/run_phase3.py         # Standalone Phase 3 script (no nbconvert)
├── run_models.py                   # Standalone Phase 2 script
├── outputs/
│   ├── plots/                      # 16 saved figures (01–16)
│   └── metrics/
│       ├── model_comparison.csv
│       ├── business_impact.csv
│       └── safety_stock_analysis.csv
├── app/
│   └── streamlit_app.py            # 4-page interactive dashboard
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone and install

```bash
git clone https://github.com/vishaalsai/m5-demand-forecasting.git
cd m5-demand-forecasting
pip install -r requirements.txt
```

### 2. Download M5 data

Download from [Kaggle M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place in `data/raw/`:

```
data/raw/sales_train_validation.csv
data/raw/calendar.csv
data/raw/sell_prices.csv
```

### 3. Verify the data pipeline

```bash
python src/data_loader.py
```

### 4. Run Phase 2 — train models and generate plots 08–14

```bash
python run_models.py
```

### 5. Run Phase 3 — business impact analysis (plots 15–16, metric CSVs)

```bash
# Windows (Anaconda)
C:\ProgramData\Anaconda3\python.exe -X utf8 notebooks/run_phase3.py

# Linux / macOS
python notebooks/run_phase3.py
```

### 6. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Live Dashboard

**[https://appapppy-uzsmfcqvv5qdjsnewofujf.streamlit.app/](https://appapppy-uzsmfcqvv5qdjsnewofujf.streamlit.app/)**

Four pages:

| Page | Contents |
|---|---|
| 1 · Overview | 5-year sales history, KPI tiles, year-over-year table |
| 2 · EDA & Patterns | Decomposition, weekly / monthly heatmaps, SNAP analysis, anomaly table |
| 3 · Model Results | Accuracy metrics, forecast plots, MAPE comparison chart |
| 4 · Business Impact | Annual cost comparison, executive dashboard, interactive safety-stock calculator |

> Raw M5 data is not bundled (too large). The Cloud deployment renders all 16 pre-generated plots and reads metric CSVs committed to the repo.

---

## Production Thinking

- **Modular `src/` package** — data loading, evaluation, and models are fully decoupled. Any component can be wrapped in an Airflow task, AWS Lambda, or scheduled Cloud Run job without touching the others.
- **MLflow experiment tracking** — every model run logs parameters, metrics, and artefacts. Reproducibility is guaranteed; model registry enables staged promotion (Staging → Production).
- **Cost-aware evaluation** — accuracy metrics (MAPE, RMSE) are augmented with a dollar-denominated cost model. This surfaces directional bias (under- vs over-prediction) that MAPE alone hides, enabling business-aligned model selection.
- **Configurable safety-stock calculator** — the interactive dashboard lets planners adjust service level, lead time, and unit cost in real time, turning a static model output into a live decision-support tool.

---

## Dataset

**M5 Forecasting — Accuracy** (Walmart, via Kaggle)

| Attribute | Value |
|---|---|
| Scope (this project) | Store CA_1, California |
| SKUs | 3,049 item-store combinations |
| Observation period | Jan 11 2011 – May 22 2016 |
| Daily observations | 1,913 |
| Full dataset size | ~30,000 item-store combinations, 1,941 days |

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
