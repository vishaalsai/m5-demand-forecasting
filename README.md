# M5 Walmart Demand Forecasting — End-to-End Time Series System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Business Problem

Retail giants like Walmart manage hundreds of thousands of SKUs across thousands of stores. Poor demand forecasting leads to two costly failure modes:

- **Overstock**: Excess inventory ties up capital, increases storage costs, and results in markdowns that erode margins.
- **Stockouts**: Lost sales, customer dissatisfaction, and long-term brand damage.

This project builds an **end-to-end demand forecasting system** on Walmart's real M5 competition dataset — scoped to the **CA_1 store** — to demonstrate how time series modeling translates directly into supply chain and inventory ROI.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | `pandas`, `numpy`, `scipy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Classical Forecasting | `statsmodels` (SARIMA) |
| ML Forecasting | `prophet` (Meta Prophet) |
| Deep Learning | `tensorflow` / Keras (LSTM) |
| Model Tracking | `mlflow` |
| Dashboard | `streamlit` |
| Environment | `jupyter`, `notebook`, `ipykernel` |

---

## Project Structure

```
m5-demand-forecasting/
├── data/
│   └── raw/               # Place M5 CSV files here (not tracked by git)
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   ├── 02_models.ipynb    # Model training & comparison
│   └── 03_business_impact.ipynb  # ROI & inventory simulation
├── src/
│   ├── data_loader.py     # Data ingestion & preprocessing
│   ├── eda.py             # EDA helper functions
│   ├── evaluation.py      # Metrics (WRMSSE, RMSE, MAPE)
│   ├── visualizations.py  # Reusable plotting utilities
│   └── models/
│       ├── sarima_model.py
│       ├── prophet_model.py
│       └── lstm_model.py
├── outputs/
│   ├── plots/             # Saved figures
│   └── metrics/           # Saved metric CSVs
├── app/
│   └── streamlit_app.py   # Interactive forecast dashboard
├── requirements.txt
└── README.md
```

---

## Phase 1: EDA

> Notebook: `notebooks/01_eda.ipynb`

Scope: CA_1 store — daily aggregated sales from 2011 to 2016.

Key analyses:
- Sales volume trends over 5 years
- Weekly and monthly seasonality decomposition
- Holiday/event impact quantification (Super Bowl, Christmas, SNAP days)
- Anomaly detection and business explanation
- Stationarity testing (ADF test)

**Results:** *(to be filled after EDA is complete)*

---

## Phase 2: Models

> Notebook: `notebooks/02_models.ipynb`

### SARIMA
Classical statistical model. Captures trend + seasonal structure with explicit AR/MA terms.

### Prophet
Meta's additive regression model. Handles multiple seasonalities, holidays, and trend changepoints out-of-the-box.

### LSTM
Sequence-to-sequence neural network. Learns non-linear temporal dependencies across long look-back windows.

**Model Comparison Results:**

| Model | RMSE | MAPE | WRMSSE |
|---|---|---|---|
| SARIMA | — | — | — |
| Prophet | — | — | — |
| LSTM | — | — | — |

*(Results to be filled after training)*

---

## Phase 3: Business Impact

> Notebook: `notebooks/03_business_impact.ipynb`

Translating forecast accuracy into dollars:
- Inventory holding cost simulation
- Stockout cost estimation
- Safety stock optimization using forecast uncertainty bands
- Break-even analysis comparing model vs. naive baseline

**Estimated Annual Savings (CA_1):** *(to be calculated)*

---

## Production Thinking

This project is designed with production readiness in mind:

- **MLflow** tracks all experiments, parameters, and metrics for reproducibility.
- **Modular `src/` package** separates data, modeling, and evaluation concerns — easy to plug into an Airflow DAG or Lambda function.
- **Streamlit dashboard** (`app/streamlit_app.py`) provides a business-facing interface for scenario planning.
- Models are serialized to disk and can be swapped without changing the serving layer.

---

## How to Run

### 1. Clone & install dependencies
```bash
git clone <repo-url>
cd m5-demand-forecasting
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download M5 data
Download the M5 Forecasting dataset from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place the following files in `data/raw/`:
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

### 3. Run EDA notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Test the data loader
```bash
python src/data_loader.py
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Dataset

**M5 Forecasting — Accuracy** (Walmart, via Kaggle)
- ~30,000 item-store combinations
- 1,941 days of daily sales (Jan 2011 – May 2016)
- 3 US states: CA, TX, WI
- This project scopes to **CA_1** for depth over breadth

---

## License

MIT License — see `LICENSE` for details.
