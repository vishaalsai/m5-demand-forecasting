"""
run_models.py
-------------
Standalone script to train SARIMA, Prophet, and LSTM models,
generate all plots, and save metrics CSV.
Run from project root: python run_models.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── paths ────────────────────────────────────────────────────────────────────
PLOT_DIR    = os.path.join(os.path.dirname(__file__), "outputs", "plots")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "metrics")
os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
from data_loader import load_m5_data
from evaluation  import evaluate_all

df = load_m5_data()
df = df.sort_values("date").reset_index(drop=True)

TEST_DAYS  = 28
train = df.iloc[:-TEST_DAYS].copy()
test  = df.iloc[-TEST_DAYS:].copy()

y_train = train["sales"].values.astype(float)
y_test  = test["sales"].values.astype(float)
dates_test = test["date"].values

print(f"Train: {len(train)} days  |  Test: {len(test)} days")
print(f"Test window: {test['date'].iloc[0].date()} → {test['date'].iloc[-1].date()}")

results = {}   # model_name -> {eval_df, preds}

# ═══════════════════════════════════════════════════════════════════════════════
# SARIMA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── SARIMA ───")
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,7),
                 enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = sarima.fit(disp=False)

sarima_fc    = sarima_fit.get_forecast(steps=TEST_DAYS)
sarima_preds = np.maximum(sarima_fc.predicted_mean.values, 0)
sarima_ci    = sarima_fc.conf_int()

sarima_eval = evaluate_all(y_test, sarima_preds, "SARIMA")
results["SARIMA"] = {"eval": sarima_eval, "preds": sarima_preds, "ci": sarima_ci}
print(sarima_eval.to_string(index=False))

# plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train["date"].values[-90:], y_train[-90:], color="#2196F3", label="Train (last 90d)", lw=1.5)
ax.plot(dates_test, y_test,  color="#333333", label="Actual",  lw=2)
ax.plot(dates_test, sarima_preds, color="#FF5722", label="SARIMA forecast", lw=2, ls="--")
lower = sarima_ci.iloc[:, 0].values
upper = sarima_ci.iloc[:, 1].values
ax.fill_between(dates_test, lower, upper, alpha=0.2, color="#FF5722", label="95% CI")
ax.set_title("SARIMA(1,1,1)(1,1,1,7) — 28-Day Forecast", fontsize=14)
ax.set_xlabel("Date"); ax.set_ylabel("Daily Sales")
ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.tight_layout()
path = os.path.join(PLOT_DIR, "08_sarima_forecast.png")
plt.savefig(path, dpi=150); plt.close()
print(f"Saved {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Prophet
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── Prophet ───")
try:
    from prophet import Prophet

    prophet_train = train.rename(columns={"date": "ds", "sales": "y"})[["ds", "y", "snap_day"]].copy()
    prophet_train["snap_day"] = prophet_train["snap_day"].astype(int)

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                interval_width=0.95)
    m.add_regressor("snap_day")
    m.fit(prophet_train)

    future = m.make_future_dataframe(periods=TEST_DAYS)
    future["snap_day"] = 0  # conservative: no SNAP on forecast days

    forecast = m.predict(future)
    prophet_preds = np.maximum(forecast["yhat"].values[-TEST_DAYS:], 0)

    prophet_eval = evaluate_all(y_test, prophet_preds, "Prophet")
    results["Prophet"] = {"eval": prophet_eval, "preds": prophet_preds}
    print(prophet_eval.to_string(index=False))

    # forecast plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train["date"].values[-90:], y_train[-90:], color="#2196F3", label="Train (last 90d)", lw=1.5)
    ax.plot(dates_test, y_test, color="#333333", label="Actual", lw=2)
    ax.plot(dates_test, prophet_preds, color="#9C27B0", label="Prophet forecast", lw=2, ls="--")
    lower_p = forecast["yhat_lower"].values[-TEST_DAYS:]
    upper_p = forecast["yhat_upper"].values[-TEST_DAYS:]
    ax.fill_between(dates_test, lower_p, upper_p, alpha=0.2, color="#9C27B0", label="95% CI")
    ax.set_title("Prophet — 28-Day Forecast with SNAP Regressor", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Sales")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "09_prophet_forecast.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

    # components plot
    fig2 = m.plot_components(forecast)
    path2 = os.path.join(PLOT_DIR, "10_prophet_components.png")
    fig2.savefig(path2, dpi=150); plt.close(fig2)
    print(f"Saved {path2}")

except Exception as e:
    print(f"Prophet failed: {e}")
    results["Prophet"] = {"eval": pd.DataFrame([{"model":"Prophet","RMSE":None,"MAE":None,"MAPE":None,"SMAPE":None}]),
                          "preds": np.full(TEST_DAYS, np.nan)}

# ═══════════════════════════════════════════════════════════════════════════════
# LSTM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── LSTM ───")
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    LOOKBACK = 30
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(y_train.reshape(-1,1)).flatten()

    def make_sequences(series, lb):
        X, y = [], []
        for i in range(lb, len(series)):
            X.append(series[i-lb:i])
            y.append(series[i])
        return np.array(X).reshape(-1, lb, 1), np.array(y)

    X_tr, y_tr = make_sequences(train_scaled, LOOKBACK)

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    lstm_model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = lstm_model.fit(X_tr, y_tr, epochs=50, batch_size=32,
                             validation_split=0.1, callbacks=[es], verbose=0)
    print(f"Trained for {len(history.history['loss'])} epochs")

    # recursive forecast
    window = list(train_scaled[-LOOKBACK:])
    lstm_scaled_preds = []
    for _ in range(TEST_DAYS):
        x = np.array(window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        p = lstm_model.predict(x, verbose=0)[0, 0]
        lstm_scaled_preds.append(p)
        window.append(float(p))

    lstm_preds = scaler.inverse_transform(
        np.array(lstm_scaled_preds).reshape(-1,1)).flatten()
    lstm_preds = np.maximum(lstm_preds, 0)

    lstm_eval = evaluate_all(y_test, lstm_preds, "LSTM")
    results["LSTM"] = {"eval": lstm_eval, "preds": lstm_preds}
    print(lstm_eval.to_string(index=False))

    # training loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train loss", color="#2196F3")
    ax.plot(history.history["val_loss"], label="Val loss",   color="#FF5722")
    ax.set_title("LSTM Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(); plt.tight_layout()
    path = os.path.join(PLOT_DIR, "11_lstm_training_loss.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

    # forecast plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train["date"].values[-90:], y_train[-90:], color="#2196F3", label="Train (last 90d)", lw=1.5)
    ax.plot(dates_test, y_test,  color="#333333", label="Actual",  lw=2)
    ax.plot(dates_test, lstm_preds, color="#4CAF50", label="LSTM forecast", lw=2, ls="--")
    ax.set_title("LSTM — 28-Day Recursive Forecast", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Sales")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "12_lstm_forecast.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

except Exception as e:
    print(f"LSTM failed: {e}")
    results["LSTM"] = {"eval": pd.DataFrame([{"model":"LSTM","RMSE":None,"MAE":None,"MAPE":None,"SMAPE":None}]),
                       "preds": np.full(TEST_DAYS, np.nan)}

# ═══════════════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n─── Model Comparison ───")
comparison = pd.concat([r["eval"] for r in results.values()], ignore_index=True)
print(comparison.to_string(index=False))

csv_path = os.path.join(METRICS_DIR, "model_comparison.csv")
comparison.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

# MAPE bar chart
fig, ax = plt.subplots(figsize=(8, 5))
models  = comparison["model"].tolist()
mapes   = comparison["MAPE"].tolist()
colors  = ["#4CAF50" if m == comparison.loc[comparison["MAPE"].idxmin(), "model"] else "#90A4AE"
           for m in models]
bars = ax.bar(models, mapes, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, mapes):
    if val is not None and not np.isnan(float(val if val else np.nan)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{float(val):.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Model Comparison — MAPE (lower is better)", fontsize=14)
ax.set_ylabel("MAPE (%)"); ax.set_ylim(0, max([float(m) for m in mapes if m]) * 1.25)
plt.tight_layout()
path = os.path.join(PLOT_DIR, "13_mape_comparison.png")
plt.savefig(path, dpi=150); plt.close()
print(f"Saved {path}")

# Overlay forecast plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train["date"].values[-60:], y_train[-60:],
        color="#2196F3", label="Train (last 60d)", lw=1.5, alpha=0.7)
ax.plot(dates_test, y_test, color="#111111", label="Actual", lw=2.5, zorder=5)

colors_map = {"SARIMA": "#FF5722", "Prophet": "#9C27B0", "LSTM": "#4CAF50"}
styles_map  = {"SARIMA": "--",      "Prophet": "-.",       "LSTM": ":"}
for name, res in results.items():
    preds = res["preds"]
    if not np.all(np.isnan(preds)):
        ax.plot(dates_test, preds, label=name, color=colors_map.get(name, "gray"),
                lw=2, ls=styles_map.get(name, "-"))

ax.set_title("All Models vs Actual — 28-Day Forecast", fontsize=14)
ax.set_xlabel("Date"); ax.set_ylabel("Daily Sales")
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout()
path = os.path.join(PLOT_DIR, "14_model_overlay.png")
plt.savefig(path, dpi=150); plt.close()
print(f"Saved {path}")

print("\n✓ All outputs generated successfully.")
