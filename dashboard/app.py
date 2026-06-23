"""
MLOps Performance Dashboard — Streamlit app for monitoring model health.

Run with: uv run streamlit run dashboard/app.py
"""

import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLOps Forecasting Dashboard", layout="wide")
st.title("⚡ aFRR Price Forecasting — Model Health Dashboard")


# --- Simulate monitoring data (replace with real COS/S3 reads in production) ---
@st.cache_data(ttl=300)
def load_monitoring_data() -> dict:
    """Generate realistic monitoring data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(
        end=datetime.date.today(), periods=90, freq="D"
    )

    horizons = ["D+1", "D+2", "D+3", "D+4"]
    data = {}

    for i, h in enumerate(horizons):
        actual = 50 + 20 * np.sin(np.arange(90) * 0.1) + np.random.normal(0, 5, 90)
        # Simulate slight degradation over time
        noise_growth = np.linspace(1, 1.5 + i * 0.2, 90)
        predicted = actual + np.random.normal(0, 3, 90) * noise_growth

        mae = np.abs(actual - predicted)
        bias = predicted - actual

        data[h] = pd.DataFrame({
            "date": dates,
            "actual": actual,
            "predicted": predicted,
            "mae": mae,
            "bias": bias,
        })

    return data


def load_drift_scores() -> pd.DataFrame:
    """Generate realistic drift detection results."""
    np.random.seed(123)
    dates = pd.date_range(end=datetime.date.today(), periods=30, freq="D")
    features = [
        "lag_1", "lag_7", "rolling_mean_7d", "fcr_price",
        "consumption_forecast", "gas_price", "spot_price",
        "wind_forecast", "solar_forecast",
    ]

    records = []
    for d in dates:
        for f in features:
            p_value = np.random.beta(5, 2) if "forecast" not in f else np.random.beta(2, 5)
            records.append({"date": d, "feature": f, "p_value": p_value, "drifted": p_value < 0.05})

    return pd.DataFrame(records)


# --- Load data ---
monitoring_data = load_monitoring_data()
drift_data = load_drift_scores()

# --- Sidebar ---
st.sidebar.header("Controls")
horizon = st.sidebar.selectbox("Forecast Horizon", ["D+1", "D+2", "D+3", "D+4"])
lookback = st.sidebar.slider("Lookback (days)", 7, 90, 30)

df = monitoring_data[horizon].tail(lookback)

# --- KPIs ---
st.subheader(f"📊 {horizon} Performance — Last {lookback} Days")
col1, col2, col3, col4 = st.columns(4)

current_mae = df["mae"].tail(7).mean()
previous_mae = df["mae"].head(7).mean()
col1.metric("MAE (7d avg)", f"{current_mae:.2f} €/MW", f"{current_mae - previous_mae:+.2f}")

current_bias = df["bias"].tail(7).mean()
col2.metric("Bias (7d avg)", f"{current_bias:.2f} €/MW")

drift_today = drift_data[drift_data["date"] == drift_data["date"].max()]
n_drifted = drift_today["drifted"].sum()
col3.metric("Features Drifted", f"{n_drifted}/{len(drift_today)}")

coverage = ((df["actual"] >= df["predicted"] - 10) & (df["actual"] <= df["predicted"] + 10)).mean()
col4.metric("±10€ Coverage", f"{coverage:.0%}")

# --- Charts ---
st.subheader("Predictions vs Actuals")
chart_df = df[["date", "actual", "predicted"]].set_index("date")
st.line_chart(chart_df)

st.subheader("Daily MAE Trend")
mae_df = df[["date", "mae"]].set_index("date")
st.area_chart(mae_df)

# --- Drift heatmap ---
st.subheader("🔍 Feature Drift (p-values, last 30 days)")
pivot = drift_data.pivot_table(index="feature", columns="date", values="p_value")
# Show as a simple table with conditional formatting
st.dataframe(
    pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
    use_container_width=True,
    height=350,
)

# --- Alerts ---
st.subheader("🚨 Alerts")
alerts = []
if current_mae > 8:
    alerts.append(f"⚠️ **{horizon}** MAE ({current_mae:.1f}) exceeds threshold (8.0 €/MW)")
if abs(current_bias) > 3:
    alerts.append(f"⚠️ **{horizon}** systematic bias detected ({current_bias:+.1f} €/MW)")
if n_drifted >= 3:
    alerts.append(f"🔴 {n_drifted} features showing statistical drift — consider retraining")

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success("✅ All metrics within normal range")

# --- Retraining recommendation ---
st.subheader("🔄 Retraining Decision")
should_retrain = current_mae > 8 or n_drifted >= 3 or abs(current_bias) > 5
if should_retrain:
    st.error("**Recommendation: Trigger retraining pipeline**")
    st.json({
        "reason": "performance_degradation" if current_mae > 8 else "data_drift",
        "mae_current": round(current_mae, 2),
        "features_drifted": int(n_drifted),
        "bias": round(current_bias, 2),
        "suggested_action": "retrain_all_horizons" if n_drifted >= 5 else f"retrain_{horizon}",
    })
else:
    st.info("Model performance is acceptable. No retraining needed.")

