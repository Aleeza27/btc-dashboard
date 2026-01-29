import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from collections import deque

# =========================
# Title
# =========================
st.title("Real-Time Bitcoin Price Forecasting Dashboard")

# =========================
# Sidebar Settingss
# =========================
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Crypto Ticker", value="BTC-USD")
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

# =========================
# Data Fetching
# =========================
@st.cache_data(ttl=60)
def get_current_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d")
    current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    return data, current_price

df, current_price = get_current_data(ticker)

st.write(
    f"**Current BTC Price (as of {datetime.now().strftime('%Y-%m-%d %H:%M')}):** "
    f"${current_price:.2f}"
)

# =========================
# DSA: Fixed-size buffer
# =========================
price_history = deque(maxlen=100)
price_history.append(current_price)

# =========================
# Forecasting Function (SAFE + 1D OUTPUT)
# =========================
def forecast_prices(data, steps):
    try:
        model = ARIMA(data["Close"], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)

        return np.array(forecast).reshape(-1)   # FORCE 1D ARRAY

    except Exception as e:
        st.error(f"Forecast Error: {e}")
        return np.zeros(steps)

forecast = forecast_prices(df, forecast_days)

# =========================
# Forecast Table
# =========================
st.subheader("Forecast for Next Days")

forecast_df = pd.DataFrame({
    "Date": pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    ),
    "Predicted Price": forecast
})

st.dataframe(forecast_df)

# =========================
# Plot
# =========================
fig, ax = plt.subplots()
ax.plot(df["Close"], label="Historical")
ax.plot(forecast_df["Date"], forecast_df["Predicted Price"], label="Forecast")
ax.legend()
st.pyplot(fig)

# =========================
# Alert Section
# =========================
st.subheader("Set Alert Threshold")

threshold_percent = st.number_input("Alert if price change > (%)", value=5.0)

if st.button("Check Alert"):
    predicted_next = forecast[0]
    change = ((predicted_next - current_price) / current_price) * 100

    if abs(change) > threshold_percent:
        st.success(
            f"BTC Alert 🚨\n"
            f"Current: ${current_price:.2f}\n"
            f"Predicted: ${predicted_next:.2f}\n"
            f"Change: {change:.2f}%"
        )
    else:
        st.info("No significant price movement predicted.")

# =========================
# Backtesting (Last 30 Days) — FIXED
# =========================
st.subheader("Backtesting (Last 30 Days)")

if len(df) >= 60:
    test_data = df["Close"][-30:].to_numpy().reshape(-1)

    train_data = df[:-30]
    back_forecast = forecast_prices(train_data, 30)

    # SAFETY SHAPE CHECK
    st.write("Test Shape:", test_data.shape)
    st.write("Forecast Shape:", back_forecast.shape)

    # Force both 1D
    test_data = test_data.reshape(-1)
    back_forecast = back_forecast.reshape(-1)

    if len(test_data) == len(back_forecast):
        mae = np.mean(np.abs(test_data - back_forecast))
        st.write(f"📉 Mean Absolute Error (MAE): **${mae:.2f}**")
    else:
        st.error("Backtest length mismatch!")
else:
    st.warning("Not enough data for backtesting (need at least 60 days)")
