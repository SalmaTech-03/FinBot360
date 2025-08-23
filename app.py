import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

# ==============================
# API Keys (set them in Streamlit secrets or here for local testing)
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="FinBot 360", layout="wide")
st.title("📊 FinBot 360 - Advanced Financial Assistant")

# Sidebar for stock selection
st.sidebar.header("Stock Forecasting")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL")
n_days = st.sidebar.slider("Forecast Days", 5, 60, 30)

# ==============================
# Data Loading
# ==============================
@st.cache_data
def load_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=365*2)  # 2 years data
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(ticker)

if data.empty:
    st.error("⚠️ No stock data found. Try another ticker.")
    st.stop()

st.subheader(f"📈 {ticker} Stock Price Data (Last 2 Years)")
st.line_chart(data["Close"])

# ==============================
# Preprocessing for LSTM
# ==============================
df = data[["Close"]].values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ==============================
# Build LSTM Model
# ==============================
@st.cache_resource
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

model = build_model()

# Train model
with st.spinner("⏳ Training LSTM model... Please wait"):
    model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

# ==============================
# Predictions
# ==============================
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Align predictions with actual data
train_plot = np.empty_like(df)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict)+time_step, :] = train_predict

test_plot = np.empty_like(df)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+(time_step*2):len(df), :] = test_predict

# ==============================
# Forecast Future Prices
# ==============================
future_input = test_data[-time_step:].reshape(1, -1)
future_input = list(future_input[0])
future_output = []

for _ in range(n_days):
    x_input = np.array(future_input[-time_step:]).reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    future_output.append(yhat[0][0])
    future_input.append(yhat[0][0])

future_output = scaler.inverse_transform(np.array(future_output).reshape(-1, 1))

# ==============================
# Visualization
# ==============================
st.subheader("🔮 LSTM Stock Forecasting")

fig = go.Figure()
fig.add_trace(go.Scatter(y=df.flatten(), mode='lines', name="Actual Price"))
fig.add_trace(go.Scatter(y=train_plot.flatten(), mode='lines', name="Train Predict"))
fig.add_trace(go.Scatter(y=test_plot.flatten(), mode='lines', name="Test Predict"))

future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=n_days)
fig.add_trace(go.Scatter(x=future_dates, y=future_output.flatten(), mode='lines+markers', name="Future Forecast"))

fig.update_layout(title=f"{ticker} Stock Price Forecast",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

st.success("✅ Forecasting complete!")
