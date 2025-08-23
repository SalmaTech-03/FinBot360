import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import datetime

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="FinBot 360", layout="wide")
st.title("📊 FinBot 360 - AI Financial Dashboard")

# ----------------------------
# Caching Stock Data
# ----------------------------
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# ----------------------------
# Stock Data Section
# ----------------------------
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

if st.sidebar.button("Load Data"):
    df = load_data(ticker, start_date, end_date)
    st.subheader(f"Stock Data for {ticker}")
    st.write(df.tail())

    # Plot stock price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price"))
    fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Forecasting Section (LSTM)
    # ----------------------------
    st.subheader("📈 Stock Price Forecasting (LSTM)")

    # Prepare data
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size-60:]

    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset)-time_step):
            X.append(dataset[i:(i+time_step), 0])
            y.append(dataset[i+time_step, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # Reshape for LSTM [samples, time_steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Lightweight Model (to reduce memory)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train only for few epochs
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    # Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot forecast
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=actual.flatten(), mode="lines", name="Actual Price"))
    fig2.add_trace(go.Scatter(y=predictions.flatten(), mode="lines", name="Predicted Price"))
    fig2.update_layout(title="LSTM Stock Price Prediction", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig2, use_container_width=True)

    st.success("✅ Forecasting complete!")
