import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
import datetime

# ------------------------------
# Gemini Setup
# ------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Stock Price Analyzer", layout="wide")

st.title("📈 Stock Price Analysis & Forecasting with Gemini 2.5-Flash")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

if st.button("Analyze Stock"):
    try:
        # ------------------------------
        # Fetch Stock Data
        # ------------------------------
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found. Please try another ticker.")
        else:
            st.success(f"✅ Data fetched for {ticker}")
            st.write(data.tail())

            # ------------------------------
            # Plot Historical Prices
            # ------------------------------
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Closing Price"))
            fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------
            # Preprocess Data for LSTM
            # ------------------------------
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]

            def create_dataset(dataset, time_step=60):
                X, y = [], []
                for i in range(len(dataset) - time_step - 1):
                    X.append(dataset[i:(i + time_step), 0])
                    y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(y)

            time_step = 60
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # ------------------------------
            # Build LSTM Model
            # ------------------------------
            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model_lstm.compile(optimizer="adam", loss="mean_squared_error")

            model_lstm.fit(X_train, y_train, batch_size=32, epochs=1, verbose=0)

            predictions = model_lstm.predict(X_test)
            predictions = scaler.inverse_transform(predictions)

            # ------------------------------
            # Plot Predictions
            # ------------------------------
            train = data[:train_size]
            valid = data[train_size:]
            valid["Predictions"] = predictions

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=train.index, y=train["Close"], mode="lines", name="Training"))
            fig2.add_trace(go.Scatter(x=valid.index, y=valid["Close"], mode="lines", name="Actual"))
            fig2.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"], mode="lines", name="Predicted"))
            fig2.update_layout(title=f"{ticker} Stock Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig2, use_container_width=True)

            # ------------------------------
            # Gemini Financial Insight
            # ------------------------------
            with st.spinner("🔍 Analyzing with Gemini 2.5-Flash..."):
                prompt = f"Provide financial insights for {ticker} stock based on historical trends and AI prediction."
                response = model.generate_content(prompt)
                st.subheader("📊 Gemini AI Financial Insight")
                st.write(response.text)

    except Exception as e:
        st.error(f"Error: {e}")
