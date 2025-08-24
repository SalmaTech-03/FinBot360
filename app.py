import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import datetime
import google.generativeai as genai

# =========================
# Gemini API Configuration
# =========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

# =========================
# Stock Forecasting Module
# =========================
def prepare_data(stock_symbol, start, end):
    df = yf.download(stock_symbol, start=start, end=end)
    data = df[['Close']]
    return data

def build_lstm_model(trainX, trainY):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_stock(stock_symbol):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365*2)

    data = prepare_data(stock_symbol, start, end)
    if data.empty:
        return None, None, None, None

    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Train-Test Split
    train_size = int(len(scaled_data)*0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size-60:]

    # Create sequences
    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(time_step, len(dataset)):
            X.append(dataset[i-time_step:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data)
    X_test, Y_test = create_dataset(test_data)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Train Model
    model = build_lstm_model(X_train, Y_train)
    model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=0)

    # Forecast Next Day
    last_60 = scaled_data[-60:]
    X_input = np.reshape(last_60, (1, last_60.shape[0], 1))
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    # Confidence Interval
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    Y_test = scaler.inverse_transform(Y_test.reshape(-1,1))

    residuals = Y_test.flatten() - predictions.flatten()
    error_std = np.std(residuals)

    lower_bound = predicted_price - 1.96*error_std
    upper_bound = predicted_price + 1.96*error_std

    # Trend Direction
    last_price = dataset[-1][0]
    trend = "📈 Up" if predicted_price > last_price else "📉 Down"

    return predicted_price, lower_bound, upper_bound, trend

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="FinBot 360", layout="wide")

# Sidebar
st.sidebar.title("📊 Navigation")
menu = st.sidebar.radio("Go to", ["Live Dashboard", "Portfolio", "Sentiment Analysis", "Stock Forecasting"])

# Main Chatbot Area
st.title("🤖 FinBot 360 - AI Financial Assistant")

user_input = st.text_area("💬 Ask FinBot (AI-Powered Chatbot):", placeholder="e.g., What is the outlook for AAPL?")
if st.button("Get Response"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = get_gemini_response(user_input)
        st.markdown(f"**FinBot:** {response}")
    else:
        st.warning("Please enter a question.")

# =========================
# Modules
# =========================
if menu == "Live Dashboard":
    st.subheader("📊 Live Stock Dashboard")
    st.info("This module can be expanded with real-time stock/market updates.")

elif menu == "Portfolio":
    st.subheader("💼 Portfolio Analysis")
    st.info("Portfolio insights, allocation, and risk metrics go here.")

elif menu == "Sentiment Analysis":
    st.subheader("📰 Sentiment Analysis")
    st.info("This module analyzes financial news and market sentiment.")

elif menu == "Stock Forecasting":
    st.subheader("🔮 Stock Forecasting (Next Day Prediction)")
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")

    if st.button("Run Forecast"):
        with st.spinner("Training LSTM and forecasting..."):
            predicted_price, lower, upper, trend = forecast_stock(stock_symbol)

        if predicted_price is not None:
            st.metric("Predicted Next-Day Price", f"${predicted_price:.2f}")
            st.metric("Confidence Interval (95%)", f"${lower:.2f} - ${upper:.2f}")
            st.metric("Trend Direction", trend)

            st.markdown("---")
            st.subheader("📝 Insights")
            st.write(
                f"The model predicts **{stock_symbol}** will close at **${predicted_price:.2f}** tomorrow. "
                f"There's a 95% confidence the price will fall between **${lower:.2f}** and **${upper:.2f}**. "
                f"Overall trend: **{trend}**."
            )
        else:
            st.error("No data found for this stock symbol.")
