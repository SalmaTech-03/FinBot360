import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
from transformers import pipeline
import datetime

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="FinBot 360",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 FinBot 360 - Advanced Financial Assistant")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["Live Dashboard", "Portfolio Analysis", "Sentiment Analysis", "Stock Forecasting"]
)

# -----------------------------
# Stock Data Loader
# -----------------------------
@st.cache_data
def load_stock_data(ticker, period="6mo"):
    return yf.download(ticker, period=period)

# -----------------------------
# Live Dashboard
# -----------------------------
if app_mode == "Live Dashboard":
    st.subheader("📊 Live Stock Dashboard")

    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")
    df = load_stock_data(ticker)

    st.line_chart(df['Close'])

    st.write("Recent Data:")
    st.dataframe(df.tail())

# -----------------------------
# Portfolio Analysis
# -----------------------------
elif app_mode == "Portfolio Analysis":
    st.subheader("💼 Portfolio Analysis")

    uploaded_file = st.file_uploader("Upload Portfolio CSV (columns: Ticker, Shares, BuyPrice)", type="csv")

    if uploaded_file:
        portfolio = pd.read_csv(uploaded_file)
        st.write("Uploaded Portfolio:", portfolio)

        current_values = []
        for i, row in portfolio.iterrows():
            ticker = row['Ticker']
            shares = row['Shares']
            buy_price = row['BuyPrice']
            current_price = yf.download(ticker, period="1d")['Close'][-1]
            pnl = (current_price - buy_price) * shares
            current_values.append([ticker, shares, buy_price, current_price, pnl])

        portfolio_df = pd.DataFrame(current_values, columns=["Ticker", "Shares", "BuyPrice", "CurrentPrice", "PnL"])
        st.dataframe(portfolio_df)

        total_pnl = portfolio_df['PnL'].sum()
        st.metric("Total Profit/Loss", f"${total_pnl:,.2f}")

# -----------------------------
# Sentiment Analysis
# -----------------------------
elif app_mode == "Sentiment Analysis":
    st.subheader("📰 Financial Sentiment Analysis")

    stock_news = st.text_area("Paste financial news or article here:")

    if stock_news:
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        results = sentiment_pipeline(stock_news)
        st.write("Sentiment:", results[0]['label'])
        st.write("Confidence:", f"{results[0]['score']:.2f}")

# -----------------------------
# Stock Forecasting with LSTM
# -----------------------------
elif app_mode == "Stock Forecasting":
    st.subheader("📈 Stock Price Forecasting with LSTM")

    ticker = st.text_input("Enter Stock Symbol for Forecasting:", "AAPL")
    df = load_stock_data(ticker, "1y")

    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step), 0])
            Y.append(dataset[i+time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['Close'].values[-len(y_test):], name="Actual"))
    fig.add_trace(go.Scatter(y=predictions.flatten(), name="Predicted"))
    fig.update_layout(title="Stock Price Forecast vs Actual", xaxis_title="Time", yaxis_title="Price ($)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Always Visible Chatbot
# -----------------------------
st.markdown("---")
st.subheader("🤖 Financial Chatbot")

user_input = st.text_input("Ask me anything about finance:", "")

if user_input:
    try:
        response = requests.post(
            "https://api.gemini.com/v1/chat",
            json={"prompt": user_input}
        )
        if response.status_code == 200:
            st.write("**Bot:**", response.json().get("response", "No response"))
        else:
            st.write("**Bot:** Sorry, API unavailable right now.")
    except Exception as e:
        st.write("**Bot Error:**", str(e))
