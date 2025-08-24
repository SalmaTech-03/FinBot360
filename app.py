import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import datetime
import google.generativeai as genai
from transformers import pipeline

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Advanced Financial Assistant", layout="wide")

# Gemini setup
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Sentiment Analysis (FinBERT)
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ------------------ NAVIGATION ------------------
menu = ["📊 Live Dashboard", "📈 Stock Forecasting", "📰 Sentiment Analysis", "💼 Portfolio Analysis", "🤖 Chatbot"]
choice = st.sidebar.radio("Navigate", menu)

# ------------------ LIVE DASHBOARD ------------------
if choice == "📊 Live Dashboard":
    st.title("📊 Live Stock Dashboard")
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, INFY.NS):", "AAPL")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=1)

    if st.button("Load Data"):
        data = yf.download(ticker, period=period, interval="1d")
        st.subheader(f"Stock Data for {ticker}")
        st.write(data.tail())

        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ------------------ STOCK FORECASTING ------------------
elif choice == "📈 Stock Forecasting":
    st.title("📈 Stock Price Forecasting (LSTM Model)")

    ticker = st.text_input("Enter Stock Ticker:", "AAPL")
    n_days = st.slider("Forecast Days:", 1, 15, 5)

    if st.button("Run Forecast"):
        df = yf.download(ticker, period="2y")
        data = df[['Close']].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, batch_size=32, epochs=3, verbose=0)

        # Forecast
        last_60 = scaled_data[-60:]
        last_60 = np.reshape(last_60, (1, last_60.shape[0], 1))

        forecasted_prices = []
        for _ in range(n_days):
            pred = model.predict(last_60, verbose=0)
            forecasted_prices.append(pred[0, 0])
            last_60 = np.append(last_60[:, 1:, :], [[pred]], axis=1)

        forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))

        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": forecasted_prices.flatten()
        })

        # 1️⃣ Forecast Table
        st.subheader("📅 Forecast Table")
        st.dataframe(forecast_df.style.format({"Predicted Price": "{:.2f}"}))

        # 2️⃣ Key Metrics
        st.subheader("📌 Key Metrics")
        next_day = forecast_df.iloc[0]["Predicted Price"]
        last_close = df["Close"].iloc[-1]
        pct_change = ((next_day - last_close) / last_close) * 100
        risk_score = np.abs(pct_change) * 2  

        col1, col2, col3 = st.columns(3)
        col1.metric("Next Day Price", f"${next_day:.2f}")
        col2.metric("Expected Change", f"{pct_change:.2f}%")
        col3.metric("Risk Score", f"{risk_score:.1f}/10")

        # 3️⃣ AI Insights
        st.subheader("🤖 AI Insights")
        prompt = f"""
        Analyze the following stock forecast for {ticker}:
        Last close: {last_close:.2f}, Next day prediction: {next_day:.2f}, Change: {pct_change:.2f}%
        Provide a short investment insight in 2-3 lines.
        """
        ai_response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        st.write(ai_response.text)

# ------------------ SENTIMENT ANALYSIS ------------------
elif choice == "📰 Sentiment Analysis":
    st.title("📰 Stock Market News Sentiment Analysis")
    news = st.text_area("Paste financial news/article text here:")

    if st.button("Analyze Sentiment"):
        if news:
            result = sentiment_pipeline(news)[0]
            st.write(f"**Sentiment:** {result['label']} (score: {result['score']:.2f})")
        else:
            st.warning("Please enter some text!")

# ------------------ PORTFOLIO ANALYSIS ------------------
elif choice == "💼 Portfolio Analysis":
    st.title("💼 Portfolio Upload & Analysis")
    uploaded_file = st.file_uploader("Upload CSV (Ticker, Shares)", type=["csv"])

    if uploaded_file:
        portfolio = pd.read_csv(uploaded_file)
        st.write("Uploaded Portfolio:", portfolio)

        total_value = 0
        for i, row in portfolio.iterrows():
            ticker = row["Ticker"]
            shares = row["Shares"]
            price = yf.download(ticker, period="1d")["Close"].iloc[-1]
            value = shares * price
            portfolio.loc[i, "Latest Price"] = price
            portfolio.loc[i, "Value"] = value
            total_value += value

        st.subheader("📊 Portfolio Performance")
        st.dataframe(portfolio)

        st.success(f"💰 Total Portfolio Value: ${total_value:,.2f}")

# ------------------ CHATBOT ------------------
elif choice == "🤖 Chatbot":
    st.title("🤖 Gemini AI Chatbot - Financial Assistant")
    user_input = st.text_area("Ask me anything about stocks/finance:")

    if st.button("Get Answer"):
        if user_input:
            response = genai.GenerativeModel("gemini-2.5-flash").generate_content(user_input)
            st.write(response.text)
        else:
            st.warning("Please enter a question!")
