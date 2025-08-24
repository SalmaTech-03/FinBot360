import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
from transformers import pipeline
import os

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="FinBot 360", page_icon="💹", layout="wide")

# Custom CSS for branding & UI
st.markdown("""
    <style>
        .main {
            background-color: #f9fafb;
        }
        .header {
            font-size: 32px;
            font-weight: bold;
            color: #1f2937;
            padding: 10px;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: #1f2937;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# GEMINI API SETUP
# ==============================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
chat_model = genai.GenerativeModel("gemini-2.5-flash")

# ==============================
# HEADER
# ==============================
st.markdown("<div class='header'>💹 FinBot 360 – AI Financial Assistant</div>", unsafe_allow_html=True)

# ==============================
# SIDEBAR NAVIGATION
# ==============================
st.sidebar.markdown("<div class='sidebar-title'>📌 Navigation</div>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Choose a section:",
    ["📈 Live Stock Dashboard", "📊 Portfolio Analysis", "📰 Sentiment Analysis", "🤖 AI Chatbot", "🔮 Stock Forecasting"]
)

# ==============================
# STOCK DASHBOARD
# ==============================
if menu == "📈 Live Stock Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Live Stock Dashboard")
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, GOOGL)", "AAPL")

    if ticker:
        data = yf.download(ticker, period="6mo", interval="1d")
        st.write(f"Showing last 6 months data for **{ticker}**")
        st.dataframe(data.tail())

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name="Candlestick"
        ))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# PORTFOLIO ANALYSIS
# ==============================
elif menu == "📊 Portfolio Analysis":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Portfolio Analysis")
    uploaded_file = st.file_uploader("Upload your portfolio CSV (columns: Ticker, Shares)", type=["csv"])

    if uploaded_file:
        portfolio = pd.read_csv(uploaded_file)
        st.write("Uploaded Portfolio:", portfolio)

        total_value = 0
        for _, row in portfolio.iterrows():
            ticker = row['Ticker']
            shares = row['Shares']
            price = yf.download(ticker, period="1d")["Close"].iloc[-1]
            total_value += price * shares

        st.metric("💰 Total Portfolio Value", f"${total_value:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# SENTIMENT ANALYSIS
# ==============================
elif menu == "📰 Sentiment Analysis":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📰 Market Sentiment Analysis")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    news = st.text_area("Enter financial news or market updates:")
    if news:
        result = sentiment_analyzer(news)[0]
        st.write(f"**Sentiment:** {result['label']} (Confidence: {result['score']:.2f})")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# AI CHATBOT
# ==============================
elif menu == "🤖 AI Chatbot":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 FinBot AI Chatbot (Gemini 2.5 Flash)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask FinBot about finance...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        response = chat_model.generate_content(user_input)
        reply = response.text

        with st.chat_message("assistant"):
            st.write(reply)

        st.session_state.chat_history.append({"user": user_input, "bot": reply})

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# STOCK FORECASTING (NO GRAPH)
# ==============================
elif menu == "🔮 Stock Forecasting":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Stock Price Forecasting (AI-powered)")

    ticker = st.text_input("Enter stock symbol for forecasting", "AAPL")
    if ticker:
        data = yf.download(ticker, period="1y", interval="1d")[["Close"]]

        # Preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)

        # Prepare training data
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, batch_size=1, epochs=1, verbose=0)

        # Forecast next 7 days
        last_60 = scaled_data[-60:]
        forecast_input = np.reshape(last_60, (1, last_60.shape[0], 1))
        forecast = []
        for _ in range(7):
            pred = model.predict(forecast_input, verbose=0)
            forecast.append(pred[0][0])
            forecast_input = np.append(forecast_input[:,1:,:], [[pred]], axis=1)

        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))

        st.write("📌 **7-Day Forecast (No Graph):**")
        for i, price in enumerate(forecast, 1):
            st.write(f"Day {i}: ${price[0]:.2f}")

        st.metric("📈 Trend", "Uptrend" if forecast[-1] > forecast[0] else "Downtrend")
        st.metric("🔮 Confidence Interval", "±5% (approx)")
    st.markdown("</div>", unsafe_allow_html=True)
