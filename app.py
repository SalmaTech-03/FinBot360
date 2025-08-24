import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import google.generativeai as genai
import logging
from io import StringIO

# --- Imports for the Sidebar Tools ---
from streamlit_autorefresh import st_autorefresh
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

# --------------------------------------------------
# Page Config & Logging
# --------------------------------------------------
st.set_page_config(page_title="FinBot 360", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------
# API Keys & Session State
# --------------------------------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except (KeyError, FileNotFoundError):
    st.sidebar.error("⚠️ Gemini API Key not found.")
    GEMINI_AVAILABLE = False
try:
    AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
    AV_AVAILABLE = True
except (KeyError, FileNotFoundError):
    AV_AVAILABLE = False

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help with your financial questions today?"}]
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    if not GEMINI_AVAILABLE: return "Chatbot is unavailable."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(show_spinner="Fetching historical data...")
def fetch_stock_data(ticker, period="5y"):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return df.dropna()

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), Dropout(0.2),
        LSTM(50, return_sequences=False), Dropout(0.2),
        Dense(25), Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def forecast_future(model, scaler, last_60_days_scaled, future_days=10):
    future_predictions = []
    current_batch = last_60_days_scaled.reshape(1, 60, 1)
    for _ in range(future_days):
        next_prediction = model.predict(current_batch, verbose=0)[0]
        future_predictions.append(next_prediction)
        current_batch = np.append(current_batch[:, 1:, :], [[next_prediction]], axis=1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions.flatten()

def get_ai_insights(ticker, forecast_df):
    if not GEMINI_AVAILABLE: return "AI Insights are unavailable."
    prompt = f"""
    Analyze the following 10-day stock forecast for {ticker} for a retail investor.
    The table shows the predicted closing price for the next 10 business days.
    **Forecast Data:**
    {forecast_df.to_string()}
    **Instructions:**
    1. Summarize the overall trend (e.g., bullish, bearish, sideways).
    2. Mention the predicted price range.
    3. Conclude with a brief, balanced perspective.
    4. Include this disclaimer at the end: "Disclaimer: This is an AI-generated analysis and not financial advice."
    """
    return get_llm_response(prompt)

# =================================================================================
# MAIN APPLICATION LAYOUT & LOGIC
# =================================================================================
st.title("📈 Advanced Financial Assistant")

if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None

# Using columns to organize the main layout
main_col1, main_col2 = st.columns([1, 2])

with main_col1:
    st.subheader("Financial Tools")
    
    with st.expander("🔴 Live Market Dashboard"):
        ticker_live = st.text_input("Enter Ticker:", "IBM", key="live_ticker").upper()
        # Your live market dashboard logic here...

    with st.expander("😊 Financial Sentiment Analysis"):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared...", key="sentiment_text")
        if st.button("Analyze Sentiment", key="sentiment_button"):
            st.write("Sentiment results would show here.")

    with st.expander("📁 Portfolio Performance Analysis"):
        uploaded_file = st.file_uploader("Upload portfolio CSV/XLSX", type=['csv', 'xlsx'], key="portfolio_uploader")
        if uploaded_file:
            st.success("Portfolio analysis would be shown here.")


with main_col2:
    st.subheader("📊 Stock Forecasting")
    ticker_forecast = st.text_input("Enter Ticker for Forecast:", "AAPL", key="forecast_ticker").upper()
    
    if st.button("Generate Forecast", key="forecast_button"):
        with st.spinner("Running full forecast analysis..."):
            data = fetch_stock_data(ticker_forecast, "2010-01-01", pd.to_datetime("today").strftime("%Y-%m-%d"))
            
            if data.empty or len(data) < 80:
                st.error(f"Not enough historical data available for {ticker_forecast} to generate a reliable forecast.")
                st.session_state.forecast_data = None
            else:
                data_close = data[["Close"]].copy()
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data_close)
                
                training_data_len = int(np.ceil(len(scaled_data) * 0.8))
                train_data = scaled_data[:training_data_len]
                x_train, y_train = [], []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                
                model = create_lstm_model((x_train.shape[1], 1))
                model.fit(x_train, y_train, batch_size=32, epochs=8, verbose=0)
                
                last_60_days = scaled_data[-60:]
                future_preds = forecast_future(model, scaler, last_60_days, future_days=10)
                
                last_date = data_close.index[-1]
                # --- FIX: Generate business day date range ---
                future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=10)
                
                forecast_df = pd.DataFrame(index=future_dates, data=future_preds, columns=["Predicted Price"])
                forecast_df["% Change"] = forecast_df["Predicted Price"].pct_change() * 100
                
                returns = data_close['Close'].pct_change().dropna()
                risk_score = returns.std() * np.sqrt(252)
                
                ai_summary = get_ai_insights(ticker_forecast, forecast__df)
                
                st.session_state.forecast_data = {
                    "ticker": ticker_forecast,
                    "last_close": data_close['Close'].iloc[-1],
                    "forecast_df": forecast_df,
                    "risk_score": risk_score,
                    "ai_summary": ai_summary,
                }
    
    # Display the results if they exist in the session state
    if st.session_state.forecast_data:
        ticker = st.session_state.forecast_data["ticker"]
        forecast_df = st.session_state.forecast_data["forecast_df"]
        last_close = st.session_state.forecast_data["last_close"]
        
        st.markdown(f"### Forecast for **{ticker}**")
        
        # Display Key Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        next_day_price = forecast_df["Predicted Price"].iloc[0]
        next_day_change = ((next_day_price - last_close) / last_close) * 100
        
        metric_col1.metric("Next Day Prediction", f"${next_day_price:,.2f}", f"{next_day_change:,.2f}% vs Last Close")
        metric_col2.metric("10-Day Predicted Change", f"{forecast_df['% Change'].sum():,.2f}%")
        metric_col3.metric("Volatility Risk Score", f"{st.session_state.forecast_data['risk_score']:.2%}")
        
        # Display Tabs for AI Insights and Forecast Table
        tab1, tab2 = st.tabs(["🤖 AI Insights", "📈 Forecast Table"])
        with tab1:
            st.markdown(st.session_state.forecast_data["ai_summary"])
        with tab2:
            st.dataframe(forecast_df.style.format({
                "Predicted Price": "${:,.2f}",
                "% Change": "{:.2f}%"
            }).applymap(lambda x: 'color: green' if x > 0 else 'color: red', subset=['% Change']))

st.markdown("---")
# =================================================================================
# Chatbot Interface
# =================================================================================
st.header("💬 Natural Language Financial Q&A")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you?"}]
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})