import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import google.generativeai as genai
from datetime import timedelta

# -----------------------------
# Page Config & API Keys
# -----------------------------
st.set_page_config(
    page_title="FinBot 360",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📈 FinBot 360 - Advanced Financial Assistant")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except (KeyError, FileNotFoundError):
    st.sidebar.error("⚠️ Gemini API Key not found.")
    GEMINI_AVAILABLE = False

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📌 Financial Tools")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["Live Dashboard", "Portfolio Analysis", "Sentiment Analysis", "Stock Forecasting"]
)

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data(show_spinner="Fetching data...")
def load_stock_data(ticker, period="1y"):
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

@st.cache_data(show_spinner=False)
def get_ai_insights(_model, prompt):
    if not GEMINI_AVAILABLE: return "AI model is unavailable."
    try:
        response = _model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Insight generation failed: {e}"

# -----------------------------
# Live Dashboard
# -----------------------------
if app_mode == "Live Dashboard":
    st.header("📊 Live Stock Dashboard")
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=3)
    with col2:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    if ticker:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if not df.empty:
            st.subheader(f"Price Chart for {ticker}")
            st.line_chart(df['Close'])
            st.subheader("Recent Data")
            st.dataframe(df.tail())
        else:
            st.error("No data found for this ticker. Please check the symbol.")

# -----------------------------
# Portfolio Analysis
# -----------------------------
elif app_mode == "Portfolio Analysis":
    st.header("💼 Portfolio Analysis")
    # ... Your working portfolio code here ...

# -----------------------------
# Sentiment Analysis
# -----------------------------
elif app_mode == "Sentiment Analysis":
    st.header("📰 Financial Sentiment Analysis")
    # ... Your working sentiment analysis code here ...

# -----------------------------
# Stock Forecasting with Tabular Output & AI Insights
# -----------------------------
elif app_mode == "Stock Forecasting":
    st.header("📈 Stock Price Forecasting")
    
    ticker_forecast = st.text_input("Enter Stock Symbol for Forecasting:", "NVDA").upper()

    if st.button("Generate Forecast", key="gen_forecast"):
        if not ticker_forecast:
            st.warning("Please enter a stock symbol.")
        else:
            with st.spinner(f"Training model for {ticker_forecast}... This can take a minute."):
                data = load_stock_data(ticker_forecast, period="5y")
                
                if data.empty or len(data) < 80:
                    st.error("Not enough historical data to generate a reliable forecast.")
                else:
                    data_close = data[['Close']].copy()
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(data_close)

                    # Create sequences for training
                    time_step = 60
                    x_train, y_train = [], []
                    for i in range(time_step, len(scaled_data)):
                        x_train.append(scaled_data[i-time_step:i, 0])
                        y_train.append(scaled_data[i, 0])
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                    
                    # Train model
                    model = create_lstm_model((x_train.shape[1], 1))
                    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

                    # Forecast future days
                    last_60_days = scaled_data[-60:]
                    future_preds = []
                    current_batch = last_60_days.reshape(1, time_step, 1)

                    for i in range(10): # Predict next 10 days
                        next_pred = model.predict(current_batch, verbose=0)[0]
                        future_preds.append(next_pred)
                        current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)

                    future_preds = scaler.inverse_transform(future_preds).flatten()
                    last_date = data.index[-1]
                    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, 11)])

                    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_preds})
                    forecast_df['% Change'] = forecast_df['Predicted Price'].pct_change() * 100
                    st.session_state.forecast_df = forecast_df
                    st.session_state.ticker_forecast = ticker_forecast
    
    # Display the table and AI insights if they exist in the session state
    if 'forecast_df' in st.session_state and st.session_state.forecast_df is not None:
        st.markdown("---")
        st.subheader(f"10-Day Forecast for {st.session_state.ticker_forecast}")
        
        forecast_df = st.session_state.forecast_df
        
        # Display Tabular Column
        st.dataframe(
            forecast_df.style.format({
                "Predicted Price": "${:,.2f}",
                "% Change": "{:,.2f}%"
            }).applymap(lambda x: 'color: green' if x > 0 else 'color: red', subset=['% Change']),
            use_container_width=True
        )

        # Display Text Forecasting (AI Insights)
        st.subheader("🤖 AI-Powered Insights")
        with st.spinner("Generating AI analysis..."):
            if GEMINI_AVAILABLE:
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
                Analyze this 10-day stock forecast for {st.session_state.ticker_forecast} and provide a brief, easy-to-understand summary for a retail investor.
                
                Forecast Table:
                {forecast_df.to_string()}

                Based on the data, describe the potential trend (e.g., "appears bullish", "shows signs of a slight downturn").
                Mention the potential price range over this period. Keep it concise.
                End with a clear disclaimer that this is not financial advice.
                """
                ai_summary = get_ai_insights(gemini_model, prompt)
                st.markdown(ai_summary)
            else:
                st.warning("AI Insights are unavailable as the Gemini API key is not configured.")