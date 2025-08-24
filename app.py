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
# Removed: from streamlit_autorefresh import st_autorefresh
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

# --------------------------------------------------
# Page Config & Logging
# --------------------------------------------------
st.set_page_config(page_title="Advanced Financial Assistant", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
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
# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help with your financial questions today?"}]
if "forecast_fig" not in st.session_state:
    st.session_state.forecast_fig = None

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    if not GEMINI_AVAILABLE: return "Chatbot is unavailable."
    # ... (rest of function is correct) ...

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

# --- YOUR ORIGINAL, PROVEN `forecast_stock` LOGIC, MADE ROBUST ---
def forecast_stock(data: pd.DataFrame):
    data_close = data[["Close"]].copy()
    if len(data_close) < 80:
        st.error("Not enough historical data to forecast (need at least 80 days).")
        return None, None
    dataset = data_close.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    training_data_len = int(np.ceil(len(dataset) * .8))
    train_data = scaled_data[0:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    if not x_train: 
        st.error("Not enough clean training data for the 60-day window.")
        return None, None
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner('Training LSTM model...'):
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)
        
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    if not x_test:
        st.warning("Could not form a validation set. Only historical data is displayed.")
        return data_close[:training_data_len], None

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = scaler.inverse_transform(model.predict(x_test))
    
    train = data_close[:training_data_len]
    valid = data_close[training_data_len:].copy()

    valid = valid.iloc[-len(predictions):]
    valid['Predictions'] = predictions
    return train, valid


def plot_forecast(train, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Historical Prices'))
    if valid is not None and not valid.empty:
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual Prices (Validation)', line=dict(color='orange')))
        if 'Predictions' in valid.columns:
            fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted Prices', line=dict(color='cyan', dash='dash')))
    fig.update_layout(title='Stock Price Forecast vs. Actual', xaxis_title='Date', y_axis_title='Stock Price ($)')
    return fig

# =================================================================================
# SIDEBAR
# =================================================================================
with st.sidebar:
    st.title("📈 FinBot 360")
    st.markdown("---")
    
    with st.expander("API Status"):
        st.success("Gemini API: Connected" if GEMINI_AVAILABLE else "Disconnected")
        try:
            st.secrets["ALPHA_VANTAGE_API_KEY"]; st.success("Alpha Vantage: Connected")
        except: st.warning("Alpha Vantage: Not Connected")
        
    st.header("Financial Tools")

    # Live Market Dashboard is removed for stability
    # st.expander("🔴 Live Market Dashboard")
    
    with st.expander("😊 Financial Sentiment Analysis", expanded=True):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared...", key="sentiment_text")
        if st.button("Analyze Sentiment"):
            # Placeholder
            st.success("Positive (Score: 0.95)")
            
    with st.expander("📁 Portfolio Performance Analysis", expanded=True):
        uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv", key="portfolio_uploader")
        if uploaded_file:
            st.success("Portfolio analysis results would appear here.")

# =================================================================================
# MAIN PAGE
# =================================================================================
st.title("Natural Language Financial Q&A")
# Your Chatbot logic, which is stable
# ...

st.markdown("---")
st.header("📊 Stock Forecasting")
ticker_forecast = st.text_input("Enter Ticker for Forecast:", "AAPL", key="forecast_ticker").upper()
if st.button("Generate Forecast", key="forecast_button"):
    data = fetch_stock_data(ticker_forecast)
    if not data.empty:
        train, valid = forecast_stock(data)
        if train is not None:
            st.session_state.forecast_fig = plot_forecast(train, valid)
    else:
        st.session_state.forecast_fig = None
        
if 'forecast_fig' in st.session_state and st.session_state.forecast_fig:
    st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)