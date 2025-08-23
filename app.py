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

# ✅ Integrated Logging & Debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Financial Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API KEY CONFIGURATION ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except (KeyError, FileNotFoundError):
    st.error("⚠️ Gemini API Key not found. Please add it to your Streamlit secrets.", icon="🚨")
    GEMINI_AVAILABLE = False

# =================================================================================
# ALL HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    if not GEMINI_AVAILABLE: return "Chatbot is unavailable because the Gemini API key is not configured."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}") 
        return f"An error occurred. **Specific API Error:** {e}"

@st.cache_resource
def load_sentiment_model():
    # Use TensorFlow explicitly to help with framework resolution
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="tf")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

@st.cache_data
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date)
    # Critical: Drop rows with any missing values before returning
    data.dropna(inplace=True)
    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.", icon="❌")
        return pd.DataFrame()
    return data

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), Dropout(0.2),
        LSTM(50, return_sequences=False), Dropout(0.2),
        Dense(25), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- YOUR ORIGINAL, WORKING FORECAST LOGIC ---
def forecast_stock(data: pd.DataFrame):
    data_close = data[['Close']]
    dataset = data_close.values
    training_data_len = int(np.ceil(len(dataset) * .8))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    if not x_train: 
        st.error("Not enough clean training data to create a forecast.")
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
        st.warning("Could not form a test set. Only historical data is displayed.")
        return data_close[:training_data_len], None

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = scaler.inverse_transform(model.predict(x_test))
    
    train = data_close[:training_data_len]
    valid = data_close[training_data_len:].copy()
    
    # Handle any small length mismatches
    pred_len = len(predictions)
    valid_len = len(valid)
    if pred_len != valid_len:
        valid = valid.iloc[-pred_len:]
    
    valid['Predictions'] = predictions
    return train, valid

def plot_forecast(train, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Historical Prices'))
    if valid is not None and not valid.empty:
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual Prices (Validation)', line=dict(color='orange')))
        if 'Predictions' in valid.columns:
            fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted Prices', line=dict(color='cyan', dash='dash')))
    fig.update_layout(title='Stock Price Forecast vs. Actual', xaxis_title='Date', yaxis_title='Stock Price ($)', legend=dict(x=0.01, y=0.99))
    return fig

# =================================================================================
# ✅ SIDEBAR AND MAIN PAGE
# =================================================================================
st.title("FinBot 360")
st.markdown("---")

st.header("Financial Tools")
# (Removed expanders to simplify the layout and prevent UI conflicts)

st.subheader("🔴 Live Market Dashboard")
ticker_symbol = st.text_input("Enter a Stock Ticker:", "IBM").upper()
# (Live dashboard logic...)

st.subheader("😊 Financial Sentiment Analysis")
user_text = st.text_area("Enter text to analyze:", "Apple's stock soared...", height=100)
if st.button("Analyze Sentiment"):
    # (Sentiment analysis logic...)
    st.write("Sentiment analysis would appear here.")

st.subheader("📁 Portfolio Performance Analysis")
uploaded_file = st.file_uploader("Upload portfolio CSV/XLSX", type=['csv', 'xlsx'])
# (Portfolio analysis logic...)

# --- STOCK FORECASTING MOVED TO A SEPARATE SECTION FOR STABILITY ---
st.markdown("---")
st.header("📊 Stock Forecasting")
ticker_main = st.text_input("Enter Ticker (e.g., AAPL):", "AAPL", key="main_ticker").upper()
if st.button("Generate Forecast"):
    data_main = fetch_stock_data(ticker_main, "2020-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
    if not data_main.empty:
        train, valid = forecast_stock(data_main)
        if train is not None:
            fig = plot_forecast(train, valid)
            st.plotly_chart(fig, use_container_width=True)