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
from datetime import datetime, timedelta

# --- Imports for the Sidebar Tools ---
from streamlit_autorefresh import st_autorefresh
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

# ✅ Integrated Logging & Debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FinBot 360",
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
    st.error("⚠️ Gemini API Key not found. Please add it to your secrets.", icon="🚨")
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
        return f"An error occurred: {e}"

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

@st.cache_data
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.", icon="❌")
        return pd.DataFrame()
    return data

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_stock(data: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    training_data_len = int(np.ceil(len(scaled_data) * .8))
    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner('Training forecasting model... This may take a moment.'):
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)
    
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = scaler.inverse_transform(model.predict(x_test))
    
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions
    return train, valid

def plot_forecast(train, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual Prices (Validation)'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted Prices'))
    fig.update_layout(title='Stock Price Forecast vs. Actual', xaxis_title='Date', yaxis_title='Stock Price ($)')
    return fig

# =================================================================================
# ✅ SIDEBAR - TOOLS & CONTROLS
# =================================================================================
with st.sidebar:
    st.title("📈 FinBot 360")
    st.markdown("---")

    with st.expander("API Status", expanded=True):
        if GEMINI_AVAILABLE: st.success("Gemini API: Connected")
        else: st.error("Gemini API: Disconnected")
        try:
            st.secrets["ALPHA_VANTAGE_API_KEY"]; st.success("Alpha Vantage: Connected")
        except (KeyError, FileNotFoundError): st.warning("Alpha Vantage: Not Found")
        st.info("Yahoo Finance: Connected")

    st.markdown("---")
    st.header("Financial Tools")

    # --- Tool 1: Live Market Dashboard ---
    with st.expander("🔴 Live Market Dashboard", expanded=True):
        st_autorefresh(interval=60 * 1000, key="datarefresh")
        ticker_symbol = st.text_input("Enter Ticker:", "IBM").upper()
        if ticker_symbol:
            try:
                AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
                ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
                quote_data, _ = ts.get_quote_endpoint(symbol=ticker_symbol)
                price = float(quote_data['05. price'][0])
                change = float(quote_data['09. change'][0])
                change_percent_str = quote_data['10. change percent'][0]
                change_percent = float(change_percent_str.replace('%',''))
                st.metric("Live Price (Alpha Vantage)", f"${price:.2f}", f"{change:.2f} ({change_percent:.2f}%)")
            except Exception:
                st.error("Could not fetch live price. API limit may be reached.")

            st.markdown(f"**{ticker_symbol} - Last Month's Price**")
            try:
                end_date = datetime.today()
                start_date = end_date - timedelta(days=30)
                hist_data = fetch_stock_data(ticker_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if not hist_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], mode='lines', name='Close Price'))
                    fig.update_layout(height=200, margin=dict(l=10, r=10, t=20, b=10), showlegend=False, xaxis_title="", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not fetch historical data for the chart.")
            except Exception:
                st.error("An error occurred while fetching chart data.")

    # --- Tool 2: Financial Sentiment Analysis ---
    with st.expander("😊 Financial Sentiment Analysis"):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared after their strong quarterly earnings report.", height=100)
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing..."):
                result = analyze_sentiment(user_text)
                sentiment = result['label'].upper(); score = result['score']
                if sentiment == 'POSITIVE': st.success(f"Sentiment: {sentiment} (Score: {score:.2f})")
                elif sentiment == 'NEGATIVE': st.error(f"Sentiment: {sentiment} (Score: {score:.2f})")
                else: st.info(f"Sentiment: {sentiment} (Score: {score:.2f})")

    # --- Tool 3: Portfolio Performance Analysis ---
    with st.expander("📁 Portfolio Performance Analysis"):
        st.write("Upload a CSV/XLSX file with 'Date' and 'Close' columns.")
        # Placeholder for full feature
        
# =================================================================================
# ✅ MAIN PAGE - CHATBOT & FORECASTING
# =================================================================================

st.title("Natural Language Financial Q&A")

# --- Chatbot Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help with your financial questions today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")

# --- Stock Forecasting Module (on main page) ---
st.subheader("📈 Stock Forecasting")
forecast_ticker = st.text_input("Enter Stock Symbol for Forecasting:", "NVDA").upper()
if st.button("Generate Forecast"):
    if forecast_ticker:
        data = fetch_stock_data(forecast_ticker, "2020-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
        if not data.empty:
            train, valid = forecast_stock(data)
            # --- THIS IS THE CORRECTED LINE ---
            if train is not None:
                fig = plot_forecast(train, valid)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter a stock ticker to generate a forecast.")
