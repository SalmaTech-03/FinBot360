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
        st.warning(f"No data found for ticker '{ticker}'. Please check the symbol.")
        return pd.DataFrame()
    return data

def analyze_portfolio(df: pd.DataFrame):
    if 'Close' not in df.columns or 'Date' not in df.columns:
        st.error("Uploaded file must contain 'Date' and 'Close' columns.")
        return None, None, None, None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    daily_returns = df['Close'].pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility != 0 else 0
    return daily_returns, cumulative_returns, volatility, sharpe_ratio

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

    # --- Tool 1: Live Market Dashboard (GRAPH REMOVED) ---
    with st.expander("🔴 Live Market Dashboard", expanded=True):
        st_autorefresh(interval=60 * 1000, key="datarefresh")
        ticker_symbol = st.text_input("Enter Ticker:", "IBM").upper()
        if ticker_symbol:
            # --- Live Price with Fallback Mechanism ---
            try:
                # PRIMARY SOURCE: Alpha Vantage
                AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
                ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
                quote_data, _ = ts.get_quote_endpoint(symbol=ticker_symbol)
                price = float(quote_data['05. price'].iloc[0])
                change = float(quote_data['09. change'].iloc[0])
                change_percent = float(quote_data['10. change percent'].iloc[0].replace('%',''))
                st.metric("Live Price (Alpha Vantage)", f"${price:.2f}", f"{change:.2f} ({change_percent:.2f}%)")
            except Exception as e:
                # FALLBACK SOURCE: Yahoo Finance
                logging.warning(f"Alpha Vantage failed: {e}. Falling back to yfinance.")
                try:
                    yf_ticker = yf.Ticker(ticker_symbol)
                    info = yf_ticker.info
                    price = info.get("currentPrice", 0)
                    previous_close = info.get("previousClose", 1)
                    change = price - previous_close
                    change_percent = (change / previous_close) * 100
                    st.metric("Live Price (Yahoo Finance)", f"${price:.2f}", f"{change:.2f} ({change_percent:.2f}%)")
                except Exception as yf_e:
                    st.error("Both Alpha Vantage & Yahoo Finance failed.")
                    logging.error(f"YFinance fallback also failed: {yf_e}")

    # --- Tool 2: Financial Sentiment Analysis ---
    with st.expander("😊 Financial Sentiment Analysis", expanded=True):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared after their strong quarterly earnings report.", height=100)
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing..."):
                result = analyze_sentiment(user_text)
                sentiment = result['label'].upper(); score = result['score']
                if sentiment == 'POSITIVE': st.success(f"Sentiment: {sentiment} (Score: {score:.2f})")
                elif sentiment == 'NEGATIVE': st.error(f"Sentiment: {sentiment} (Score: {score:.2f})")
                else: st.info(f"Sentiment: {sentiment} (Score: {score:.2f})")

    # --- Tool 3: Portfolio Performance Analysis ---
    with st.expander("📁 Portfolio Performance Analysis", expanded=True):
        uploaded_file = st.file_uploader("Upload portfolio CSV/XLSX file", type=['csv', 'xlsx'], key="portfolio_uploader")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                daily_returns, cum_returns, volatility, sharpe = analyze_portfolio(df)
                if cum_returns is not None:
                    st.metric("Total Return", f"{cum_returns.iloc[-1]:.2%}")
                    st.metric("Annualized Volatility", f"{volatility:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
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

# Initialize session state for the forecast DataFrame
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'last_forecast_ticker' not in st.session_state:
    st.session_state.last_forecast_ticker = ""

forecast_ticker = st.text_input("Enter Stock Symbol for Forecasting:", st.session_state.last_forecast_ticker or "NVDA").upper()

# Clear old forecast if ticker changes
if forecast_ticker != st.session_state.last_forecast_ticker:
    st.session_state.forecast_df = None
    st.session_state.last_forecast_ticker = forecast_ticker

if st.button("Generate Forecast"):
    if forecast_ticker:
        data = fetch_stock_data(forecast_ticker, "2020-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
        if not data.empty:
            train, valid = forecast_stock(data)
            if valid is not None:
                # Store the DataFrame with results in session state
                st.session_state.forecast_df = valid[['Close', 'Predictions']]
            else:
                st.session_state.forecast_df = None
        else:
            st.session_state.forecast_df = None
    else:
        st.warning("Please enter a stock ticker to generate a forecast.")
        st.session_state.forecast_df = None

# Always display the DataFrame from session state if it exists
if st.session_state.forecast_df is not None:
    st.subheader("Forecast vs. Actual Prices")
    st.dataframe(st.session_state.forecast_df)
