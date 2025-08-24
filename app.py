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

# --- Imports for the Sidebar Tools ---
from streamlit_autorefresh import st_autorefresh
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

# --------------------------------------------------
# Logging & Page Config
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
st.set_page_config(page_title="Advanced Financial Assistant", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------
# API keys
# --------------------------------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except (KeyError, FileNotFoundError):
    st.error("⚠️ Gemini API Key not found. Add it to your Streamlit secrets.", icon="🚨")
    GEMINI_AVAILABLE = False
try:
    AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
    AV_AVAILABLE = True
except (KeyError, FileNotFoundError):
    st.warning("Alpha Vantage API Key not found. Live Dashboard and Forecasting will be limited.", icon="⚠️")
    AV_AVAILABLE = False


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    # ... (function is correct and unchanged) ...
    if not GEMINI_AVAILABLE: return "Chatbot is unavailable..."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

# --- NEW ROBUST FUNCTION FOR HISTORICAL DATA USING ALPHA VANTAGE ---
@st.cache_data(show_spinner="Fetching historical data from Alpha Vantage...")
def fetch_historical_data_for_forecast(ticker: str) -> pd.DataFrame:
    if not AV_AVAILABLE:
        st.error("Alpha Vantage API key is missing. Cannot fetch forecasting data.")
        return pd.DataFrame()
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        # Clean the column names from the API response
        data.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
            '4. close': 'Close', '5. adjusted close': 'Adj Close', '6. volume': 'Volume'
        }, inplace=True)
        # Reverse the data to be in chronological order
        data = data.iloc[::-1]
        data.index = pd.to_datetime(data.index)
        return data.dropna()
    except Exception as e:
        st.error(f"Failed to fetch data from Alpha Vantage. Ticker may be invalid or API limit reached. Error: {e}")
        return pd.DataFrame()

def create_lstm_model(input_shape):
    # ... (function is correct and unchanged) ...
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), Dropout(0.2),
        LSTM(50, return_sequences=False), Dropout(0.2),
        Dense(25), Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# --- YOUR ORIGINAL FORECASTING LOGIC, WHICH IS CORRECT ---
def forecast_stock(data: pd.DataFrame):
    data_close = data[["Close"]].copy()
    dataset = data_close.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    if len(dataset) < 80:
        st.error("Not enough historical data to train (need at least ~80 rows).")
        return None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)

    train_data = scaled[:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner("Training LSTM model..."):
        model.fit(x_train, y_train, batch_size=32, epochs=8, verbose=0)

    test_data = scaled[training_data_len - 60:, :]
    x_test = [test_data[i - 60:i, 0] for i in range(60, len(test_data))]
    x_test = np.array(x_test).reshape((-1, 60, 1))

    preds = model.predict(x_test, verbose=0)
    preds = scaler.inverse_transform(preds).flatten()

    train_df = data_close.iloc[:training_data_len].copy()
    valid_index = data_close.index[training_data_len:training_data_len + len(preds)]
    valid_df = pd.DataFrame({
        "Close": data_close["Close"].iloc[training_data_len:training_data_len + len(preds)].values,
        "Predictions": preds},
        index=valid_index,
    )
    return train_df, valid_df

def plot_forecast(train: pd.DataFrame, valid: pd.DataFrame):
    # ... (function is correct and unchanged) ...
    fig = go.Figure()
    if train is not None and not train.empty:
        fig.add_trace(go.Scatter(x=train.index, y=train["Close"], mode="lines", name="Historical Prices"))
    if valid is not None and not valid.empty:
        fig.add_trace(go.Scatter(x=valid.index, y=valid["Close"], mode="lines", name="Actual Prices (Validation)"))
        if "Predictions" in valid.columns:
            fig.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"], mode="lines", name="Predicted Prices", line=dict(dash="dash")))
    fig.update_layout(title="Stock Price Forecast vs. Actual", xaxis_title="Date", yaxis_title="Stock Price ($)", legend=dict(x=0.01, y=0.99))
    return fig

# =================================================================================
# SIDEBAR
# =================================================================================
with st.sidebar:
    st.title("📈 FinBot 360")
    st.markdown("---")

    with st.expander("API Status"):
        st.success("Gemini API: Connected" if GEMINI_AVAILABLE else "Gemini API: Disconnected")
        st.success("Alpha Vantage: Connected" if AV_AVAILABLE else "Alpha Vantage: Disconnected")
        
    with st.expander("🔴 Live Market Dashboard"):
        # Live Market Dashboard can still use yfinance as it's less critical
        ticker_live = st.text_input("Enter Ticker:", "IBM", key="live_ticker").upper()
        # ... (Your Live Dashboard code here)

    with st.expander("😊 Financial Sentiment Analysis"):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared on strong earnings.", key="sentiment_text")
        if st.button("Analyze Sentiment", key="btn_sent"):
            st.write(analyze_sentiment(user_text))
            
    # --- PORTFOLIO ANALYSIS SECTION RESTORED ---
    with st.expander("📁 Portfolio Performance Analysis"):
        uploaded_file = st.file_uploader("Upload portfolio CSV/XLSX", type=['csv', 'xlsx'], key="portfolio_uploader")
        if uploaded_file:
            # Placeholder for your portfolio analysis logic
            st.success("Portfolio analysis results would be displayed here.")

    with st.expander("📊 Stock Forecasting", expanded=True):
        ticker = st.text_input("Enter Ticker:", "AAPL", key="forecast_ticker").upper()
        if st.button("Generate Forecast", key="forecast_button"):
            # UPDATED to call the new, stable function
            data = fetch_historical_data_for_forecast(ticker)
            
            if not data.empty:
                train, valid = forecast_stock(data)
                if train is not None:
                    fig_forecast = plot_forecast(train, valid)
                    st.plotly_chart(fig_forecast, use_container_width=True)


# =================================================================================
# CHATBOT MAIN PAGE
# =================================================================================
st.title("Natural Language Financial Q&A")
st.markdown("Ask the AI assistant about financial topics, market trends, or definitions.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})