import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import google.genergenerativeai as genai
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
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="tf")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

@st.cache_data
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date)
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

# --- FINAL, ROBUST, AND CORRECTED FORECAST FUNCTION ---
def forecast_stock(data: pd.DataFrame):
    # 1. Prepare data
    data_close = data[['Close']]
    dataset = data_close.values
    training_data_len = int(np.ceil(len(dataset) * .8))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # 2. Create training data
    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    if not x_train: 
        st.error("Not enough training data to create a forecast.")
        return None, None
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # 3. Build and Train Model
    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner('Training LSTM model... This may take a moment.'):
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)
        
    # 4. Create test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    if not x_test:
        st.warning("Not enough test data to create a prediction.")
        return data_close[:training_data_len], None

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # 5. Get Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # 6. --- THE DEFINITIVE FIX: Manually construct the final dataframes ---
    train_df = data_close.iloc[:training_data_len]
    
    # Get the dates and actual prices for the validation period
    validation_dates = data_close.index[training_data_len:]
    actual_prices = data_close.values[training_data_len:]

    # Handle any potential length mismatch due to the 60-day window
    # This ensures the predictions align perfectly with the dates
    if len(predictions) < len(validation_dates):
        validation_dates = validation_dates[-len(predictions):]
        actual_prices = actual_prices[-len(predictions):]

    # Create the validation dataframe from scratch to guarantee perfect alignment
    valid_df = pd.DataFrame({
        'Close': actual_prices.flatten(),
        'Predictions': predictions.flatten()
    }, index=validation_dates)
    
    return train_df, valid_df

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
with st.sidebar:
    st.image("path/to/your/logo.png", width=50) # Optional: Add a logo
    st.title("FinBot 360")
    st.markdown("Your AI-Powered Financial Co-pilot")
    st.markdown("---")
    
    with st.expander("API Status", expanded=True):
        if GEMINI_AVAILABLE: st.success("Gemini API: Connected")
        else: st.error("Gemini API: Disconnected")
        try:
            st.secrets["ALPHA_VANTAGE_API_KEY"]; st.success("Alpha Vantage: Connected")
        except (KeyError, FileNotFoundError): st.warning("Alpha Vantage: Not Connected")
    st.markdown("---")

st.header("Financial Tools")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("🔴 Live Market Dashboard")
        ticker_live = st.text_input("Enter a Stock Ticker:", "IBM", key="live_ticker").upper()
        if st.button("Get Live Data", key="live_button"):
            # Your live data logic here...
            st.success(f"Data for {ticker_live} would be shown here.")

    with st.container(border=True):
        st.subheader("😊 Financial Sentiment Analysis")
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared...", key="sentiment_text")
        if st.button("Analyze Sentiment", key="sentiment_button"):
            with st.spinner("Analyzing..."):
                result = analyze_sentiment(user_text)
                # Display results...
                st.write(result)
    
    with st.container(border=True):
        st.subheader("📁 Portfolio Performance Analysis")
        uploaded_file = st.file_uploader("Upload portfolio CSV/XLSX", type=['csv', 'xlsx'], key="portfolio_uploader")
        if uploaded_file:
            # Your portfolio analysis logic here...
            st.success("Portfolio would be analyzed here.")

with col2:
    with st.container(border=True):
        st.header("📊 Stock Forecasting")
        ticker_main = st.text_input("Enter Ticker (e.g., AAPL):", "AAPL", key="main_ticker").upper()
        if st.button("Generate Forecast", key="forecast_button"):
            data_main = fetch_stock_data(ticker_main, "2020-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
            if not data_main.empty:
                train, valid = forecast_stock(data_main)
                if train is not None:
                    fig = plot_forecast(train, valid)
                    st.plotly_chart(fig, use_container_width=True)

# Chatbot Interface can go here or in a separate tab if preferred
# ...