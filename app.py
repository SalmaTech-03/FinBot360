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
from alpha_vintage.fundamentaldata import FundamentalData
from alpha_vintage.timeseries import TimeSeries

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
try:
    AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
    AV_AVAILABLE = True
except (KeyError, FileNotFoundError):
    AV_AVAILABLE = False


# =================================================================================
# ALL HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    if not GEMINI_AVAILABLE: return "Chatbot is unavailable because the Gemini API key is not configured."
    # ... (rest of function is correct) ...
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}") 
        return f"An error occurred. **Specific API Error:** {e}"

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

# --- REWRITTEN `fetch_stock_data` TO USE RELIABLE ALPHA VANTAGE API ---
@st.cache_data
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    if not AV_AVAILABLE:
        st.error("Alpha Vantage API Key not found. Cannot fetch historical data.", icon="🚨")
        return pd.DataFrame()
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        # Get daily adjusted data for the last ~5 years
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        # The API returns data with numbered column names, let's rename them
        data.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
            '4. close': 'Close', '5. adjusted close': 'Adj Close', '6. volume': 'Volume'
        }, inplace=True)
        # Reverse the dataframe so that dates are in chronological order for plotting
        data = data.iloc[::-1]
        data.index = pd.to_datetime(data.index) # Ensure index is datetime
        return data
    except Exception as e:
        st.error(f"Could not fetch data for '{ticker}' from Alpha Vantage. Error: {e}")
        return pd.DataFrame()

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), Dropout(0.2),
        LSTM(50, return_sequences=False), Dropout(0.2),
        Dense(25), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- UPDATED `forecast_stock` to ensure perfect data alignment ---
def forecast_stock(data: pd.DataFrame):
    data_close = data[['Close']].copy()
    data_close.dropna(inplace=True)
    if len(data_close) < 80:
        st.error("Not enough valid data points to forecast (need at least 80).")
        return None, None
    dataset = data_close.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    training_data_len = int(np.ceil(len(dataset) * 0.8))
    train_data = scaled_data[0:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    if not x_train:
        st.error("Training data is too short for the 60-day lookback window.")
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
        st.warning("Not enough test data for prediction.")
        return data_close[:training_data_len], None

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train_df = data_close.iloc[:training_data_len]
    valid_df = data_close.iloc[training_data_len:].copy()
    
    # Handle any length mismatch between predictions and validation dataframe
    pred_len = len(predictions)
    valid_len = len(valid_df)
    
    if pred_len != valid_len:
        # Align predictions to the end of the validation dataframe
        valid_df['Predictions'] = np.nan
        valid_df.iloc[-pred_len:, valid_df.columns.get_loc('Predictions')] = predictions.flatten()
    else:
        valid_df['Predictions'] = predictions.flatten()
        
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
    st.title("📈 FinBot 360")
    st.markdown("---")
    # ... (Rest of sidebar remains the same) ...

# ... (Main Page remains the same, except for the forecasting section call)
# (For brevity, only showing the updated part of the sidebar)
with st.sidebar:
    st.title("📈 FinBot 360")
    # ... (API status etc.)

    st.header("Financial Tools")

    # ... (Live Dashboard, Sentiment, Portfolio Analysis are unchanged) ...
    with st.expander("📊 Stock Forecasting"):
        ticker = st.text_input("Enter Ticker (e.g., AAPL):", "AAPL").upper()
        if st.button("Generate Forecast"):
            # This now uses the new, reliable Alpha Vantage function
            data = fetch_stock_data(ticker)
            if not data.empty:
                train, valid = forecast_stock(data)
                if train is not None:
                    fig = plot_forecast(train, valid)
                    st.plotly_chart(fig, use_container_width=True)
