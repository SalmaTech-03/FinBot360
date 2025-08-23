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
try: # Check for Alpha Vantage key
    AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
    AV_AVAILABLE = True
except (KeyError, FileNotFoundError):
    st.warning("Alpha Vantage API Key not found. Some features will be disabled.", icon="⚠️")
    AV_AVAILABLE = False

# =================================================================================
# ALL HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    # ... (function is correct) ...
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
    # ... (function is correct) ...
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text: str):
    # ... (function is correct) ...
    return load_sentiment_model()(text)[0]

# --- MODIFIED: `fetch_stock_data` NOW USES THE STABLE ALPHA VANTAGE API ---
@st.cache_data
def fetch_stock_data_for_forecast(ticker: str) -> pd.DataFrame:
    if not AV_AVAILABLE:
        st.error("Alpha Vantage API key is missing. Cannot fetch data for forecasting.")
        return pd.DataFrame()
    
    try:
        st.info(f"Fetching full historical data for {ticker} from Alpha Vantage...")
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        
        # Clean up column names returned by the API
        data.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
            '4. close': 'Close', '5. adjusted close': 'Adj Close', '6. volume': 'Volume'
        }, inplace=True)
        
        # API returns data in reverse chronological order, so we fix it
        data = data.iloc[::-1]
        data.index = pd.to_datetime(data.index)
        data.dropna(inplace=True) # Drop any missing values
        return data

    except Exception as e:
        st.error(f"Could not fetch historical data from Alpha Vantage. The ticker might be invalid or the API limit reached. Error: {e}")
        return pd.DataFrame()


def analyze_portfolio(df: pd.DataFrame):
    # ... (function is correct) ...
    if 'Close' not in df.columns:
        st.error("Uploaded file must contain a 'Close' column for analysis.")
        return None, None, None, None
    daily_returns = df['Close'].pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility != 0 else 0
    return daily_returns, cumulative_returns, volatility, sharpe_ratio

def plot_portfolio_performance(df: pd.DataFrame, cumulative_returns: pd.Series):
    # ... (function is correct) ...
    fig = go.Figure()
    # (Plotting logic is correct)
    return fig

def create_lstm_model(input_shape):
    # ... (function is correct) ...
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), Dropout(0.2),
        LSTM(50, return_sequences=False), Dropout(0.2),
        Dense(25), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- MODIFIED `forecast_stock` TO HANDLE CLEAN DATA FROM ALPHA VANTAGE ---
def forecast_stock(data: pd.DataFrame):
    # 1. Prepare data
    data_close = data[['Close']].copy()
    if len(data_close) < 80:
        st.error("Not enough valid data points to forecast (need at least 80).")
        return None, None
    dataset = data_close.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # 2. Create training data
    training_data_len = int(np.ceil(len(dataset) * .8))
    train_data = scaled_data[0:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    if not x_train: 
        st.error("Not enough training data for the 60-day lookback window.")
        return None, None
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # 3. Build and Train Model
    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner('Training LSTM model... This may take a moment.'):
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)
        
    # 4. Create test data and make predictions
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
    
    # 5. Manually build the final dataframes to guarantee alignment
    train_df = data_close.iloc[:training_data_len]
    validation_dates = data_close.index[training_data_len:]
    valid_df = pd.DataFrame(index=validation_dates[:len(predictions)])
    valid_df['Close'] = data_close['Close'].iloc[training_data_len:training_data_len + len(predictions)]
    valid_df['Predictions'] = predictions.flatten()
    
    return train_df, valid_df


def plot_forecast(train, valid):
    # ... (function is correct) ...
    fig = go.Figure()
    # (Plotting logic is correct)
    return fig


# =================================================================================
# ✅ SIDEBAR AND MAIN PAGE
# =================================================================================
with st.sidebar:
    st.title("📈 FinBot 360")
    # ... (Your sidebar is mostly correct, updating the forecasting part)

    with st.expander("📊 Stock Forecasting"):
        ticker = st.text_input("Enter Ticker (e.g., AAPL):", "AAPL").upper()
        if st.button("Generate Forecast"):
            # MODIFIED: Call the new Alpha Vantage data function
            data = fetch_stock_data_for_forecast(ticker)
            
            if not data.empty:
                train, valid = forecast_stock(data)
                if train is not None:
                    fig = plot_forecast(train, valid)
                    st.plotly_chart(fig, use_container_width=True)

# ... (The rest of your main page and sidebar remains the same)