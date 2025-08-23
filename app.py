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

# --- Imports for Sidebar Tools ---
from streamlit_autorefresh import st_autorefresh
from alpha_vantage.timeseries import TimeSeries

# ✅ Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
except Exception:
    st.error("⚠️ Gemini API Key not found. Please add it to your Streamlit secrets.", icon="🚨")
    GEMINI_AVAILABLE = False

# =================================================================================
# HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    if not GEMINI_AVAILABLE:
        return "Chatbot unavailable (Gemini API key missing)."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API failed: {e}")
        return f"API Error: {e}"

@st.cache_resource
def load_sentiment_model():
    # ✅ Force PyTorch backend (works on Streamlit Cloud)
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

@st.cache_data(ttl=1800)
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_for_forecasting(data: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data["Close"].values.reshape(-1, 1)), scaler

def analyze_portfolio(df: pd.DataFrame):
    daily_returns = df["Close"].pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility != 0 else 0
    return daily_returns, cumulative_returns, volatility, sharpe_ratio

def plot_portfolio_performance(df: pd.DataFrame, cumulative_returns: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Portfolio Price"))
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode="lines", 
                             name="Cumulative Returns", yaxis="y2"))
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2=dict(title="Cumulative Returns (%)", overlaying="y", side="right"),
        template="plotly_white"
    )
    return fig

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def forecast_stock(data: pd.DataFrame):
    scaled_data, scaler = preprocess_for_forecasting(data)
    training_len = int(len(scaled_data) * 0.8)
    
    x_train, y_train = [], []
    for i in range(60, training_len):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner("Training LSTM model..."):
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    test_data = scaled_data[training_len-60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = scaler.inverse_transform(model.predict(x_test))
    train = data[:training_len]
    valid = data[training_len:].copy()
    valid["Predictions"] = predictions
    return train, valid

def plot_forecast(train, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train["Close"], name="Training Data"))
    fig.add_trace(go.Scatter(x=valid.index, y=valid["Close"], name="Actual"))
    fig.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"], name="Predicted"))
    fig.update_layout(title="Stock Price Forecast", xaxis_title="Date", yaxis_title="Price ($)")
    return fig

# =================================================================================
# SIDEBAR
# =================================================================================
with st.sidebar:
    st.title("📈 FinBot 360")

    st.subheader("API Status")
    if GEMINI_AVAILABLE:
        st.success("Gemini API Connected")
    else:
        st.error("Gemini API Missing")

    try:
        AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
        st.success("Alpha Vantage Connected")
    except Exception:
        st.warning("Alpha Vantage Key Missing")

    st.markdown("---")

    # 🔴 Live Market Dashboard
    with st.expander("🔴 Live Market Dashboard", expanded=True):
        st_autorefresh(interval=900 * 1000, key="refresh")  # every 15 minutes
        ticker_symbol = st.text_input("Enter Stock Ticker:", "AAPL").upper()

        price = None
        try:
            ts = TimeSeries(key=AV_API_KEY, output_format="pandas")
            quote, _ = ts.get_quote_endpoint(symbol=ticker_symbol)
            price = float(quote["05. price"][0])
            st.metric(f"Live Price {ticker_symbol}", f"${price:.2f}")
        except Exception:
            st.warning("Alpha Vantage limit reached. Using Yahoo Finance...")
            data = yf.Ticker(ticker_symbol).history(period="1d")
            if not data.empty:
                price = data["Close"].iloc[-1]
                st.metric(f"Live Price {ticker_symbol}", f"${price:.2f}")

    # 😊 Financial Sentiment
    with st.expander("😊 Financial Sentiment Analysis"):
        user_text = st.text_area("Enter text:", "Apple's stock soared after strong earnings.")
        if st.button("Analyze Sentiment"):
            result = analyze_sentiment(user_text)
            st.write(f"**Sentiment:** {result['label']} (Score: {result['score']:.2f})")

    # 📊 Stock Forecasting
    with st.expander("📊 Stock Forecasting"):
        ticker = st.text_input("Forecast Ticker:", "AAPL").upper()
        if st.button("Generate Forecast"):
            data = fetch_stock_data(ticker, "2020-01-01", pd.to_datetime("today").strftime("%Y-%m-%d"))
            if not data.empty:
                train, valid = forecast_stock(data)
                st.plotly_chart(plot_forecast(train, valid), use_container_width=True)

# =================================================================================
# MAIN PAGE
# =================================================================================
st.title("Natural Language Financial Q&A")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

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

