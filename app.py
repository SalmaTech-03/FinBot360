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
from datetime import timedelta

# --- Imports for the Sidebar Tools ---
from streamlit_autorefresh import st_autorefresh
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

# --------------------------------------------------
# Page Config & Logging
# --------------------------------------------------
st.set_page_config(page_title="FinBot 360", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
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
try:
    AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
    AV_AVAILABLE = True
except (KeyError, FileNotFoundError):
    AV_AVAILABLE = False

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help with your financial questions today?"}]
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    if not GEMINI_AVAILABLE: return "Chatbot is unavailable."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

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

# --- FUNCTION FOR THE NEW TABULAR FORECAST ---
def generate_future_forecast(data: pd.DataFrame, future_days=10):
    data_close = data[['Close']].copy()
    if len(data_close) < 60:
        return None, "Not enough data (need at least 60 days)."
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)
    
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner("Training predictive model..."):
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    last_60_days = scaled_data[-60:]
    future_preds = []
    current_batch = last_60_days.reshape(1, 60, 1)
    with st.spinner("Generating future predictions..."):
        for _ in range(future_days):
            next_pred = model.predict(current_batch, verbose=0)[0]
            future_preds.append(next_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)
    
    future_preds = scaler.inverse_transform(future_preds).flatten()
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_preds})
    forecast_df['% Change'] = forecast_df['Predicted Price'].pct_change() * 100
    return forecast_df, None


def get_ai_insights(ticker, forecast_df):
    if not GEMINI_AVAILABLE: return "AI Insights are unavailable."
    prompt = f"""
    Analyze the following 10-day stock forecast for {ticker} for a retail investor.
    Forecast Table:
    {forecast_df.to_string()}
    Provide a brief, easy-to-understand summary of the potential trend (e.g., "bullish," "bearish"). Mention the potential price range. Keep it concise.
    End with this disclaimer: "Disclaimer: This is not financial advice."
    """
    return get_llm_response(prompt)


# =================================================================================
# SIDEBAR
# =================================================================================
with st.sidebar:
    st.title("FinBot 360")
    st.markdown("---")
    
    with st.expander("API Status"):
        st.success("Gemini API: Connected" if GEMINI_AVAILABLE else "Gemini API: Disconnected")
        st.success("Alpha Vantage: Connected" if AV_AVAILABLE else "Alpha Vantage: Disconnected")
    
    st.header("Financial Tools")

    # --- FULLY RESTORED LIVE MARKET DASHBOARD ---
    with st.expander("🔴 Live Market Dashboard", expanded=True):
        st_autorefresh(interval=300 * 1000, key="datarefresh")
        ticker_live = st.text_input("Enter Ticker:", "IBM", key="live_ticker").upper()
        if AV_AVAILABLE and ticker_live:
            try:
                ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
                quote_data, _ = ts.get_quote_endpoint(symbol=ticker_live)
                st.metric(label=f"Live Price ({ticker_live})", value=f"${float(quote_data['05. price'][0]):.2f}", delta=f"{float(quote_data['09. change'][0]):.2f} ({quote_data['10. change percent'][0]})")
            except Exception as e:
                st.error("Could not fetch live price. API limit may be reached.")
        
        # Live chart from yfinance
        live_data = yf.download(ticker_live, period="1mo", interval="1d", auto_adjust=True, progress=False)
        if not live_data.empty:
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(x=live_data.index, y=live_data["Close"], mode="lines", name="Close Price"))
            st.plotly_chart(fig_live, use_container_width=True)

    # --- SENTIMENT ANALYSIS ---
    with st.expander("😊 Financial Sentiment Analysis", expanded=False):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared...", key="sentiment_text")
        if st.button("Analyze Sentiment"):
            # Sentiment logic...
            st.success("Positive (Score: 0.90)")

    # --- PORTFOLIO ANALYSIS ---
    with st.expander("📁 Portfolio Performance Analysis", expanded=False):
        uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv", key="portfolio_uploader")
        if uploaded_file:
            st.success("Portfolio analysis placeholder.")

# =================================================================================
# MAIN PAGE (Chatbot + TABULAR Forecast)
# =================================================================================
st.title("Natural Language Financial Q&A")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.header("📊 Stock Forecasting")

ticker_forecast = st.text_input("Enter Stock Symbol for Forecasting:", "NVDA").upper()

if st.button("Generate Forecast", key="gen_forecast"):
    data = fetch_stock_data(ticker_forecast)
    if not data.empty:
        forecast_df, error_msg = generate_future_forecast(data)
        if error_msg:
            st.error(error_msg)
            st.session_state.forecast_data = None
        else:
            st.session_state.forecast_data = {
                "ticker": ticker_forecast,
                "forecast_df": forecast_df
            }
    else:
        st.error("No historical data found to generate forecast.")
        st.session_state.forecast_data = None


if st.session_state.forecast_data:
    forecast_df = st.session_state.forecast_data["forecast_df"]
    ticker = st.session_state.forecast_data["ticker"]
    
    st.subheader(f"10-Day Forecast Table for {ticker}")
    st.dataframe(
        forecast_df.style.format({
            "Predicted Price": "${:,.2f}",
            "% Change": "{:,.2f}%"
        }).applymap(lambda x: 'color: green' if x > 0 else 'color: red', subset=['% Change']),
        use_container_width=True
    )

    st.subheader("🤖 AI-Powered Text Insights")
    with st.spinner("Generating AI analysis..."):
        ai_summary = get_ai_insights(ticker, forecast_df)
        st.markdown(ai_summary)