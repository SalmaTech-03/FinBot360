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

from streamlit_autorefresh import st_autorefresh
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Advanced Financial Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def extract_gemini_text(response) -> str:
    """Robustly extract text from a Gemini response (works even when .text is empty)."""
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        # Fallback: walk candidates/parts
        if getattr(response, "candidates", None):
            parts = getattr(response.candidates[0].content, "parts", [])
            texts = []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    texts.append(t)
            return "\n".join(texts) if texts else "No text returned."
        return "No response generated."
    except Exception as e:
        return f"Failed to parse model output: {e}"

def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    """Use a fast, reliable Gemini model and handle empty .text cases."""
    if not GEMINI_AVAILABLE:
        return "Chatbot is unavailable because the Gemini API key is not configured."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return extract_gemini_text(response)
    except Exception as e:
        logging.exception("Gemini API call failed")
        return f"An error occurred. **Specific API Error:** {e}"

@st.cache_resource
def load_sentiment_model():
    # FinBERT is finance-tuned and small enough for Streamlit Cloud
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Explicitly set auto_adjust to avoid yfinance FutureWarning drift
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    df = df.dropna()
    return df

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def forecast_stock(data: pd.DataFrame):
    # 1) Prepare close prices
    data_close = data[["Close"]].copy()
    dataset = data_close.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    if len(dataset) < 80:
        st.error("Not enough historical data to train (need at least ~80 rows).")
        return None, None

    # 2) Scale and window
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)

    train_data = scaled[:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # 3) Train
    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner("Training LSTM model..."):
        model.fit(x_train, y_train, batch_size=32, epochs=8, verbose=0)

    # 4) Test window
    test_data = scaled[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test).reshape((-1, 60, 1))

    # 5) Predict & invert scale
    preds = model.predict(x_test, verbose=0)
    preds = scaler.inverse_transform(preds).flatten()

    # Build aligned frames
    train_df = data_close.iloc[:training_data_len].copy()
    valid_index = data_close.index[training_data_len:training_data_len + len(preds)]
    valid_df = pd.DataFrame(
        {"Close": data_close["Close"].iloc[training_data_len:training_data_len + len(preds)].values,
         "Predictions": preds},
        index=valid_index,
    )
    return train_df, valid_df

def plot_forecast(train: pd.DataFrame, valid: pd.DataFrame):
    fig = go.Figure()
    if train is not None and not train.empty:
        fig.add_trace(go.Scatter(x=train.index, y=train["Close"], mode="lines", name="Historical Prices"))
    if valid is not None and not valid.empty:
        fig.add_trace(go.Scatter(x=valid.index, y=valid["Close"], mode="lines", name="Actual Prices (Validation)"))
        if "Predictions" in valid.columns:
            fig.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"], mode="lines", name="Predicted Prices", line=dict(dash="dash")))
    fig.update_layout(
        title="Stock Price Forecast vs. Actual",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📈 FinBot 360")
    st.markdown("---")

    with st.expander("API Status", expanded=False):
        st.success("Gemini API: Connected" if GEMINI_AVAILABLE else "Gemini API: Disconnected")
        try:
            _ = st.secrets["ALPHA_VANTAGE_API_KEY"]
            st.success("Alpha Vantage: Connected")
        except (KeyError, FileNotFoundError):
            st.warning("Alpha Vantage: Not Connected")

    # ---------- Live Market Dashboard ----------
    with st.expander("🔴 Live Market Dashboard", expanded=True):
        ticker_live = st.text_input("Enter Ticker:", "IBM", key="live_ticker").upper()
        colA, colB = st.columns([1, 1])
        with colA:
            period = st.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
        with colB:
            interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
        if st.button("Get Live Data", key="btn_live"):
            live = yf.download(ticker_live, period=period, interval=interval, auto_adjust=True, progress=False)
            if live.empty:
                st.error("No data returned. Check ticker or interval.")
            else:
                st.line_chart(live["Close"])
                st.caption(f"{ticker_live} close price ({period}/{interval})")
                last = float(live["Close"].iloc[-1])
                st.metric("Last Close", f"${last:,.2f}")

    # ---------- Sentiment ----------
    with st.expander("😊 Financial Sentiment Analysis", expanded=True):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared on strong earnings.")
        if st.button("Analyze Sentiment", key="btn_sent"):
            with st.spinner("Analyzing..."):
                try:
                    res = analyze_sentiment(user_text)
                    st.write(res)
                except Exception as e:
                    st.error(f"Sentiment pipeline error: {e}")

    # ---------- Portfolio Upload (placeholder) ----------
    with st.expander("📁 Portfolio Performance Analysis"):
        uploaded_file = st.file_uploader("Upload Portfolio", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                dfp = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.success(f"Loaded {len(dfp)} rows.")
                st.dataframe(dfp.head())
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    # ---------- Forecast ----------
    with st.expander("📊 Stock Forecasting", expanded=True):
        ticker = st.text_input("Enter Ticker:", "AAPL", key="forecast_ticker").upper()
        if st.button("Generate Forecast", key="forecast_button"):
            data = fetch_stock_data(ticker, "2000-01-01", pd.to_datetime("today").strftime("%Y-%m-%d"))
            if data.empty:
                st.session_state.forecast_fig = None
                st.error(f"No data for {ticker}.")
            else:
                train, valid = forecast_stock(data)
                st.session_state.forecast_fig = plot_forecast(train, valid) if train is not None else None

        if st.session_state.get("forecast_fig") is not None:
            st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)

# --------------------------------------------------
# Chat area
# --------------------------------------------------
st.title("Natural Language Financial Q&A")
st.markdown("Ask the AI assistant about financial topics, market trends, or definitions.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with your financial questions today?"}
    ]

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
