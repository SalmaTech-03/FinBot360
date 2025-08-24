# app.py
import os
import logging
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# TensorFlow / Keras imports (used for LSTM forecasting)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Transformers for sentiment
from transformers import pipeline

# Gemini
import google.generativeai as genai

# Optional utilities you used earlier (kept for compatibility)
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from streamlit_autorefresh import st_autorefresh

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="FinBot 360", page_icon="💹", layout="wide")
st.markdown(
    """
    <style>
    .card {
        background: #ffffff;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
    }
    .header {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .subtle { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='header'>💹 FinBot 360 — Advanced Financial Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Live dashboard • Forecasting (table + metrics + AI insights) • Sentiment • Portfolio • Chatbot</div>", unsafe_allow_html=True)
st.write("")

# -------------------------
# Gemini setup (robust)
# -------------------------
GEMINI_AVAILABLE = False
try:
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_AVAILABLE = True
except Exception as e:
    logging.exception("Failed to configure Gemini API: %s", e)
    GEMINI_AVAILABLE = False

def extract_gemini_text(response) -> str:
    """Robustly extract text from a Gemini response object."""
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if getattr(response, "candidates", None):
            parts = getattr(response.candidates[0].content, "parts", [])
            texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
            return "\n".join([t for t in texts if t]) or "No text returned."
        return "No response generated."
    except Exception as e:
        logging.exception("Failed to parse Gemini response: %s", e)
        return f"Failed to parse model output: {e}"

def get_llm_response(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    if not GEMINI_AVAILABLE:
        return "Gemini API key not configured. Please add it to secrets."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return extract_gemini_text(response)
    except Exception as e:
        logging.exception("Gemini call failed: %s", e)
        return f"LLM error: {e}"

# -------------------------
# Cached heavy items
# -------------------------
@st.cache_resource
def cached_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df is None:
        return pd.DataFrame()
    return df.dropna()

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

def train_and_forecast(data_close: pd.Series, forecast_days: int = 5, epochs: int = 5):
    """
    Train LSTM on historical close prices and iteratively forecast `forecast_days`.
    Returns: forecast_df (Date, Predicted Price), metrics dict, residual_std (approx used for confidence)
    """
    prices = data_close.values.reshape(-1, 1)  # shape (n,1)
    if len(prices) < 80:
        raise ValueError("Not enough historical rows to train (need at least ~80).")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)  # shape (n,1)

    # prepare dataset for training (60-window)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, 60, 1)

    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    # iterative forecast using last 60 points
    last_60 = scaled[-60:].reshape(1, 60, 1)
    preds = []
    for _ in range(forecast_days):
        p = model.predict(last_60, verbose=0)
        preds.append(float(p.ravel()[0]))  # scalar in scaled space
        # slide window
        last_60 = np.append(last_60[:, 1:, :], [[p]], axis=1)

    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).flatten()  # shape (forecast_days,)

    # Build dates
    last_date = pd.to_datetime(data_close.index[-1])
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": preds_inv})

    # Compute metrics
    today_price = float(data_close.iloc[-1])
    next_pred = float(preds_inv[0])
    pct_change = ((next_pred - today_price) / today_price) * 100.0
    trend = "Uptrend" if preds_inv[-1] > preds_inv[0] else "Downtrend"
    # approximate residuals on train set for simple confidence interval
    train_preds = model.predict(X, verbose=0)
    train_preds_inv = scaler.inverse_transform(train_preds).flatten()
    train_actual_inv = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    residuals = train_actual_inv - train_preds_inv
    resid_std = float(np.std(residuals))
    # create a simple confidence interval (± 1.96 * std) around predictions (approx)
    ci_pct = (resid_std / today_price) * 100.0 * 1.96

    metrics = {
        "today_price": today_price,
        "next_pred": next_pred,
        "pct_change": pct_change,
        "trend": trend,
        "resid_std": resid_std,
        "confidence_pct": ci_pct
    }
    return forecast_df, metrics

# -------------------------
# UI: Sidebar + layout
# -------------------------
st.sidebar.title("📌 FinBot 360")
st.sidebar.write("Choose a section")
section = st.sidebar.radio("", ["Live Dashboard", "Stock Forecasting", "Sentiment Analysis", "Portfolio", "Chatbot"])

# -------------------------
# Live Dashboard (unchanged)
# -------------------------
if section == "Live Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Live Market Dashboard")
    ticker_live = st.text_input("Enter Ticker:", value="IBM", key="live_ticker").upper()
    period = st.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y", "5y", "max"], index=1)
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

    if st.button("Get Live Data"):
        live = yf.download(ticker_live, period=period, interval=interval, auto_adjust=True, progress=False)
        if live.empty:
            st.error("No data returned for that ticker/interval.")
        else:
            st.write(live.tail())
            # use Plotly candlestick for live dashboard (kept)
            fig = go = None
            try:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Candlestick(
                    x=live.index, open=live["Open"], high=live["High"], low=live["Low"], close=live["Close"]
                )])
                fig.update_layout(title=f"{ticker_live} ({period}/{interval})", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write("Chart failed to render, showing table instead.")
                st.write(live)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Stock Forecasting (no graph)
# -------------------------
elif section == "Stock Forecasting":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Stock Forecasting (Table + Metrics + AI Insight)")
    ticker = st.text_input("Ticker for forecasting:", value="AAPL", key="forecast_ticker").upper()
    forecast_days = st.slider("Forecast days", 1, 15, value=5)
    epochs = st.slider("LSTM epochs (affects speed)", 1, 10, value=3)

    if st.button("Run Forecast"):
        with st.spinner("Training model and generating forecast..."):
            try:
                end = pd.to_datetime("today")
                start = end - pd.Timedelta(days=365 * 2)  # last 2 years
                df = fetch_stock_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                if df.empty or "Close" not in df.columns:
                    st.error("No data found for that ticker.")
                else:
                    forecast_df, metrics = train_and_forecast(df["Close"], forecast_days, epochs)

                    # 1) Forecast table
                    st.subheader("📅 Forecast Table")
                    st.dataframe(forecast_df.style.format({"Predicted Price": "${:,.2f}"}))

                    # allow CSV download
                    csv = forecast_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download forecast CSV", csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")

                    # 2) Key metrics
                    st.subheader("📌 Key Metrics")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Today's Price", f"${metrics['today_price']:.2f}")
                    c2.metric(f"Next Day Predicted", f"${metrics['next_pred']:.2f}", delta=f"{metrics['pct_change']:.2f}%")
                    c3.metric("Trend", metrics['trend'])

                    # Confidence interval display
                    st.info(f"Approx. confidence interval: ±{metrics['confidence_pct']:.2f}% (based on training residuals)")

                    # 3) AI Insight (Gemini)
                    st.subheader("🤖 AI Insight")
                    prompt = (
                        f"You're a succinct financial advisor. Based on the ticker {ticker}:\n"
                        f"Last close price: {metrics['today_price']:.2f}\n"
                        f"Next-day predicted price: {metrics['next_pred']:.2f}\n"
                        f"Expected change: {metrics['pct_change']:.2f}%\n\n"
                        "Give a 2-3 sentence investment-style insight and a short risk note."
                    )
                    ai_text = get_llm_response(prompt, model_name="gemini-2.5-flash")
                    st.write(ai_text)

            except Exception as e:
                logging.exception("Forecast failed")
                st.error(f"Forecast failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Sentiment Analysis (unchanged)
# -------------------------
elif section == "Sentiment Analysis":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📰 Financial Sentiment Analysis")
    news_text = st.text_area("Paste news or article text here:")
    if st.button("Analyze Sentiment"):
        if not news_text.strip():
            st.warning("Please paste text to analyze.")
        else:
            try:
                sentiment_model = cached_sentiment_pipeline()
                res = sentiment_model(news_text)[0]
                st.write(f"**Sentiment:** {res['label']} — confidence {res['score']:.2f}")
            except Exception as e:
                logging.exception("Sentiment analysis failed")
                st.error(f"Sentiment analysis failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Portfolio (unchanged)
# -------------------------
elif section == "Portfolio":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💼 Portfolio Upload & Performance")
    uploaded = st.file_uploader("Upload CSV with columns: Ticker, Shares", type=["csv"])
    if uploaded is not None:
        try:
            port = pd.read_csv(uploaded)
            if not {"Ticker", "Shares"}.issubset(port.columns):
                st.error("CSV must contain 'Ticker' and 'Shares' columns.")
            else:
                port = port.copy()
                values = []
                for idx, row in port.iterrows():
                    tk = row["Ticker"]
                    sh = float(row["Shares"])
                    try:
                        last_price = yf.download(tk, period="1d", progress=False)["Close"].iloc[-1]
                    except Exception:
                        last_price = np.nan
                    value = (last_price * sh) if not np.isnan(last_price) else np.nan
                    port.loc[idx, "Latest Price"] = last_price
                    port.loc[idx, "Value"] = value
                    values.append(value if not np.isnan(value) else 0.0)
                total_value = sum([v for v in values if v])
                st.dataframe(port)
                st.success(f"Total portfolio value (approx): ${total_value:,.2f}")
                # allow download
                out_csv = port.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download portfolio (CSV)", out_csv, file_name="portfolio_with_values.csv", mime="text/csv")
        except Exception as e:
            logging.exception("Portfolio upload failed")
            st.error(f"Failed to read portfolio: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Chatbot (st.chat_message)
# -------------------------
elif section == "Chatbot":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 Gemini Chatbot (Financial Assistant)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_msg = st.chat_input("Ask a financial question...")
    if user_msg:
        # show user message
        with st.chat_message("user"):
            st.write(user_msg)
        # generate reply
        reply = get_llm_response(user_msg, model_name="gemini-2.5-flash")
        with st.chat_message("assistant"):
            st.write(reply)
        # save history
        st.session_state.chat_history.append({"user": user_msg, "assistant": reply})
    # show previous messages below
    if st.session_state.chat_history:
        st.markdown("### Recent conversation")
        for item in st.session_state.chat_history[-6:]:
            st.info(f"**You:** {item['user']}")
            st.success(f"**Bot:** {item['assistant']}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# End of file
# -------------------------
