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


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    if not GEMINI_AVAILABLE: return "Chatbot is unavailable."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"
        
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(show_spinner="Fetching data...")
def load_stock_data(ticker, period="5y"):
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

# --- YOUR ORIGINAL, STABLE FORECASTING LOGIC ---
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
    fig = go.Figure()
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
                st.metric(
                    label=f"Live Price ({ticker_live})",
                    value=f"${float(quote_data['05. price'][0]):.2f}",
                    delta=f"{float(quote_data['09. change'][0]):.2f} ({quote_data['10. change percent'][0]})"
                )
            except Exception as e:
                st.error("Could not fetch live price. API limit may be reached.")
        st.subheader("Live Financial News")
        try:
            feed = feedparser.parse("http://feeds.reuters.com/reuters/businessNews")
            for entry in feed.entries[:3]:
                st.markdown(f"- [{entry.title}]({entry.link})")
        except:
            st.error("Could not fetch news feed.")

    # --- SENTIMENT ANALYSIS (BUTTONLESS) ---
    with st.expander("😊 Financial Sentiment Analysis", expanded=True):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared on strong earnings.", key="sentiment_text")
        if user_text:
            with st.spinner("Analyzing..."):
                try:
                    sentiment_pipeline = load_sentiment_model()
                    result = sentiment_pipeline(user_text)[0]
                    sentiment = result['label'].title()
                    score = result['score']
                    if sentiment == 'Positive': st.success(f"{sentiment} (Score: {score:.2f})")
                    elif sentiment == 'Negative': st.error(f"{sentiment} (Score: {score:.2f})")
                    else: st.info(f"{sentiment} (Score: {score:.2f})")
                except Exception as e:
                    st.error(f"Could not analyze sentiment. Error: {e}")
                    
    # --- PORTFOLIO ANALYSIS (BUTTONLESS) ---
    with st.expander("📁 Portfolio Performance Analysis", expanded=True):
        uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv", key="portfolio_uploader")
        if uploaded_file is not None:
            # Your portfolio logic would go here
            st.success("File uploaded. Portfolio analysis would be displayed in the main area.")

    # --- STOCK FORECASTING ---
    with st.expander("📊 Stock Forecasting", expanded=True):
        ticker_forecast = st.text_input("Enter Ticker:", "AAPL", key="forecast_ticker").upper()
        if st.button("Generate Forecast", key="forecast_button"):
            data = load_stock_data(ticker_forecast)
            if not data.empty:
                train, valid = forecast_stock(data)
                if train is not None:
                    st.session_state.forecast_fig = plot_forecast(train, valid)
                else:
                    st.session_state.forecast_fig = None
            else:
                st.error(f"No data found for {ticker_forecast}.")
                st.session_state.forecast_fig = None


# =================================================================================
# MAIN PAGE (Chatbot + Forecast Chart Display)
# =================================================================================

st.title("Natural Language Financial Q&A")

# Display the forecast chart on the main page if it exists
if 'forecast_fig' in st.session_state and st.session_state.forecast_fig is not None:
    st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
    st.markdown("---")

# --- CHATBOT RESTORED ---
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
    st.session_state.messages.append({"role": "assistant", "content": response})s