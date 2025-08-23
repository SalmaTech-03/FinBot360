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

# =================================================================================
# ALL HELPER FUNCTIONS (Your Original, Correct Logic)
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
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

@st.cache_data
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.", icon="❌")
        return pd.DataFrame()
    return data

def preprocess_for_forecasting(data: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data['Close'].values.reshape(-1, 1)), scaler

def analyze_portfolio(df: pd.DataFrame):
    if 'Close' not in df.columns:
        st.error("Uploaded file must contain a 'Close' column for analysis.")
        return None, None, None, None
    daily_returns = df['Close'].pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility != 0 else 0
    return daily_returns, cumulative_returns, volatility, sharpe_ratio

def plot_portfolio_performance(df: pd.DataFrame, cumulative_returns: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Portfolio Price'))
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Cumulative Returns', yaxis='y2'))
    fig.update_layout(title='Portfolio Price and Cumulative Returns', xaxis_title='Date', yaxis_title='Portfolio Price ($)', yaxis2=dict(title='Cumulative Returns (%)', overlaying='y', side='right', showgrid=False), legend=dict(x=0.01, y=0.99))
    return fig

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_stock(data: pd.DataFrame):
    scaled_data, scaler = preprocess_for_forecasting(data)
    if scaled_data is None: return None, None
    training_data_len = int(np.ceil(len(scaled_data) * .8))
    x_train, y_train = [], []
    for i in range(60, len(scaled_data[:training_data_len])):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner('Training LSTM model... This may take a moment.'):
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = scaler.inverse_transform(model.predict(x_test))
    train = data[:training_data_len]; valid = data[training_data_len:].copy(); valid['Predictions'] = predictions
    return train, valid

def plot_forecast(train, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual Prices (Validation)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted Prices', line=dict(color='cyan', dash='dash')))
    fig.update_layout(title='Stock Price Forecast vs. Actual', xaxis_title='Date', yaxis_title='Stock Price ($)', legend=dict(x=0.01, y=0.99))
    return fig

# =================================================================================
# ✅ SIDEBAR AND MAIN PAGE (Your Original Code)
# =================================================================================
with st.sidebar:
    st.title("📈 FinBot 360")
    st.markdown("---")
    st.subheader("API Status")
    if GEMINI_AVAILABLE: st.success("Gemini API: Connected", icon="✅")
    else: st.error("Gemini API: Disconnected", icon="❌")
    try:
        st.secrets["ALPHA_VANTAGE_API_KEY"]; st.success("Alpha Vantage: Connected", icon="✅")
    except (KeyError, FileNotFoundError): st.warning("Alpha Vantage: Not Found", icon="⚠️")
    st.info("To toggle Dark Mode, use the Settings menu (top right).")
    st.markdown("---")
    st.header("Financial Tools")

    with st.expander("🔴 Live Market Dashboard"):
        st_autorefresh(interval=60 * 1000, key="datarefresh")
        st.markdown("Data from Alpha Vantage & Reuters.")
        ticker_symbol = st.text_input("Enter a Stock Ticker:", "IBM").upper()
        try:
            AV_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
            if ticker_symbol:
                fd = FundamentalData(key=AV_API_KEY, output_format='pandas')
                overview_data, _ = fd.get_company_overview(symbol=ticker_symbol)
                ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
                quote_data, _ = ts.get_quote_endpoint(symbol=ticker_symbol)
                st.subheader(f"{overview_data.loc['Name'][0]}")
                st.metric("Price", f"${float(quote_data['05. price'][0]):.2f}", f"{float(quote_data['09. change'][0]):.2f} ({quote_data['10. change percent'][0]})")
                st.text(f"P/E Ratio: {overview_data.loc['PERatio'][0]}")
                st.text(f"Market Cap: ${int(overview_data.loc['MarketCapitalization'][0]):,}")
        except Exception as e:
            st.error(f"Could not fetch live data. API limit may have been reached.")
        st.subheader("Live Financial News")
        feed = feedparser.parse("http://feeds.reuters.com/reuters/businessNews")
        for entry in feed.entries[:3]: st.markdown(f"[{entry.title}]({entry.link})")

    with st.expander("😊 Financial Sentiment Analysis"):
        user_text = st.text_area("Enter text to analyze:", "Apple's stock soared after their strong quarterly earnings report.", height=100)
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing..."):
                result = analyze_sentiment(user_text)
                sentiment = result['label'].upper(); score = result['score']
                if sentiment == 'POSITIVE': st.success(f"Sentiment: {sentiment} (Score: {score:.2f})")
                elif sentiment == 'NEGATIVE': st.error(f"Sentiment: {sentiment} (Score: {score:.2f})")
                else: st.info(f"Sentiment: {sentiment} (Score: {score:.2f})")

    with st.expander("📁 Portfolio Performance Analysis"):
        uploaded_file = st.file_uploader("Upload portfolio CSV/XLSX", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                if 'Date' in df.columns and 'Close' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date']); df = df.set_index('Date')
                    _, cum_returns, volatility, sharpe = analyze_portfolio(df)
                    st.metric("Total Return", f"{cum_returns.iloc[-1]:.2%}")
                    st.metric("Annualized Volatility", f"{volatility:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                else: st.error("File must contain 'Date' and 'Close' columns.")
            except Exception as e: st.error(f"Error processing file: {e}")

    with st.expander("📊 Stock Forecasting"):
        ticker = st.text_input("Enter Ticker (e.g., AAPL):", "AAPL").upper()
        if st.button("Generate Forecast"):
            data = fetch_stock_data(ticker, "2020-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
            if not data.empty:
                train, valid = forecast_stock(data)
                if train is not None:
                    fig = plot_forecast(train, valid)
                    st.plotly_chart(fig, use_container_width=True)

st.title("Natural Language Financial Q&A")
st.markdown("Ask the AI assistant about financial topics, market trends, or definitions. Use the tools in the sidebar for specific analysis.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your financial questions today?"}]
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