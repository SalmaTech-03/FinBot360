import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import google.generativeai as genai # <--- THIS LINE IS NOW CORRECTED
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
def get_llm_response(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    if not GEMINI_AVAILABLE: 
        return "Chatbot is unavailable because the Gemini API key is not configured."
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

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_stock(data: pd.DataFrame):
    # 1. Prepare and clean the data
    data_close = data[['Close']].copy()
    data_close.dropna(inplace=True)
    if len(data_close) < 80:
        st.error("Not enough valid data points to forecast (need at least 80).")
        return None, None
    dataset = data_close.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 2. Create training data
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

    # 3. Train the model
    model = create_lstm_model((x_train.shape[1], 1))
    with st.spinner('Training LSTM model... This may take a moment.'):
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)

    # 4. Create test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    if not x_test:
        st.warning("Not enough data to form a validation set.")
        return data_close[:training_data_len], None

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 5. Get predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # 6. Construct final dataframes for plotting
    train_df = data_close[:training_data_len]
    valid_df = data_close[training_data_len:]
    valid_df['Predictions'] = predictions
    
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
# ✅ SIDEBAR AND MAIN PAGE (No changes below this line)
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
        except Exception:
            try:
                st.warning("⚠️ Alpha Vantage failed or limit reached. Showing Yahoo Finance data instead.")
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                st.subheader(info.get("longName", ticker_symbol))
                st.metric("Price", f"${info.get('currentPrice', info.get('previousClose', 0))}", f"{info.get('regularMarketChange', 0):.2f} ({info.get('regularMarketChangePercent', 0):.2f}%)")
                st.text(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
                st.text(f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if info.get("marketCap") else "Market Cap: N/A")
            except Exception as e2:
                st.error(f"Could not fetch data from either Alpha Vantage or yfinance. Error: {e2}")
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

    with st.expander("📊 Stock Forecasting", expanded=True):
        ticker = st.text_input("Enter Ticker (e.g., AAPL):", "AAPL", help="For non-US stocks, add exchange suffix (e.g., DMART.NS)").upper()
        if 'forecast_fig' not in st.session_state:
            st.session_state.forecast_fig = None
        if st.button("Generate Forecast"):
            st.session_state.forecast_fig = None
            data = fetch_stock_data(ticker, "2020-01-01", pd.to_datetime("today").strftime('%Y-%m-%d'))
            if not data.empty:
                train, valid = forecast_stock(data)
                if train is not None:
                    st.session_state.forecast_fig = plot_forecast(train, valid)
        if st.session_state.forecast_fig:
            st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
        else:
            st.info("Enter a ticker and click 'Generate Forecast' to see the stock price prediction.")

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