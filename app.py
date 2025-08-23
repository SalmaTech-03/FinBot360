import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import google.generativeai as genai
import logging
from io import StringIO

# --- Imports for the Sidebar Tools ---
from streamlit_autorefresh import st_autorefresh
import feedparser
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

# Note: TensorFlow and Scikit-learn imports have been removed.

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
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text: str):
    return load_sentiment_model()(text)[0]

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

# Note: All forecasting functions (forecast_stock, plot_forecast, etc.) have been removed.

# =================================================================================
# ✅ SIDEBAR - TOOLS & CONTROLS
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
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce') # Ensure 'Close' is numeric
                    df.dropna(subset=['Close'], inplace=True) # Drop rows where 'Close' is not a number
                    
                    daily_returns, cum_returns, volatility, sharpe = analyze_portfolio(df)

                    if cum_returns is not None:
                         st.metric("Total Return", f"{cum_returns.iloc[-1]:.2%}")
                         st.metric("Annualized Volatility", f"{volatility:.2%}")
                         st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                         # Add the performance chart for the portfolio
                         fig = plot_portfolio_performance(df, cum_returns)
                         st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Could not calculate portfolio metrics.")

                else: st.error("File must contain 'Date' and 'Close' columns.")
            except Exception as e: st.error(f"Error processing file: {e}")

    # Note: The "Stock Forecasting" expander has been removed.

# =================================================================================
# ✅ MAIN PAGE - CHATBOT INTERFACE
# =================================================================================

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