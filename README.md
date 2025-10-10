# 📈 FinBot 360: Your AI-Powered Financial Co-Pilot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finbot360-4dgbeq9msxs4ywczrmet4v.streamlit.app/)
&nbsp;
[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**FinBot 360** is a comprehensive, AI-driven financial dashboard designed to empower investors with real-time data, predictive insights, and conversational financial analysis. This application integrates multiple AI and data science modules into a single, intuitive interface, serving as a one-stop solution for market analysis.

---

## ✨ Live Application

### 👉 [**Launch FinBot 360**](https://finbot360-4dgbeq9msxs4ywczrmet4v.streamlit.app/)

---

## 🚀 Core Features

This project combines several powerful modules into one seamless user experience:

*   **🤖 Natural Language Financial Q&A:** A conversational AI assistant, powered by Google's **Gemini**, that can answer complex questions about market trends, financial concepts, and definitions.
*   **📈 LSTM-Based Stock Forecasting:** A predictive tool that uses a Long Short-Term Memory (LSTM) neural network to forecast future stock prices based on historical data. The resulting chart is persistent, allowing for comparison and analysis.
*   **🔴 Real-Time Market Dashboard:** A live sidebar widget that provides:
    *   Real-time stock prices with a robust **fallback system** (tries Alpha Vantage first, then seamlessly switches to Yahoo Finance if the API limit is reached).
    *   A dynamic chart showing the stock's performance over the last month.
*   **😊 Financial Sentiment Analysis:** An NLP tool that uses a specialized **FinBERT** model to analyze the sentiment (positive, negative, or neutral) of financial news, articles, or any text.
*   **📁 Portfolio Performance Analysis:** A utility to upload a portfolio's historical data (CSV/XLSX) and instantly calculate key performance metrics like Total Return, Annualized Volatility, and the Sharpe Ratio.

---

## 🛠️ Tech Stack & Architecture

FinBot 360 is built on a modern data science and AI stack, designed for performance and scalability.

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation & Analysis:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning / Deep Learning:** [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [Scikit-learn](https://scikit-learn.org/)
*   **Natural Language Processing (NLP):** [Hugging Face Transformers](https://huggingface.co/transformers/), [Google Generative AI](https://ai.google.dev/)
*   **Financial Data APIs:** [Alpha Vantage](https://www.alphavantage.co/), [yfinance](https://pypi.org/project/yfinance/)
*   **Plotting:** [Plotly](https://plotly.com/)

#### System Architecture
The application uses a modular design, with a central Streamlit script orchestrating calls to various internal functions and external APIs. A robust fallback mechanism for live data ensures high availability and a seamless user experience.

---

## ⚙️ Getting Started

To run this project locally, follow these simple steps.

### 1. Prerequisites

*   Python 3.10 or higher
*   An environment manager like `venv` or `conda` is recommended.

### 2. Installation & Setup

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/SalmaTech-03/FinBot360.git
cd FinBot360

# Install all required packages
pip install -r requirements.txt
3. Configure API Keys
The application requires API keys for Google Gemini and Alpha Vantage. Store them securely.
Create a folder named .streamlit in the root of the project.
Inside this folder, create a file named secrets.toml.
Add your keys to this file in the following format:
code
Toml
# .streamlit/secrets.toml

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY_HERE"
4. Run the Application
Launch the Streamlit app from your terminal.
code
Bash
streamlit run app.py
The application will open in your default web browser.
