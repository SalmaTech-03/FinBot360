+-------------------------------------------------+
|               USER INTERFACE (UI)               |
|            (Built with Streamlit)               |
+-------------------------------------------------+
|                                                 |
|   [MAIN PAGE]                [SIDEBAR]          |
|   - Chatbot Interface        - API Status       |
|   - Forecasting Module       - Financial Tools  |
|                                |                |
+--------------------------------|----------------+
                 |               |
                 +---------------+
                         |
                         v
+-------------------------------------------------+
|              BACKEND LOGIC (app.py)             |
|        (Processes all user interactions)        |
+-------------------------------------------------+
                         |
                         | (Selects appropriate module based on user input)
                         |
+------------------------+------------------------+
|                        |                        |
v                        v                        v
+-----------------+  +-----------------+  +------------------+
| CHATBOT MODULE  |  | FORECAST MODULE |  |   TOOLS MODULE   |
| (get_llm_response)|  (forecast_stock) |(Live, Sentiment, etc.)|
+-----------------+  +-----------------+  +------------------+
        |                    |                  /      |      \
        |                    |                 /       |       \
        v                    v                v        v        v
+-----------------+  +-----------------+  +----------+ +--------+ +-----------+
| EXTERNAL APIs   |  | DATA SOURCES    |  |  AlphaV  | | yfinance | | FinBERT   |
+-----------------+  +-----------------+  +----------+ +--------+ +-----------+
| - Google Gemini |  | - Yahoo Finance |  | (Primary)|(Fallback)| |(Local Model)|
+-----------------+  +-----------------+  +----------+ +--------+ +-----------+
