import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# App Title
# ----------------------------
st.title("📈 Stock Price Forecasting with LSTM")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# ----------------------------
# Fetch Data
# ----------------------------
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

data = load_data(ticker, start_date, end_date)

# ----------------------------
# Show Raw Data
# ----------------------------
st.subheader(f"Raw Data for {ticker}")
st.write(data.tail())

# ----------------------------
# Plot Closing Price
# ----------------------------
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# ----------------------------
# Preprocessing & Forecast
# ----------------------------
def forecast_stock(data):
    data_close = data[['Date', 'Close']].copy()
    data_close['Date'] = pd.to_datetime(data_close['Date'])
    data_close.set_index('Date', inplace=True)

    dataset = data_close.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training dataset
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    # Create testing dataset
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # ✅ Safe alignment fix
    validation_dates = data_close.index[training_data_len:]
    pred_len = len(predictions)
    date_len = len(validation_dates)
    min_len = min(pred_len, date_len)

    valid_df = pd.DataFrame({
        'Close': data_close['Close'].iloc[training_data_len : training_data_len + min_len].values,
        'Predictions': predictions.flatten()[:min_len]
    }, index=validation_dates[:min_len])

    return valid_df

# ----------------------------
# Run Forecast
# ----------------------------
st.subheader("Forecasted vs Actual Prices")
forecast_data = forecast_stock(data)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['Close'], name="Actual Price"))
fig2.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['Predictions'], name="Predicted Price"))
fig2.layout.update(title_text="LSTM Forecast", xaxis_rangeslider_visible=True)
st.plotly_chart(fig2)

st.write("### Forecasted Data (Last 10 Rows)")
st.write(forecast_data.tail())
