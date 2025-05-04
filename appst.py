import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import ta
import joblib

start_date = "2022-01-01"
end_date = "2025-04-30"

st.title("Stock Price Prediction")

ticker = st.text_input("Enter ticker:", "AAPL") 

df=yf.download(ticker, start=start_date, end=end_date)

st.subheader("Stock Data")
st.write(df.tail())

#visualize the data
st.subheader("Stock Price Chart")
fig1=plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.title(f'{ticker} Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
st.pyplot(fig1)

df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

st.subheader("Stock Price Chart with Moving Averages")
fig2=plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['MA50'], label='MA50',color='red', alpha=0.5)
plt.plot(df['MA200'], label='MA200', alpha=0.5)
plt.title(f'{ticker} Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
st.pyplot(fig2)


st.subheader("RSI Indicator")
rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'].squeeze(), window=14)
df['RSI'] = rsi_indicator.rsi()

fig3=plt.figure(figsize=(10, 4))
plt.plot(df['RSI'], label='RSI (14-day)', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought')
plt.axhline(30, color='green', linestyle='--', label='Oversold')
plt.title('RSI - Relative Strength Index')
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(fig3)

#data preparation
df_model=df[['Close', 'MA50', 'MA200', 'RSI']].dropna()
df_model.head()

df_model['target']= df_model['Close'].shift(-1)
df_model.dropna()

X=df_model.drop('target', axis=1)
y=df_model['target']
df_model = df_model.dropna()



#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#loading the model
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

y_pred=model.predict(X_scaled)
y=y.dropna()
y_pred = y_pred[:-1]



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

#st.subheader("    Model Evaluation    ")
#st.write("    Model Performance Metrics:    ")
#st.write(f"MAE: {mae}")
#st.write(f"MSE: {mse}")
#st.write(f"R²: {r2}")

st.subheader("Predicted vs Actual Prices")
fig4=plt.figure(figsize=(10,6))
plt.plot(y.index, y, label='Actual Prices', color='blue')

plt.title('Stock Price Prediction: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

fig5=plt.figure(figsize=(10,6))
plt.plot(y.index, y_pred, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig5)

latest_data = df_model.drop('target', axis=1).iloc[-1:]
latest_scaled = scaler.transform(latest_data)
next_day_prediction = model.predict(latest_scaled)

st.write("Predicted next closing price:", next_day_prediction[0])