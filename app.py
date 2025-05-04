import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


sns.set(style="darkgrid")


# Define the ticker symbol and date range
ticker = input("enter ticker:")  
start_date = "2022-01-01"
end_date = "2025-04-30"

df=yf.download(ticker, start=start_date, end=end_date)

#we are using moving averages to smooth the data and identify trends
# Calculate moving averages

df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Plot closing price and moving averages
plt.figure(figsize=(12, 5))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['MA50'], label='50-Day MA', color='orange')
plt.plot(df['MA200'], label='200-Day MA', color='rEd')
plt.title('AAPL Stock Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

import ta 
rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'].squeeze(), window=14)
df['RSI'] = rsi_indicator.rsi()

df_model=df[['Close', 'MA50', 'MA200', 'RSI', 'Volume']].dropna()


df_model['target']= df_model['Close'].shift(-1)
df_model.dropna()

X=df_model.drop('target', axis=1)


y=df_model['target']
df_model = df_model.dropna() 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#model training
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train_scaled, y_train)


#model prediction
y_pred=model.predict(X_test_scaled)
y_test=y_test.dropna()
y_pred = y_pred[:-1]


#evaluating the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")

print("last predictions")
print(y_pred[-1:])
print("last actual values")
print(y_test[-1:])



import matplotlib.pyplot as plt

# Plot actual vs predicted prices
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red', linestyle='--')
plt.title('Stock Price Prediction: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

latest_data = df_model.drop('target', axis=1).iloc[-1:]
latest_scaled = scaler.transform(latest_data)
next_day_prediction = model.predict(latest_scaled)

print("Predicted next closing price:", next_day_prediction[0])