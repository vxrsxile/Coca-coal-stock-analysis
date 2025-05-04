import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

ticker = 'KO'
data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data.reset_index(inplace=True)

print(data.isnull().sum())
data.fillna(method='ffill', inplace=True)
data.fillna(0, inplace=True)
print(data.isnull().sum())

data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data.dropna(inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], data['MA_20'], linestyle='--', label='MA 20')
plt.plot(data['Date'], data['MA_50'], linestyle='--', label='MA 50')
plt.title('Coca-Cola Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
target = 'Close'
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")

live_data = yf.download(ticker, period='1d', interval='1m')
live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
live_data['Daily_Return'] = live_data['Close'].pct_change()
live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
live_data.fillna(0, inplace=True)
latest_features = live_data[features].iloc[-1:]
live_prediction = model.predict(latest_features)
print(f"Predicted Closing Price: {live_prediction[0]}")

st.title('Coca-Cola Stock Price Prediction')
st.line_chart(data.set_index('Date')[['Close', 'MA_20', 'MA_50']])
st.write(f"Predicted Closing Price: {live_prediction[0]}")