import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker = 'KO'
data = yf.download(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))
data = data[['Close']].reset_index()

data['Date'] = pd.to_datetime(data['Date'])
data['Days'] = (data['Date'] - data['Date'].min()).dt.days  

X = data[['Days']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)


future_days = 30
last_day = data['Days'].iloc[-1]
future_X = pd.DataFrame({'Days': np.arange(last_day + 1, last_day + future_days + 1)})
future_preds = model.predict(future_X)


plt.figure(figsize=(12,6))
plt.plot(data['Date'], y, label='Actual Prices')
future_dates = [data['Date'].max() + timedelta(days=i) for i in range(1, future_days+1)]
plt.plot(future_dates, future_preds, label='Predicted Prices', linestyle='--', color='red')
plt.title(f'{ticker} Stock Price Prediction (Next {future_days} Days)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
