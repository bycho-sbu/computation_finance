import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

"""
1. generate random walk data (non-stationary, like stock price)
2. fit arima(1,1,1) model
   - p=1: ar(1), use 1 lag of past value
   - d=1: first differencing, to make data stationary
   - q=1: ma(1), use 1 lag of past error
3. print model summary
4. forecast next 10 steps
5. visualize original data + forecast
"""

# fixed data 
np.random.seed(42)
n = 100
eps = np.random.normal(0, 1, n)
price = [100]  # starting price
for t in range(1, n):
    price.append(price[-1] + eps[t])  # random walk (non-stationary)

series = pd.Series(price)

# ARIMA(1,1,1) (ar, diff, ma)
model = ARIMA(series, order=(1,1,1))  # (ar 1day before, d=1 for stationary, ma 1day before)
model_fit = model.fit()

# summary
print(model_fit.summary())

# forecasting
forecast = model_fit.forecast(steps=10)
print("Forecast:\n", forecast)

# visualization
plt.figure(figsize=(10,5))
plt.plot(series, label="Original Series")
plt.plot(range(len(series), len(series)+10), forecast, label="Forecast", color="red")
plt.legend()
plt.show()
