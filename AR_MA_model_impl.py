import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def simulate_ar2_process():
    # example: AR(2) with phi1=0.6, phi2=-0.3
    ar = np.array([1, -0.6, 0.3])  # ar lag poly: 1 - phi1*L - phi2*L^2
    ma = np.array([1])             # no MA terms
    ar2_process = ArmaProcess(ar, ma)

    # generate sample
    np.random.seed(42)
    y = ar2_process.generate_sample(nsample=200)

    # plot time series
    plt.figure(figsize=(12,4))
    plt.plot(y)
    plt.title("simulated AR(2) time series")
    plt.show()

    # plot acf/pacf
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plot_acf(y, lags=20, ax=plt.gca())
    plt.title("ACF of AR(2)")
    plt.subplot(1,2,2)
    plot_pacf(y, lags=20, ax=plt.gca())
    plt.title("PACF of AR(2)")
    plt.tight_layout()
    plt.show()

def simulate_ma1_process():
    # example: MA(1) with theta1=0.8
    ar = np.array([1])           # no AR terms
    ma = np.array([1, 0.8])      # MA(1) process
    ma1_process = ArmaProcess(ar, ma)

    y_ma = ma1_process.generate_sample(nsample=200)

    plt.figure(figsize=(12,4))
    plt.plot(y_ma)
    plt.title("simulated MA(1) time series")
    plt.show()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plot_acf(y_ma, lags=20, ax=plt.gca())
    plt.title("ACF of MA(1)")
    plt.subplot(1,2,2)
    plot_pacf(y_ma, lags=20, ax=plt.gca())
    plt.title("PACF of MA(1)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_ar2_process()
    simulate_ma1_process()