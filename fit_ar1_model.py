import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

def fit_ar1_model(train_data: pd.DataFrame):
    """
    Fits an AR(1) model to the provided time series training data.

    Parameters
    ----------
    train_data : pd.DataFrame
        A pandas DataFrame containing the training time series. 
        Only the first column is used.

    Returns
    -------
    model_fit : ARIMAResults
        The fitted AR(1) model object from statsmodels.
    """
    # ensure we are working with the first column as a 1D series
    series = train_data.iloc[:, 0].astype(float)

    # ARIMA with order (1,0,0) is AR(1)
    model = ARIMA(series, order=(1, 0, 0))
    model_fit = model.fit()

    return model_fit

if __name__ == "__main__":
    # make a tiny fake AR(1)-like series for testing
    import numpy as np
    np.random.seed(42)
    n = 100
    eps = np.random.normal(0, 1, n)
    y = [0]
    for t in range(1, n):
        y.append(0.7 * y[-1] + eps[t])   # AR(1) with phi=0.7

    df = pd.DataFrame(y, columns=["value"])

    model_fit = fit_ar1_model(df)
    print(model_fit.summary())
