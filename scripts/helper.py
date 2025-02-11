import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def generate_varimax_data(n=100, seed=42, p=2, q=2, d=1, anomalies=True):


    np.random.seed(seed)

    # Generate X variables
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack((x1, x2))

    # Initialize y and error terms
    y = np.zeros(n)
    # generate normal dist error with mean zero and variance 2
    
    e = np.random.normal(0, 0.1, n)

    # Coefficients for autoregressive (AR) and moving average (MA) components
    phi = np.random.uniform(-0.5, 0.5, size=p)  # AR coefficients
    theta = np.random.uniform(-0.5, 0.5, size=q)  # MA coefficients
    gamma = np.random.uniform(-1, 1, size=2)  # Coefficients for exogenous variables

    # Generate the VARIMA-X time series
    for t in range(max(p, q), n):
        ar_terms = sum(phi[i] * Y[t - i - 1] for i in range(p)) if p > 0 else 0
        ma_terms = sum(theta[j] * errors[t - j - 1] for j in range(q)) if q > 0 else 0
        exog_terms = gamma.dot(exogenous[t])
        Y[t] = ar_terms + ma_terms + exog_terms + errors[t]

    # Apply differencing for stationarity if d > 0
    for _ in range(d):
        Y = np.diff(Y, prepend=0)

    # Induce anomalies if specified
    if induce_anomalies:
        if anomaly_indices is None:
            anomaly_indices = np.random.choice(range(max(p, q), n), size=n // 10, replace=False)
        for idx in anomaly_indices:
            Y[idx] += np.random.uniform(5, 10)  # Add a large anomaly to these points

    # Create DataFrame
    data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2}, index=pd.date_range(start='2020-01-01', periods=n))
    return data


# simulate data function for VARIMA-X


if __name__ == '__main__':
    print("Hello")
    generate_varimax_data(n=100, seed=42, p=2, q=2, d=1, anomalies=True)



