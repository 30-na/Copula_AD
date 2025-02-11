from helper import *

# simulate data function for VARIMA-X
import numpy as np
from statsmodels.tsa.api import VAR
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD"

def generate_data(n, cov_matrix, num_outliers=8, seed=467):
    np.random.seed(seed)
    data = np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix, size=n)
    columns = ['y'] + [f'x{i}' for i in range(1, cov_matrix.shape[0])]
    df = pd.DataFrame(data, columns=columns)
    outlier_indices = np.random.choice(n, num_outliers, replace=False)
    
    for idx in outlier_indices:
        outlier = np.random.uniform(5, 10)
        df.loc[idx, :] += outlier
        print(idx)
    return df

def plot_time_series(data):

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot 'y'
    axs[0].plot(data.index, data['y'], label='y',  linewidth=2)
    axs[0].set_title('Time Series Plot of y')
    axs[0].set_ylabel('y')
    axs[0].grid(True)
    axs[0].legend()

    # Plot 'x1'
    axs[1].plot(data.index, data['x1'], label='x1',  linewidth=2)
    axs[1].set_title('Time Series Plot of x1')
    axs[1].set_ylabel('x1')
    axs[1].grid(True)
    axs[1].legend()

    # Plot 'x2'
    axs[2].plot(data.index, data['x2'], label='x2',  linewidth=2)
    axs[2].set_title('Time Series Plot of x2')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('x2')
    axs[2].grid(True)
    axs[2].legend()

    # Adjust layout and show
    plt.savefig(os.path.join(path, r"figures\sim.png"))
    plt.clf()
    plt.close()



data = generate_data(n=10000, cov_matrix=np.array([[1, .95, .95],[.95, 1, .95], [.95, .95, 1]]))
plot_time_series(data)


model = SARIMAX(data['y'], exog=data[["x1", "x2"]], order=(1,1,1))


results = model.fit(disp=False)
data['fitted'] = results.fittedvalues
plt.figure(figsize=(12, 6))
plt.plot(data['y'], label='Actual', linewidth=2)
plt.plot(data['fitted'], label='Fitted', linestyle='--', linewidth=2)
plt.title('Actual vs Fitted Values')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path, r"figures\actual_vs_fitted.png"))
plt.clf()
plt.close()

# Diagnostics plots
results.plot_diagnostics(figsize=(12, 8))
plt.savefig(os.path.join(path, r"figures\diagnostics.png"))
plt.clf()
plt.close()

# Extract and print model residuals
residuals = results.resid
print("Residuals Summary:")
print(residuals.describe())

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals', linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title('Residuals Plot')
plt.legend()
plt.savefig(os.path.join(path, r"figures\residuals.png"))
plt.clf()
plt.close()