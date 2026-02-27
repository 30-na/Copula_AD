
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm

path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Simulated_data"

def simulate_arima(c, phi, theta, omega_mean, omega_Sigma, n, seed=534):
    np.random.seed(seed)

    m = len(omega_mean)  # Number of variables
    p = len(phi)  # AR order
    q = len(theta)  # MA order

    # Simulate noise (Multivariate Normal)
    omega = np.random.multivariate_normal(omega_mean, omega_Sigma, n)

    # Initialize Data Storage
    simulated_data = np.zeros((n, m))  # Shape (n, m)
    simulated_data[:max(p, q), :] = 1 # Set initial values


    # Generate ARIMA simulated data
    for v in range(m):
        for t in range(max(p, q), n):
            AR_term = phi @ simulated_data[t-p:t, v][::-1]
            Ma_term = theta @ omega[t-q:t, v][::-1]
            simulated_data[t, v] = c + AR_term + Ma_term + omega[t, v]

    return simulated_data

def plot_time_series(data, titles):
    m = data.shape[1]  # Number of variables (columns)
    
    fig, axes = plt.subplots(m, 1, figsize=(8, 4), sharex=True)
    
    if m == 1:
        axes = [axes]  # Ensure axes is iterable for a single variable case
    
    for i in range(m):
        axes[i].plot(data[:, i],
                      color='gray', 
                      alpha=0.8, 
                      linewidth=0.5, 
                      marker='*', 
                      markersize=1)
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].set_title(titles[i])
        axes[i].grid(True)

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

def add_anomalies(simulated_data, num_earthquakes, scale, duration, seed=123):
    np.random.seed(seed)
    n, m = simulated_data.shape  # Get data shape
    earthquake_data = simulated_data.copy()
    
    for _ in range(num_earthquakes):
        # Choose a random start point for each earthquake
        start = np.random.randint(0, n - duration)
        peak = start + duration // 2  # Middle of the earthquake

        # Apply earthquake effect: grows, peaks, then weakens
        for t in range(start, start + duration):
            if t <= peak:
                factor = (t - start) / (duration / 2)  # Increasing phase
            else:
                factor = (start + duration - t) / (duration / 2)  # Decreasing phase
            earthquake_data[t, :] += np.random.randn(m) * scale * factor  # Modify data
    return earthquake_data


def compute_sigma2_values(data, window_size, overlap_size, order):
    def create_windows(data, window_size, overlap_size):
        step = window_size - overlap_size
        num_windows = (len(data) - overlap_size) // step
        return np.array([data[i:i+window_size] for i in range(0, num_windows * step, step)])
    
    def fit_arima_sigma2(windows, order):
        sigma2_values = []
        for window in windows:
            model = sm.tsa.ARIMA(window, order=order)
            result = model.fit()
            sigma2_values.append(result.params[-1])  # Extract sigma²
        return np.array(sigma2_values)

    # Compute sigma² for each variable
    sigma2_matrix = []
    for i in range(data.shape[1]):  # Iterate over all columns (variables)
        windows = create_windows(data[:, i], window_size, overlap_size)
        sigma2_values = fit_arima_sigma2(windows, order)
        sigma2_matrix.append(sigma2_values)

    return np.column_stack(sigma2_matrix)  # Stack results column-wise



# data simulation
import numpy as np
import pandas as pd

def simulate_data(seed_label, seed, start_time):
    X_layer = simulate_arima(
        c=0, 
        phi=np.array([1.2, -0.5]),
        theta=np.array([.7, .3]),
        omega_mean=np.array([0, 0, 0, 0]),
        omega_Sigma=np.array([[1, 0.7, 0.7, 0.7],
                              [0.7, 1, 0.7, 0.7],
                              [0.7, 0.7, 1, 0.7],
                              [0.7, 0.7, 0.7, 1]]),
        n=50000,
        seed=seed
    )
    
    Y_layer = simulate_arima(
        c=0, 
        phi=np.array([0]),
        theta=np.array([.2, .4]),
        omega_mean=np.array([0, 0, 0]),
        omega_Sigma=np.array([[1, 0.5, 0.5],
                              [0.5, 1, 0.5],
                              [0.5, 0.5, 1]]),
        n=50000,
        seed=seed + 100
    )
    
    Z_layer = simulate_arima(
        c=0, 
        phi=np.array([.5, -.3, .2, .1]),
        theta=np.array([.3]),
        omega_mean=np.array([0, 0, 0, 0, 0]),
        omega_Sigma=np.array([[1, 0.9, 0.9, 0.9, 0.9],
                              [0.9, 1, 0.9, 0.9, 0.9],
                              [0.9, 0.9, 1, 0.9, 0.9],
                              [0.9, 0.9, 0.9, 1, 0.9],
                              [0.9, 0.9, 0.9, 0.9, 1]]),
        n=50000,
        seed=seed + 200
    )
    
    all_layers = np.concatenate([X_layer, Y_layer, Z_layer], axis=1)
    all_layers =  add_anomalies(all_layers, num_earthquakes=3, scale=2, duration=100, seed=5451)
    num_x = X_layer.shape[1]
    num_y = Y_layer.shape[1]
    num_z = Z_layer.shape[1]
    
    column_names = [f"{seed_label}%X{i+1}" for i in range(num_x)] + \
                   [f"{seed_label}%Y{i+1}" for i in range(num_y)] + \
                   [f"{seed_label}%Z{i+1}" for i in range(num_z)]
    
    df = pd.DataFrame(all_layers, columns=column_names)
    df.index = pd.date_range(start=start_time, periods=df.shape[0], freq='30min')
    df.index.name = "UtcTime"
    
    return df

# Loop over different seed numbers with labels
seed_labels = ["one", "two", "three"]
seeds = [100, 200, 300]
start_time = "2023-08-01 00:00:00+00:00"
data_frames = [simulate_data(label, seed, start_time) for label, seed in zip(seed_labels, seeds)]

# Merge all data into one DataFrame
final_df = pd.concat(data_frames, axis=1)

# Save to CSV
final_df.to_csv("simulated_data.csv")


# Save to CSV
final_df.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Simulated_data\mergedOutput.csv", index=True)



#####################

long_df_raw_ignore = transform_to_long_format_rawdata(final_df)
mtad_eucl_mag = mtad_analysis_by_sample(df_long=long_df_raw_ignore, window_size=30, distance="Euclidean")
plot_timeseries_with_resets(mtad_eucl_mag, reset=None)

path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Simulated_data\mergedOutput.csv"
df = pd.read_csv(path, index_col="UtcTime", parse_dates=True)

# Filter only 'seed=100' variables (label 'one')
df_seed100 = final_df.loc[:, final_df.columns.str.startswith("one%")]
df_seed100_subset = df_seed100.iloc[:, :4]
import matplotlib.pyplot as plt

def plot_all_variables(df, title="Simulated Time Series (seed=100)"):
    n_cols = df.shape[1]
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 2.5 * n_cols), sharex=True)
    axes = axes.flatten() if n_cols > 1 else [axes]

    for i, col in enumerate(df.columns):
        axes[i].plot(df.index, df[col], linewidth=0.5, color='black')
        axes[i].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


plot_all_variables(df_seed100_subset)



