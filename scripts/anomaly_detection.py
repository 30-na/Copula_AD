
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
from pmdarima import auto_arima

path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD"


def read_time_series(file_name):
    eq_raw = pd.read_csv(
    os.path.join(path, file_name),
    skiprows=18,
    parse_dates=["Time"],
    low_memory=False
    )

    eq_raw.rename(columns={" Sample": "Sample"}, inplace=True)
    eq_raw['Sample'] = pd.to_numeric(eq_raw['Sample'], errors='coerce')

    # Convert "Time" with invalid rows becoming NaT
    eq_raw["Time"] = pd.to_datetime(eq_raw["Time"], errors='coerce')

    # Drop rows with NaT in the "Time" column
    eq_raw = eq_raw.dropna(subset=["Time"])
    eq_raw["Time"] = pd.to_datetime(eq_raw["Time"])
    eq_raw = eq_raw.set_index("Time")
    return eq_raw

def simulate_arima_exog(c, phi, theta, beta, omega_mean, omega_Sigma, initial_state, n, seed=534):
    np.random.seed(seed)

    m = len(omega_mean)  # Number of variables
    p = len(phi)  # AR order
    q = len(theta)  # MA order

    # Simulate noise (Multivariate Normal)
    omega = np.random.multivariate_normal(omega_mean, omega_Sigma, n)

    # Initialize Data Storage
    simulated_data = np.zeros((n, m))  # Shape (n, m)
    simulated_data[:max(p, q), :] = 1 # Correct shape

  # Set initial values

    # Generate ARIMA simulated data
    for v in range(m):
        for t in range(max(p, q), n):
            AR_term = phi @ simulated_data[t-p:t, v][::-1]
            Ma_term = theta @ omega[t-q:t, v][::-1]
            simulated_data[t, v] = c + AR_term + Ma_term + omega[t, v]

    # Add exogenous variables
    exon_term = simulated_data[:, 1:] @ beta
    simulated_data[:, 0] += exon_term

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

def add_earthquakes(simulated_data, num_earthquakes, scale, duration, seed=123):
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
        num_windows = (len(data) - window_size) // step + 1
        return np.array([data[i:i+window_size] for i in range(0, num_windows * step, step)])

    def fit_arima_sigma2(windows, order):
        sigma2_values = []
        for window in tqdm(windows, desc="Fitting ARIMA"):
            model = sm.tsa.ARIMA(window, order=order)
            result = model.fit()
            sigma2_values.append(result.params[-1])
        return np.array(sigma2_values)

    windows = create_windows(np.asarray(data), window_size, overlap_size)
    sigma2_values = fit_arima_sigma2(windows, order)

    return sigma2_values


def compute_sigma2_values_timestamps(data, window_size, overlap_size, order):
    def create_windows(data_array, window_size, overlap_size):
        step = window_size - overlap_size
        num_windows = (len(data_array) - window_size) // step + 1
        start_indices = [i for i in range(0, num_windows * step, step)]
        windows = np.array([data_array[i:i+window_size] for i in start_indices])
        end_timestamps = [data.index[i + window_size - 1] for i in start_indices]
        return windows, pd.to_datetime(end_timestamps)

    def fit_arima_sigma2(windows, order):
        sigma2_values = []
        for window in tqdm(windows, desc="Fitting ARIMA"):
            model = sm.tsa.ARIMA(window, order=order)
            result = model.fit()
            sigma2_values.append(result.params[-1])
        return np.array(sigma2_values)

    data = data.copy()
    windows, timestamps = create_windows(np.asarray(data), window_size, overlap_size)
    sigma2_values = fit_arima_sigma2(windows, order)

    return pd.Series(sigma2_values, index=timestamps)


def compute_sigma2_values_timestamps_autoArima(data, window_size, overlap_size):
    def create_windows(data_array, window_size, overlap_size):
        step = window_size - overlap_size
        num_windows = (len(data_array) - window_size) // step + 1
        start_indices = [i for i in range(0, num_windows * step, step)]
        windows = np.array([data_array[i:i+window_size] for i in start_indices])
        end_timestamps = [data.index[i + window_size - 1] for i in start_indices]
        return windows, pd.to_datetime(end_timestamps)

    def fit_arima_sigma2(windows):
        sigma2_values = []
        for window in tqdm(windows, desc="Fitting Auto ARIMA"):
            model = auto_arima(window, seasonal=False, stepwise=True, error_action="ignore", suppress_warnings=True)
            res = model.arima_res_
            sigma2_values.append(getattr(res, "sigma2", res.params[-1]))
        return np.array(sigma2_values)

    data = data.copy()
    windows, timestamps = create_windows(np.asarray(data), window_size, overlap_size)
    sigma2_values = fit_arima_sigma2(windows)

    return pd.Series(sigma2_values, index=timestamps)


def break_time_series(eq_raw):
    segments = []
    segment_start_idx = 0

    for i in range(1, len(eq_raw)):
        if  eq_raw.index[i-1] >= eq_raw.index[i]:
            # Break the data when the condition is met
            segments.append(eq_raw[segment_start_idx:i])
            segment_start_idx = i
    
    # Append the last segment
    segments.append(eq_raw[segment_start_idx:])
    
    return segments

def extract_arima_coefficients(y_series):
    model = pm.auto_arima(y_series, seasonal=False, stepwise=False, suppress_warnings=False)
    model_fit = model.fit(y_series)
    return model_fit.params()

def find_min_l2_lag(reference_series: pd.Series, series_a: pd.Series, max_lag: int = 100):
    reference_series = reference_series.to_numpy()
    series_a = series_a.to_numpy()
    
    min_len = min(len(reference_series), len(series_a))
    reference_series = reference_series[:min_len]
    series_a = series_a[:min_len]

    best_lag = 0
    min_dist = float('inf')

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted_b = series_a[-lag:min_len]
            ref = reference_series[:min_len + lag]
        else:
            shifted_b = series_a[:min_len - lag]
            ref = reference_series[lag:min_len]
        
        if len(shifted_b) == 0:
            continue

        dist = np.linalg.norm(ref - shifted_b)
        if dist < min_dist:
            min_dist = dist
            best_lag = lag

    return best_lag

def generate_window_timestamps(start, end, window_size, overlap, fs):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    step = window_size - overlap
    total_points = int(((end - start).total_seconds()) * fs)
    n_windows = (total_points - window_size) // step + 1
    
    offsets = [int(i * step + window_size // 2) for i in range(n_windows)]
    timestamps = [start + pd.to_timedelta(offset / fs, unit='s') for offset in offsets]
    
    return pd.to_datetime(timestamps)



def compute_sigma2_autoarima(data, window_size, overlap_size, seasonal=True, m=86400):
    def create_windows(data_array, window_size, overlap_size):
        step = window_size - overlap_size
        num_windows = (len(data_array) - window_size) // step + 1
        start_indices = [i for i in range(0, num_windows * step, step)]
        windows = np.array([data_array[i:i+window_size] for i in start_indices])
        end_timestamps = [data.index[i + window_size - 1] for i in start_indices]
        return windows, pd.to_datetime(end_timestamps)

    def fit_autoarima_sigma2(windows, seasonal, m):
        sigma2_values = []
        for window in tqdm(windows, desc="Fitting Auto-ARIMA"):
            try:
                model = pm.auto_arima(
                    window,
                    seasonal=seasonal,
                    m=m,
                    suppress_warnings=True,
                    error_action="ignore",
                    stepwise=True
                )
                sigma2_values.append(model.arima_res_.params[-1])  # sigma²
            except:
                sigma2_values.append(np.nan)
        return np.array(sigma2_values)

    data = data.copy()
    windows, timestamps = create_windows(np.asarray(data), window_size, overlap_size)
    sigma2_values = fit_autoarima_sigma2(windows, seasonal, m)

    return pd.Series(sigma2_values, index=timestamps)

# Load data
afi_raw = pd.read_csv("processedData/afi.csv", index_col=0, parse_dates=True)
funa_aligned = pd.read_csv("processedData/funa_aligned_5days.csv", index_col=0, parse_dates=True)
tara_aligned = pd.read_csv("processedData/tara_aligned_5days.csv", index_col=0, parse_dates=True)
rao_aligned  = pd.read_csv("processedData/rao_aligned_5days.csv", index_col=0, parse_dates=True)


# round to milliseconds
afi_raw.index = afi_raw.index.round('ms')
funa_aligned.index = funa_aligned.index.round('ms')
tara_aligned.index = tara_aligned.index.round('ms')
rao_aligned.index = rao_aligned.index.round('ms')


# Plot raw data
plt.figure(figsize=(15, 8))

plt.subplot(4, 1, 1)
plt.plot(afi_raw.index, afi_raw['Sample'], '.', color='dimgray', markersize=.1)
plt.title('AFI Station')
plt.ylabel('Sample')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 2)
plt.plot(funa_aligned.index, funa_aligned['Sample'], '.', color='dimgray', markersize=.1)
plt.title('FUNA Station')
plt.ylabel('Sample')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 3)
plt.plot(tara_aligned.index, tara_aligned['Sample'], '.', color='dimgray', markersize=.1)
plt.title('TARA Station')
plt.ylabel('Sample')
plt.xlabel('Time')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 4)
plt.plot(rao_aligned.index, rao_aligned['Sample'], '.', color='dimgray', markersize=.1)
plt.title('RAO Station')
plt.ylabel('Sample')
plt.xlabel('Time')
plt.gca().set_ylabel('')

plt.tight_layout()
plt.show()

# 4 days bfore the eartquake
start = "2018-08-15 1:00:00"
end = "2018-08-19 23:00:00"

afi_5d = afi_raw.loc[start:end]
funa_5d = funa_aligned.loc[start:end]
tara_5d = tara_aligned.loc[start:end]
rao_5d  = rao_aligned.loc[start:end]

df = pd.concat([afi_5d, funa_5d, tara_5d, rao_5d], axis=1, join='inner')
df.columns = ['AFI', 'FUNA', 'TARA', 'RAO']

afi_sigma2_5d = compute_sigma2_autoarima(data=afi_5d["Sample"], window_size=400, overlap_size=40, seasonal=True, m=86400)
funa_sigma2_5d = compute_sigma2_autoarima(funa_5d, window_size=400, overlap_size=40, seasonal=True, m=86400)
tara_sigma2_5d = compute_sigma2_autoarima(tara_5d, window_size=400, overlap_size=40, seasonal=True, m=86400)
rao_sigma2_5d = compute_sigma2_autoarima(rao_5d, window_size=400, overlap_size=40, seasonal=True, m=86400)


afi_sigma2_5d= compute_sigma2_values_timestamps(afi_5d["Sample"], window_size=400, overlap_size=40, order=(1, 2, 1))
funa_sigma2_5d= compute_sigma2_values_timestamps(funa_5d["Sample"], window_size=400, overlap_size=40, order=(1, 2, 1))
tara_sigma2_5d= compute_sigma2_values_timestamps(tara_5d["Sample"], window_size=400, overlap_size=40, order=(1, 2, 1))
rao_sigma2_5d= compute_sigma2_values_timestamps(rao_5d["Sample"],  window_size=400, overlap_size=40, order=(1, 2, 1))

df_5d = pd.concat([afi_sigma2_5d, funa_sigma2_5d, tara_sigma2_5d, rao_sigma2_5d], axis=1, join='inner')
df_5d.columns = ['AFI', 'FUNA', 'TARA', 'RAO']
df_5d.to_csv("processedData/sigma2_5d_121.csv")

afi_sigma2_5d.to_csv("processedData/afi_sigma2_5d.csv")
funa_sigma2_5d.to_csv("processedData/funa_sigma2_5d.csv")
tara_sigma2_5d.to_csv("processedData/tara_sigma2_5d.csv")
rao_sigma2_5d.to_csv("processedData/rao_sigma2_5d.csv")

