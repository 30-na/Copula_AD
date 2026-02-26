
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

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
        num_windows = (len(data) - overlap_size) // step
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



# Load data
afi_raw = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t00_42_55z_AFI.csv")

funa_raw = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t02_38_12z_FUNA.csv")
eq_seperate_list_funa = break_time_series(funa_raw)
funa_raw_00 = eq_seperate_list_funa[0]
funa_raw_01 = eq_seperate_list_funa[1]

tara_raw = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t01_39_26z_TARA.csv")
eq_seperate_list_tara = break_time_series(tara_raw)
tara_raw_00 = eq_seperate_list_tara[0]
tara_raw_01 = eq_seperate_list_tara[1]

rao_raw = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t03_05_29z_RAO.csv")
eq_seperate_list_rao = break_time_series(rao_raw)
rao_raw_00 = eq_seperate_list_rao[0]
rao_raw_01 = eq_seperate_list_rao[1]

len(rao_raw_00)
len(tara_raw_01)
len(afi_raw)
len(funa_raw_00)

# Plot raw data
plt.figure(figsize=(15, 8))

plt.subplot(4, 1, 1)
plt.plot(afi_raw.index, afi_raw['Sample'], '.', color='dimgray', markersize=.1)
plt.title('AFI Station')
plt.ylabel('Sample')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 2)
plt.plot(funa_raw_00.index, funa_raw_00['Sample'], '.', color='dimgray', markersize=.1)
plt.title('FUNA Station')
plt.ylabel('Sample')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 3)
plt.plot(tara_raw_01.index, tara_raw_01['Sample'], '.', color='dimgray', markersize=.1)
plt.title('TARA Station')
plt.ylabel('Sample')
plt.xlabel('Time')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 4)
plt.plot(rao_raw_00.index, rao_raw_00['Sample'], '.', color='dimgray', markersize=.1)
plt.title('RAO Station')
plt.ylabel('Sample')
plt.xlabel('Time')
plt.gca().set_ylabel('')

plt.tight_layout()
plt.show()



# Zoom in at time of earthquick

start = "2018-08-19 00:15:00"
end =   "2018-08-19 00:40:00"

afi_eq_time = afi_raw.loc[start:end]
funa_eq_time = funa_raw_00.loc[start:end]
tara_eq_time = tara_raw_01.loc[start:end]
rao_eq_time = rao_raw_00.loc[start:end]


###########################
afi_eq =  pd.read_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\data\afi.csv", index_col=0)
funa_eq =  pd.read_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\data\funa.csv", index_col=0)
rao_eq =  pd.read_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\data\rao.csv", index_col=0)
tara_eq =  pd.read_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\data\tara.csv", index_col=0)

afi_raw = afi_eq.copy()
funa_raw = funa_eq.copy()
rao_raw = rao_eq.copy()
tara_raw = tara_eq.copy()
################################################

# Plot raw data
plt.figure(figsize=(15, 8))

plt.subplot(4, 1, 1)
plt.plot(afi_eq['Sample'].values, '.', color='dimgray', markersize=0.1)
plt.title('AFI Station')
plt.gca().set_xticks([])
plt.gca().set_ylabel('')

plt.subplot(4, 1, 2)
plt.plot(funa_raw['Sample'].values, '.', color='dimgray', markersize=0.1)
plt.title('FUNA Station')
plt.gca().set_xticks([])
plt.gca().set_ylabel('')

plt.subplot(4, 1, 3)
plt.plot(tara_raw['Sample'].values, '.', color='dimgray', markersize=0.1)
plt.title('TARA Station')
plt.gca().set_xticks([])
plt.gca().set_ylabel('')

plt.subplot(4, 1, 4)
plt.plot(rao_raw['Sample'].values, '.', color='dimgray', markersize=0.1)
plt.title('RAO Station')
plt.gca().set_xticks([])
plt.gca().set_ylabel('')

plt.tight_layout()
plt.show()



## Find the lag

def find_lag_l2_distance(reference_series: pd.Series, series_a: pd.Series, max_lag: int = 7200):
    reference_series = reference_series.to_numpy()
    series_a = series_a.to_numpy()

    # Ensure equal length
    n = min(len(reference_series), len(series_a))
    reference_series = reference_series[:n]
    series_a = series_a[:n]

    ref = reference_series[max_lag:n - max_lag]
    distances = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        start_idx = max_lag + lag
        end_idx = n - max_lag + lag
        shifted = series_a[start_idx:end_idx]
        d = np.linalg.norm(ref - shifted)
        distances.append(d)

    return {"lag": list(lags), "distance": distances}

funa_lags_l2dist = find_lag_l2_distance(afi_raw, funa_raw, max_lag=7200) # 3 minutes as max lag
tara_lags_l2dist = find_lag_l2_distance(afi_raw, tara_raw, max_lag=7200)
rao_lags_l2dist = find_lag_l2_distance(afi_raw, rao_raw, max_lag=7200)


# Get lag and distance arrays
funa_lags = np.array(funa_lags_l2dist["lag"])
funa_dist = np.array(funa_lags_l2dist["distance"])
funa_best_idx = np.argmin(funa_dist)
funa_best_lag = funa_lags[funa_best_idx]

tara_lags = np.array(tara_lags_l2dist["lag"])
tara_dist = np.array(tara_lags_l2dist["distance"])
tara_best_idx = np.argmin(tara_dist)
tara_best_lag = tara_lags[tara_best_idx]

rao_lags = np.array(rao_lags_l2dist["lag"])
rao_dist = np.array(rao_lags_l2dist["distance"])
rao_best_idx = np.argmin(rao_dist)
rao_best_lag = rao_lags[rao_best_idx]

# Plot
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(funa_lags, funa_dist, '.', color='dimgray', markersize=1)
plt.plot(funa_best_lag, funa_dist[funa_best_idx], 'o', color='red', markersize=6,
         label=f'Min Distance (lag={funa_best_lag})')
plt.title("FUNA: L2 Distance vs Lag")
plt.xlabel("Lag")
plt.ylabel("Distance")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(tara_lags, tara_dist, '.', color='dimgray', markersize=1)
plt.plot(tara_best_lag, tara_dist[tara_best_idx], 'o', color='red', markersize=6,
         label=f'Min Distance (lag={tara_best_lag})')
plt.title("TARA: L2 Distance vs Lag")
plt.xlabel("Lag")
plt.ylabel("Distance")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(rao_lags, rao_dist, '.', color='dimgray', markersize=1)
plt.plot(rao_best_lag, rao_dist[rao_best_idx], 'o', color='red', markersize=6,
         label=f'Min Distance (lag={rao_best_lag})')
plt.title("RAO: L2 Distance vs Lag")
plt.xlabel("Lag")
plt.ylabel("Distance")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




# Get best lags
best_lag_funa = funa_lags_l2dist["lag"][np.argmin(funa_lags_l2dist["distance"])]
best_lag_tara = tara_lags_l2dist["lag"][np.argmin(tara_lags_l2dist["distance"])]
best_lag_rao  = rao_lags_l2dist["lag"][np.argmin(rao_lags_l2dist["distance"])]

# Align series by shifting index (no data shift)
funa_aligned = funa_raw.copy()
tara_aligned = tara_raw.copy()
rao_aligned  = rao_raw.copy()

funa_aligned.index = pd.to_datetime(funa_aligned.index)
tara_aligned.index = pd.to_datetime(tara_aligned.index)
rao_aligned.index = pd.to_datetime(rao_aligned.index)


funa_aligned.index = funa_aligned.index - pd.to_timedelta(best_lag_funa * 0.025, unit='s')
tara_aligned.index = tara_aligned.index - pd.to_timedelta(best_lag_tara * 0.025, unit='s')
rao_aligned.index  = rao_aligned.index  - pd.to_timedelta(best_lag_rao  * 0.025, unit='s')

# Save aligned data
funa_aligned.to_csv("processedData/funa_aligned_5days.csv")
tara_aligned.to_csv("processedData/tara_aligned_5days.csv")
rao_aligned.to_csv("processedData/rao_aligned_5days.csv")



# Zoom in at time of earthquick for aligned time

start = "2018-08-19 00:15:00"
end =   "2018-08-19 00:40:00"

afi_eq_time = afi_raw.loc[start:end]
funa_eq_time = funa_aligned.loc[start:end]
tara_eq_time = tara_aligned.loc[start:end]
rao_eq_time = rao_aligned.loc[start:end]


# Plot raw data
plt.figure(figsize=(15, 8))

plt.subplot(4, 1, 1)
plt.plot(afi_eq_time.index, afi_eq_time['Sample'], '.', color='dimgray', markersize=.1)
plt.title('AFI Station')
plt.ylabel('Sample')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 2)
plt.plot(funa_eq_time.index, funa_eq_time['Sample'], '.', color='dimgray', markersize=.1)
plt.title('FUNA Station')
plt.ylabel('Sample')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 3)
plt.plot(tara_eq_time.index, tara_eq_time['Sample'], '.', color='dimgray', markersize=.1)
plt.title('TARA Station')
plt.ylabel('Sample')
plt.xlabel('Time')
plt.gca().set_ylabel('')

plt.subplot(4, 1, 4)
plt.plot(rao_eq_time.index, rao_eq_time['Sample'], '.', color='dimgray', markersize=.1)
plt.title('RAO Station')
plt.ylabel('Sample')
plt.xlabel('Time')
plt.gca().set_ylabel('')

plt.tight_layout()
plt.show()

