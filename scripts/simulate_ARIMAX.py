import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import scipy.stats as stats
import seaborn as sns

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

def add_earthquakes_random_start(simulated_data, num_earthquakes, scale, duration, seed=123):
    np.random.seed(seed)
    n, m = simulated_data.shape
    earthquake_data = simulated_data.copy()

    for _ in range(num_earthquakes):
        starts = np.random.randint(0, n - duration, size=m)  # Random start for each variable

        for i in range(m):  # For each variable
            start = starts[i]
            peak = start + duration // 2

            for t in range(start, start + duration):
                if t >= n:
                    break
                if t <= peak:
                    factor = (t - start) / (duration / 2)
                else:
                    factor = (start + duration - t) / (duration / 2)

                earthquake_data[t, i] += np.random.randn() * scale * factor

    return earthquake_data


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

def extract_arima_coefficients(y_series):
    model = pm.auto_arima(y_series, seasonal=False, stepwise=False, suppress_warnings=False)
    model_fit = model.fit(y_series)
    return model_fit.params()


def align_columns_to_first(data, max_lag=5000):
    n, d = data.shape
    aligned = np.full((n, d), np.nan)
    aligned[:, 0] = data[:, 0]  # Reference column unchanged

    ref = data[:, 0]

    for i in range(1, d):
        series = data[:, i]
        best_lag = 0
        min_dist = float('inf')

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                ref_trim = ref[:lag]
                s_trim = series[-lag:]
            else:
                ref_trim = ref[lag:]
                s_trim = series[:n - lag]

            if len(ref_trim) == 0:
                continue

            dist = np.linalg.norm(ref_trim - s_trim)
            if dist < min_dist:
                min_dist = dist
                best_lag = lag

        # Apply lag with correct NaN padding
        if best_lag < 0:
            aligned[-best_lag:, i] = data[:n + best_lag, i]
        elif best_lag > 0:
            aligned[:n - best_lag, i] = data[best_lag:, i]
        else:
            aligned[:, i] = data[:, i]

    return aligned


def randomly_shift_columns(data, max_shift, seed=42):
    np.random.seed(seed)
    n, d = data.shape
    shifted = np.full_like(data, np.nan)

    for i in range(d):
        shift = np.random.randint(0, max_shift + 1)
        shifted[shift:, i] = data[:n - shift, i]

    return shifted

# Load data
# eq_raw_afi = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t00_42_55z_AFI.csv")

# Extract ARIMA coefficients and use it for data simmulation

# coefficients = extract_arima_coefficients(eq_raw_afi["Sample"])
# select all exept the last one

#coefficients[-1]
# >>> coefficients
# ar.L1     2.562053e+00
# ar.L2    -2.405983e+00
# ar.L3     1.064306e+00
# ar.L4    -2.215547e-01
# sigma2    1.979285e+06
v = 1.979285e+06

# simulate data
simulated_data = simulate_arima_exog(
    
    c=0, 
    phi=np.array([2.562053, -2.405983, 1.064306, -.2215547]),  # AR coefficients
    theta=np.array([0]),  # MA coefficients
    beta=np.array([0.1, 0.1]),  # Exogenous variable coefficients
    omega_mean=np.array([0, 0, 0]),  # Mean of error terms
    omega_Sigma=np.array([[v, 0.7, 0.7], 
                          [0.7, v, 0.7],
                          [0.7, 0.7, v]]),  # Covariance of error terms
    initial_state=np.array([1, 1, 1, 1]),  # Initial state
    n=10000,  # Number of time steps
    seed=534
)

plot_time_series(simulated_data, titles=["Y", "X1", "X2"])

earthquake_data = add_earthquakes(simulated_data, num_earthquakes=1, scale=2000000000, duration=200)
plot_time_series(earthquake_data, titles=["Y", "X1" , "X2"])


shifted_earthquake_data = randomly_shift_columns(earthquake_data, max_shift=2000)
plot_time_series(shifted_earthquake_data, titles=["Y", "X1" , "X2"])

aligned_eq_data = align_columns_to_first(shifted_earthquake_data)
plot_time_series(aligned_eq_data, titles=["Y", "X1" , "X2"])

sigma2_values = compute_sigma2_values(earthquake_data, window_size=100, overlap_size=10, order=(1,1,1))
plot_time_series(sigma2_values, titles=["Y", "X1" , "X2"])
pd.DataFrame(sigma2_values).to_csv(os.path.join(path, "sigma2_values_sim.csv"))


aligned_eq_data = align_columns_to_first(earthquake_data)
plot_time_series(aligned_eq_data, titles=["Y", "X1" , "X2"])


sigma2_values_aligned = compute_sigma2_values(aligned_eq_data, window_size=100, overlap_size=10, order=(1,1,1))
plot_time_series(sigma2_values_aligned, titles=["Y", "X1" , "X2"])
# plot joint distribution of sigma2 values
x = np.mean(sigma2_values[:, [1, 2]], axis=1)
y = sigma2_values[:, 0]

h = sns.jointplot(x=x, y=y)
h.ax_joint.scatter(x, y, color="black", s=20, alpha=0.6)
h.set_axis_labels("Mean of X1 and x2", "Y")
plt.show()