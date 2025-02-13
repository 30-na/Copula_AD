import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import scipy.stats as stats
import seaborn as sns

path = r"C:\Users\sina.mokhtar\Documents\problems\Copula_AD"


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

# Load data
eq_raw_afi = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t00_42_55z_AFI.csv")

# Extract ARIMA coefficients
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
    beta=np.array([0.1]),  # Exogenous variable coefficients
    omega_mean=np.array([0, 0]),  # Mean of error terms
    omega_Sigma=np.array([[v, 0.7], 
                          [0.7, v]]),  # Covariance of error terms
    initial_state=np.array([1, 1, 1, 1]),  # Initial state
    n=1000000,  # Number of time steps
    seed=534
)

plot_time_series(simulated_data, titles=["Y", "X1"])

earthquake_data = add_earthquakes(simulated_data, num_earthquakes=3, scale=20000, duration=200)
plot_time_series(earthquake_data, titles=["Y", "X1" ])

 
sigma2_values = compute_sigma2_values(earthquake_data, window_size=500, overlap_size=100, order=(1,1,1))
plot_time_series(sigma2_values, titles=["Y", "X1"])



def plot_anomalies_histogram_chi2(sigma2_values):
    # Compute 95% confidence intervals using Chi-Square distribution
    # Plot histograms for each column
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    for i, ax in enumerate(axes):
        ax.hist(sigma2_values[:, i], bins=300, alpha=0.7, color='blue', label=f'Column {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_xlim(0, 2e8)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_anomalies_histogram_chi2(sigma2_values)

h = sns.jointplot(x=sigma2_values[:, 1], y=sigma2_values[:, 0], kind='kde')
h.set_axis_labels("X1", "Y")
plt.show()

# Convert to uniform distribution using empirical CDF
U_x = stats.rankdata(sigma2_values[:, 0]) / (len(sigma2_values) + 1)
U_y = stats.rankdata(sigma2_values[:, 1]) / (len(sigma2_values) + 1)

Z_x = stats.norm.ppf(U_x)
Z_y = stats.norm.ppf(U_y)

# Compute empirical correlation (copula correlation)
copula_corr = np.corrcoef(Z_x, Z_y)[0, 1]

# Step 3: Define Gaussian Copula Function
def gaussian_copula_pdf(U_x, U_y, rho):
    """Compute Gaussian copula density."""
    norm_x = stats.norm.ppf(U_x)
    norm_y = stats.norm.ppf(U_y)
    joint_pdf = stats.multivariate_normal.pdf(
        np.column_stack((norm_x, norm_y)),
        mean=[0, 0],
        cov=[[1, rho], [rho, 1]]
    )
    marginal_pdf_x = stats.norm.pdf(norm_x)
    marginal_pdf_y = stats.norm.pdf(norm_y)
    return joint_pdf / (marginal_pdf_x * marginal_pdf_y)

# Compute Gaussian copula density
copula_density = gaussian_copula_pdf(U_x, U_y, copula_corr)

# Step 4: Visualize Copula Density
plt.figure(figsize=(8, 6))
scatter = plt.scatter(U_x, U_y, c=copula_density, cmap="coolwarm", edgecolors='k', alpha=0.75)
plt.colorbar(scatter, label="Copula Density")
plt.xlabel("U_x (Uniform Transformed)")
plt.ylabel("U_y (Uniform Transformed)")
plt.title("Gaussian Copula Density")
plt.show()

import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Convert to Uniform Marginals using Empirical CDF
U_x = stats.rankdata(sigma2_values[:, 0]) / (len(sigma2_values) + 1)
U_y = stats.rankdata(sigma2_values[:, 1]) / (len(sigma2_values) + 1)

# Step 2: Estimate Gumbel Copula Parameter (Theta)
tau, _ = stats.kendalltau(U_x, U_y)  # Estimate Kendall's Tau
theta = 1 / (1 - tau)  # Gumbel parameter (theta >= 1)

# Step 3: Define Gumbel Copula CDF
def gumbel_copula_cdf(U_x, U_y, theta):
    """Compute the Gumbel Copula CDF."""
    U_x = np.clip(U_x, 1e-10, 1 - 1e-10)  # Avoid log(0)
    U_y = np.clip(U_y, 1e-10, 1 - 1e-10)
    W = (-np.log(U_x))**theta + (-np.log(U_y))**theta
    return np.exp(-W**(1/theta))

# Compute Gumbel Copula Values
copula_cdf_values = gumbel_copula_cdf(U_x, U_y, theta)

# Step 4: Visualize the Gumbel Copula
plt.figure(figsize=(8, 6))
scatter = plt.scatter(U_x, U_y, c=copula_cdf_values, cmap="coolwarm", edgecolors='k', alpha=0.75)
plt.colorbar(scatter, label="Gumbel Copula CDF")
plt.xlabel("U_x (Uniform Transformed)")
plt.ylabel("U_y (Uniform Transformed)")
plt.title("Gumbel Copula CDF (Upper Tail Dependence)")
plt.show()


# Convert data into a DataFrame for ARIMAX fitting
df = pd.DataFrame(simulated_data, columns=['y'] + [f'X{i+1}' for i in range(m - 1)])
exog_vars = df.iloc[:, 1:]  # Select exogenous variables (all except y)
model = sm.tsa.ARIMA(df['y'], order=(p, 0, q), exog=exog_vars)
fitted_model = model.fit()
fitted_model.summary()
df['y_fitted'] = fitted_model.fittedvalues



# Function to generate correlated p-values using a Gaussian copula
def generate_correlated_pvalues(n, rho):
    """Generate n sets of correlated p-values with correlation rho."""
    mean = [0, 0]  # Mean vector for multivariate normal
    cov = [[1, rho],  # Covariance matrix (controls dependence)
           [rho, 1]]
    
    
    mvn_samples = np.random.multivariate_normal(mean, cov, size=n)
    
    # Convert normal samples to uniform (via CDF)
    uniform_samples = stats.norm.cdf(mvn_samples)
    
    return uniform_samples  

# Function to compute combined p-value using Gaussian copula
def copula_combined_pvalue(p_values):
    """Compute the combined p-value using a Gaussian copula."""
    k = len(p_values)  # Number of p-values
    empirical_cdf = np.mean(np.all(p_values >= p_values, axis=1))  # Empirical copula
    return 1 - empirical_cdf

# Generate 5 sets of correlated p-values with ρ=0.5
np.random.seed(42)
p_values_sample = generate_correlated_pvalues(n=5, rho=0.5)

# Compute combined p-value for each time step
combined_p_values = np.array([copula_combined_pvalue(p) for p in p_values_sample])

# Set anomaly threshold
alpha = 0.05  # Significance level

# Identify anomalies
anomalies = combined_p_values < alpha

# Display results
import pandas as pd
import ace_tools as tools

df = pd.DataFrame({
    "p1": p_values_sample[:, 0],
    "p2": p_values_sample[:, 1],
    "p3": p_values_sample[:, 2],
    "Combined p-value": combined_p_values,
    "Anomaly?": anomalies
})

tools.display_dataframe_to_user(name="Copula-Based p-Value Pooling", dataframe=df)
