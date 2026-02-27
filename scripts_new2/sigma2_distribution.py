from EDA.my_functions import *
from scipy.stats import chi2

######################### 
# Time Series Data Simulation
# Given coefficients
np.random.seed(576)
c = 0
phi1 = 0.4
phi2 = 0.6
theta1 = 0.7
theta2 = 0.3
series_length = 100000
x0 = 0
x1 = 0
sigma2 = 1
noise = np.random.normal(0, sigma2, series_length)
ts_simulated_data = np.zeros(series_length)
ts_simulated_data[0] = x0
ts_simulated_data[1] = x1

# Generate simulated data
for t in range(2, series_length):
    ts_simulated_data[t] =  phi1 * ts_simulated_data[t-1] + phi2 * ts_simulated_data[t-2] + noise[t] + theta1 * noise[t-1] + theta2 * noise[t-2]

plt.figure()
plt.plot(ts_simulated_data)
plt.show()
len(ts_simulated_data)

#################

def fit_arima(group):
    #auto_model = pm.auto_arima(group, seasonal=True, trace=False, error_action='ignore', suppress_warnings=True, stepwise=False, max_p=2, max_q=2)
    #order = auto_model.order
    order = (2,1,2)
    model = sm.tsa.ARIMA(group, order=order)
    fit = model.fit()
    params = fit.params
    return params


N = 200
simulated_data_series = pd.Series(ts_simulated_data)
start_indices = range(0, len(simulated_data_series) - N + 1, N // 2)
windows = [simulated_data_series.iloc[i:i + N] for i in start_indices]
windows_df = pd.DataFrame({'window': windows})

def extract_sigma2(sample):
    try:
        params = fit_arima(sample)
        return params[-1]  # Assuming sigma2 is the last parameter
    except np.linalg.LinAlgError:
        print("Skipping a window due to LU decomposition error.")
        return None  # Returns None to indicate no value


# Use apply to apply the function to each window
ts_bootstrap_samples = []
windows_df['sigma2'] = windows_df['window'].apply(extract_sigma2)

ts_bootstrap_samples = windows_df['sigma2'].tolist()
ts_bootstrap_samples = [x for x in ts_bootstrap_samples if np.isfinite(x)]

# Somthing is wrong in the code and mabye the python function
ts_bootstrap_samples = [x for x in ts_bootstrap_samples if x <= 10]
np.sum(~np.isfinite(ts_bootstrap_samples))
np.sum(ts_bootstrap_samples == 0)

plt.figure()
plt.boxplot(ts_bootstrap_samples, vert=True, patch_artist=True)
plt.title('Box Plot of Bootstrap Samples')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.show()

sigma2_est_mu = np.mean(ts_bootstrap_samples)
sigma2_est_std = np.std(ts_bootstrap_samples)


################

mean = 0
sigma2 = 1
sample_size = 200
df = sample_size - 1
num_samples = len(ts_bootstrap_samples)

# Step 1: Simulate data
data = np.random.normal(mean, np.sqrt(sigma2), (num_samples, sample_size))
variances = np.var(data, axis=1, ddof=1)
y_lim = 5
# Create a figure with three subplots
plt.figure(figsize=(10, 16))

# Plot A: Histogram of Sample Variances
plt.subplot(3, 1, 1)
plt.hist(variances, bins=50, density=True, alpha=0.6, color='b', label='Sample Variances')
x = np.linspace(0, 5, 100)
chi2_pdf = stats.chi2.pdf(x * (sample_size - 1), df) * (df / sigma2)
plt.plot(x, chi2_pdf, 'r-', label='Chi-square PDF (df=199)')
plt.title('Histogram of Sample Variances and Chi-square PDF')
plt.xlabel('')
plt.ylabel('Density')
plt.xlim(0, 5)  # Set x-axis limits
plt.ylim(0, y_lim)  # Set y-axis limits
plt.legend()
plt.grid()

# Plot B: Histogram of estimated sigma2 from ts_bootstrap_samples
plt.subplot(3, 1, 2)
plt.hist(ts_bootstrap_samples, bins=50, density=True, alpha=0.6, color='b', label='Sample Variances')
plt.plot(x, chi2_pdf, 'r-', label='Chi-square PDF (df=199)')
plt.title('Histogram of estimated sigma2 and Chi-square PDF')
plt.xlabel('')
plt.ylabel('Density')
plt.xlim(0, 5)  # Set x-axis limits
plt.ylim(0, y_lim)  # Set y-axis limits
plt.legend()
plt.grid()

# Plot C: Histogram using sigma2 mean instead of true variance
sigma2_est_mu = np.mean(ts_bootstrap_samples)
plt.subplot(3, 1, 3)
plt.hist(ts_bootstrap_samples, bins=50, density=True, alpha=0.6, color='b', label='Sample Variances')
chi2_pdf = stats.chi2.pdf(x * (sample_size - 1), df) * (df / sigma2_est_mu)
plt.plot(x, chi2_pdf, 'r-', label='Chi-square PDF (df=199)')
plt.title('Histogram of estimated sigma2 and Chi-square PDF (Scaled with mean of sigma2)')
plt.xlabel('')
plt.ylabel('Density')
plt.xlim(0, 5)  # Set x-axis limits
plt.ylim(0, y_lim)  # Set y-axis limits
plt.legend()
plt.grid()

# Adjust layout
plt.tight_layout()
plt.show()
















#######################################
# Number of bootstrap samples
M = 1000
N = 10
bootstrap_samples = []

# Randomly choose a start index
for _ in range(M):
    start_idx = np.random.randint(0, series_length-N-1)
    end_idx = start_idx + N
    sample = simulated_data[start_idx:end_idx]
    params = fit_arima(sample)
    sigma2 = params[-1]
    bootstrap_samples.append(sigma2)

sigma2_est_mu = np.mean(bootstrap_samples)
sigma2_est_std = np.std(bootstrap_samples)





np.argmax(bootstrap_samples)


plt.figure()
plt.hist(bootstrap_samples, bins=30, density=True, alpha=0.6, color='b', label='bootstrap Variances',edgecolor='black')
df = N - 1 
x = np.linspace(min(bootstrap_samples), max(bootstrap_samples), 1000)
chi2_density = chi2.pdf(x * (N - 1) / 1, df) * (N - 1) / 1
plt.plot(x, chi2_density, 'r-', lw=2, label=f"chi2 (df={N-1})")
plt.xlabel('Sample Variance')
plt.ylabel('Density')
plt.title('Distribution of Sample Variances with Chi-Square Density')
plt.legend()
plt.show()










pd.Series(bootstrap_samples).describe()




plt.figure()
#plt.hist(bootstrap_samples, bins=30, edgecolor='black')
x = np.linspace(min(bootstrap_samples), max(bootstrap_samples), 1000)
y = chi2.pdf(x, df=20)

# Plot the chi-square distribution curve
plt.plot(x, y, color='red', linewidth=2)

plt.show()
plt.show()


mean = 50
variance = 100
std_dev = np.sqrt(variance)

# Generate x values for the normal curve
x = np.linspace(min(chi), max(chi), 1000)
y = norm.pdf(x, mean, std_dev)

# Plot the normal distribution curve
plt.plot(x, y, color='red', linewidth=2)
plt.show()
path = os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "sigma_dist.png") 
plt.savefig(path)
plt.close()



num_simulations = 100
res_series = []
for _ in range(num_simulations):
    # Initialize array to store simulated values
    simulated_data = np.zeros(series_length)
    simulated_data[0] = x0
    simulated_data[1] = x1
    # Generate white noise
    noise = np.random.normal(0, sigma2, series_length)
    
    # Generate simulated data
    for t in range(2, series_length):
        simulated_data[t] =  phi1 * simulated_data[t-1] + phi2 * simulated_data[t-2] + noise[t] + theta1 * noise[t-1] + theta2 * noise[t-2]
    
    simulated_series.append(simulated_data)
    res_series.append(res)

# Calculate residuals

X = np.mean(simulated_series, axis=0)
R = np.mean(res_series, axis=0)
