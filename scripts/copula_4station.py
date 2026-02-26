import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm, multivariate_normal
from scipy.stats import rankdata, t, multivariate_t
from copulae.archimedean import GumbelCopula
import matplotlib.pyplot as plt
from tqdm import tqdm




def compute_joint_cdf_gaussian(df: pd.DataFrame) -> pd.Series:
    n = len(df)

    # Step 1: Convert to uniform marginals using empirical CDF
    U = np.array([rankdata(df[col]) / (n + 1) for col in df.columns]).T

    # Step 2: Transform to standard normal space
    Z = norm.ppf(U)

    # Step 3: Estimate correlation matrix
    corr_matrix = np.corrcoef(Z, rowvar=False)

    # Step 4: Compute Gaussian copula joint CDF with progress bar
    copula = multivariate_normal(mean=np.zeros(Z.shape[1]), cov=corr_matrix)
    joint_cdf_vals = np.array([copula.cdf(z) for z in tqdm(Z, desc="Computing joint CDF")])

    return pd.Series(joint_cdf_vals, index=df.index, name="joint_cdf")

def joint_upper_all_gaussian(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    U = np.array([rankdata(df[c])/(n+1) for c in df.columns]).T
    Z = norm.ppf(U)
    corr = np.corrcoef(Z, rowvar=False)
    cop = multivariate_normal(mean=np.zeros(Z.shape[1]), cov=corr)
    vals = np.array([cop.cdf(-z) for z in Z])
    return pd.Series(vals, index=df.index, name="joint_upper_all")


p_all  = joint_upper_all_gaussian(sigma2_df)

def compute_joint_cdf_tcopula(df: pd.DataFrame, df_t: int = 4) -> pd.Series:
    n = len(df)

    # Step 1: Convert each column to uniform [0,1] using empirical CDF
    U = np.array([rankdata(df[col]) / (n + 1) for col in df.columns]).T

    # Step 2: Transform uniform to t-space (inverse CDF of t)
    T = t.ppf(U, df_t)

    # Step 3: Estimate correlation matrix in t-space
    corr_matrix = np.corrcoef(T, rowvar=False)

    # Step 4: Use multivariate t-distribution for joint CDF
    copula = multivariate_t(loc=np.zeros(T.shape[1]), shape=corr_matrix, df=df_t)
    joint_cdf_vals = np.array([copula.cdf(t_row) for t_row in tqdm(T, desc="Computing t-copula joint CDF")])

    return pd.Series(joint_cdf_vals, index=df.index, name="joint_cdf_t")


def compute_joint_cdf_gumbel_multivariate(df: pd.DataFrame) -> pd.Series:
    n, d = df.shape
    U = np.vstack([rankdata(df[col]) / (n + 1) for col in df.columns]).T
    cop = GumbelCopula(dim=d)
    cop.fit(U)
    cdf_vals = cop.cdf(U)
    return pd.Series(cdf_vals, index=df.index, name="joint_cdf_gumbel")

afi_raw = pd.read_csv("processedData/afi.csv", index_col=0, parse_dates=True).resample("1s").mean()
funa_aligned = pd.read_csv("processedData/funa_aligned_5days.csv", index_col=0, parse_dates=True).resample("1s").mean()
tara_aligned = pd.read_csv("processedData/tara_aligned_5days.csv", index_col=0, parse_dates=True).resample("1s").mean()
rao_aligned  = pd.read_csv("processedData/rao_aligned_5days.csv", index_col=0, parse_dates=True).resample("1s").mean()

nan_count = df.isna().sum()
inf_count = np.isinf(df).sum()


df = pd.concat([afi_raw, funa_aligned, tara_aligned, rao_aligned], axis=1, join='inner')
df.columns = ['AFI', 'FUNA', 'TARA', 'RAO']

df = pd.read_csv("processedData/sigma2_5d_121.csv", index_col=0 parse_dates=True)
joint_cdf_series = compute_joint_cdf(df)


# Set horizontal threshold value (adjust as needed)
threshold = 0.95

# Identify points above threshold
above_threshold = joint_cdf_series > threshold

plt.figure(figsize=(12, 4))

# Plot normal points
plt.plot(joint_cdf_series.index[~above_threshold], joint_cdf_series.values[~above_threshold],
         '.', color='dimgray', markersize=0.2)

# Plot anomalous points in red
plt.plot(joint_cdf_series.index[above_threshold], joint_cdf_series.values[above_threshold],
         '.', color='red', markersize=0.2)

# Horizontal dashed line at threshold
plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1)

# X-axis formatting
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.title("Joint CDF Time Series from Gaussian Copula")
plt.xlabel("Time")
plt.ylabel("Joint CDF")
plt.grid(True)
plt.tight_layout()
plt.show()











## Copula on raw data
df = pd.concat([afi_raw, funa_aligned, tara_aligned, rao_aligned], axis=1, join='inner')
df.columns = ['AFI', 'FUNA', 'TARA', 'RAO']

joint_cdf_series = compute_joint_cdf(df)



# Load sigma2 data
afi = pd.read_csv("processedData/afi_sigma2_5.csv", index_col=0, parse_dates=True)
funa = pd.read_csv("processedData/funa_sigma2_5d.csv", index_col=0, parse_dates=True)
tara = pd.read_csv("processedData/tara_sigma2_5d.csv", index_col=0, parse_dates=True)
rao = pd.read_csv("processedData/rao_sigma2_5d.csv", index_col=0, parse_dates=True)

len(afi)
# Combine all by inner join on timestamp
afi.columns = ['afi']
funa.columns = ['funa']
tara.columns = ['tara']
rao.columns = ['rao']

df = pd.concat([afi, funa, tara, rao], axis=1, join='inner')



df = pd.read_csv("processedData/sigma2_5d.csv", index_col=0, parse_dates=True).drop(columns='RAO')





n = len(df)
# Step 1: Convert to uniform marginals using empirical CDF
U = np.array([rankdata(df[col]) / (n + 1) for col in df.columns]).T  # shape (n, 4)


# Step 2: Transform to standard normal space
Z = norm.ppf(U)  # shape (n, 4)


# Step 3: Estimate correlation matrix
corr_matrix = np.corrcoef(Z, rowvar=False)


# Step 4: Compute Gaussian copula joint CDF
copula = multivariate_normal(mean=np.zeros(4), cov=corr_matrix)
joint_cdf_vals = np.array([copula.cdf(z) for z in Z])


# Step 5: Flag anomalies
alpha = 0.05
threshold = 1 - alpha
flag_indices = np.where(joint_cdf_vals > threshold)[0]


# Create time series of joint CDF values
joint_cdf_series = pd.Series(joint_cdf_vals, index=df.index, name="joint_cdf")

len(joint_cdf_series)
# Plot time series
plt.figure(figsize=(12, 4))
plt.plot(joint_cdf_series.index, joint_cdf_series.values, '.', color='dimgray', linewidth=1)
plt.scatter(joint_cdf_series.index[flag_indices], joint_cdf_series.values[flag_indices], 
            color='red', marker='*', s=100, label=f'Anomalies (CDF > {threshold:.2f})')

plt.title("Joint CDF Time Series from Gaussian Copula")
plt.xlabel("Time")
plt.ylabel("Joint CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


df_combined = pd.concat([df, joint_cdf_series], axis=1)
df_combined.index = pd.to_datetime(df_combined.index)

import matplotlib.pyplot as plt

df_combined['joint_cdf'].plot()
plt.xlabel('Time')
plt.ylabel('joint_cdf')
plt.title('Joint CDF over Time')
plt.show()



# Plot
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

cols = ['AFI', 'FUNA', 'TARA',  'joint_cdf']

# Plot with small gray points
for i, col in enumerate(cols):
    axes[i].scatter(df_combined.index, df_combined[col], color='#555555', s=6)
    axes[i].set_ylabel(col)
    axes[i].set_xlim(df_combined.index.min(), df_combined.index.max())
    axes[i].set_xticks(df_combined.index[::max(1, len(df_combined)//6)])
    axes[i].tick_params(axis='y', labelsize=9)

axes[-1].set_ylim(0,1.2)
axes[-1].axhline(y=0.95, color='gray', linestyle='--', linewidth=1)
axes[-1].set_xlabel('Time')

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()

################################# T copula
import openturns as ot
import pandas as pd
import numpy as np

# Suppose `df` is your aligned sigma² DataFrame
data = df.to_numpy()
n = data.shape[0]

# Step 1: Convert to uniform marginals using ranks
uniform_data = np.array([
    pd.Series(col).rank(method='average') / (n + 1)
    for col in data.T
]).T

# Step 2: Create OpenTURNS sample
sample = ot.Sample(uniform_data.tolist())

# Step 3: Fit Student (t) copula
student_copula = ot.StudentCopulaFactory().build(sample)

# Step 4: Compute joint CDF
joint_cdf_vals = [student_copula.computeCDF(row) for row in sample]
joint_cdf_series = pd.Series(joint_cdf_vals, index=df.index)

# Step 5: Plot and flag anomalies
alpha = 0.05
threshold = 1 - alpha
flagged = joint_cdf_series[joint_cdf_series > threshold]

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(joint_cdf_series.index, joint_cdf_series, '-', color='dimgray', label='Joint CDF')
plt.scatter(flagged.index, flagged.values, color='red', marker='*', s=100, label='Anomalies')
plt.title("Anomaly Detection using t-Copula (OpenTURNS)")
plt.xlabel("Time")
plt.ylabel("Joint CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()






from copulas.multivariate import GaussianMultivariate
# Fit to your data (assumes DataFrame with numerical columns)
model = GaussianMultivariate()
model.fit(df)  # df is your DataFrame of aligned sigma² values

# Get CDF values
cdf_vals = model.cumulative_distribution(df)
# Plot time series
plt.figure(figsize=(12, 4))
plt.plot(joint_cdf_series.index, cdf_vals, '.', color='dimgray', linewidth=1)
plt.scatter(joint_cdf_series.index[flag_indices], joint_cdf_series.values[flag_indices], 
            color='red', marker='*', s=100, label=f'Anomalies (CDF > {threshold:.2f})')

plt.title("Joint CDF Time Series from Gaussian Copula")
plt.xlabel("Time")
plt.ylabel("Joint CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





import openturns as ot
import pandas as pd
import numpy as np

# Load your multivariate Sigma² data (already aligned)
df = pd.concat([afi, funa, tara, rao], axis=1, join='inner')
df.columns = ['afi', 'funa', 'tara', 'rao']
data = df.dropna().to_numpy()
n = data.shape[0]

# Step 1: Convert to empirical marginals using ranks
uniform_data = np.array([
    (pd.Series(col).rank(method='average') / (n + 1)).values
    for col in data.T
]).T

# Step 2: Create OpenTURNS sample
sample = ot.Sample(uniform_data.tolist())

# Step 3: Fit Gaussian copula
copula = ot.NormalCopulaFactory().build(sample)

# Step 4: Compute joint CDF
joint_cdf_vals = [copula.computeCDF(row) for row in sample]

# Step 5: Create time series and flag anomalies
joint_cdf_series = pd.Series(joint_cdf_vals, index=df.index[:n], name="joint_cdf")

alpha = 0.05
threshold = 1 - alpha
flagged_times = joint_cdf_series[joint_cdf_series > threshold].index

# Optional: Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(joint_cdf_series.index, joint_cdf_series, '-', color='dimgray')
plt.scatter(flagged_times, joint_cdf_series[flagged_times], color='red', marker='*', s=100)
plt.title("Joint CDF Time Series via OpenTURNS Gaussian Copula")
plt.xlabel("Time")
plt.ylabel("Joint CDF")
plt.grid(True)
plt.tight_layout()
plt.show()
