import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import scipy.stats as stats
import seaborn as sns
from scipy.stats import norm, rankdata, multivariate_normal


path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD"


# Read the sigma2 simmulation CSV file
sigma2_values = pd.read_csv(os.path.join(path, "sigma2_values.csv"), index_col=0)

# --- Step 0: Define the data ---
y = np.array(sigma2_values[:, 0])
x = np.array(sigma2_values[:, 1])
n = len(x)

# step01 Convert to uniform distribution using empirical CDF
U_x = stats.rankdata(x) / (len(sigma2_values) + 1)
U_y = stats.rankdata(y) / (len(sigma2_values) + 1)

# --- Step 2: Transform the uniform marginals to standard normal quantiles ---
z_x = norm.ppf(U_x)
z_y = norm.ppf(U_y)


# --- Step 3: Estimate the dependence (correlation) using the transformed data ---
rho_est = np.corrcoef(z_x, z_y)[0, 1]


# --- Step 4: Compute the joint CDF using the Gaussian copula ---
# The Gaussian copula joint CDF is given by:
#   F(x, y) = Phi_rho(z_x, z_y)
# where Phi_rho is the bivariate normal CDF with correlation rho.
def gaussian_copula_cdf(z1, z2, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return multivariate_normal(mean=mean, cov=cov).cdf([z1, z2])

# Compute the joint CDF for each (x,y) pair.
joint_cdf_values = [gaussian_copula_cdf(z_x[i], z_y[i], rho_est) for i in range(n)]


# Set alpha and calculate threshold (1 - alpha)
joint_cdf_values = np.array(joint_cdf_values)
alpha = 0.05
threshold = 1 - alpha  

# Identify indices where joint_cdf_values is greater than the threshold
flag_indices = np.where(joint_cdf_values > threshold)[0]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(z_x, z_y, c=joint_cdf_values, cmap="coolwarm", edgecolors='k', alpha=0.75)

# Flag the points using a red star marker
plt.scatter(z_x[flag_indices], z_y[flag_indices], marker='*', color='red', s=100, label=f'joint_cdf > {threshold:.2f}')

# Optionally, annotate the flagged points with their joint CDF values
for idx in flag_indices:
    plt.annotate(f"{1 - joint_cdf_values[idx]:.4f}",
                 (z_x[idx], z_y[idx]),
                 textcoords="offset points",
                 xytext=(5, 5),
                 ha='center',
                 fontsize=9,
                 color='red')

plt.legend()
plt.show()

