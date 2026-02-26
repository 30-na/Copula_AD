import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, weibull_min

# Step 1: Simulate independent X and Y
np.random.seed(42)
n = 1000

# X ~ Exponential(1)
x = expon.rvs(scale=1, size=n)

# Y ~ Weibull(2, 1)
y = weibull_min.rvs(c=2, scale=1, size=n)

# Print first 5 values
print("First 5 values of X (Exponential):", x[:5])
print("First 5 values of Y (Weibull):", y[:5])

# Plot X vs Y
plt.figure(figsize=(6, 4))
plt.scatter(x, y, alpha=0.5)
plt.title("Step 1: Independent X ~ Exponential and Y ~ Weibull")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate Z ~ N(0, 1)
z = np.random.normal(0, 1, 100000)
u = norm.cdf(z)  # U = Φ(Z)

# Plot 1: Histogram of Z (Normal)
plt.figure(figsize=(6, 4))
plt.hist(z, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Histogram of Z ~ N(0, 1)")
plt.xlabel("z")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# Plot 2: Histogram of U = Φ(Z) (Uniform)
plt.figure(figsize=(6, 4))
plt.hist(u, bins=50, density=True, alpha=0.7, color='salmon', edgecolor='black')
plt.title("Histogram of U = Φ(Z): Should Be Uniform")
plt.xlabel("u = Φ(z)")
plt.ylabel("Density")
plt.grid(True)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

z = np.random.normal(0, 1, 100000)
u = norm.cdf(z)

plt.hist(u, bins=50, density=True)
plt.title("Histogram of Φ(Z): Should Look Uniform")
plt.xlabel("u = Φ(z)")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# Plot z1 vs z2 (standard normal)
plt.figure(figsize=(6, 4))
plt.scatter(z1, z2, alpha=0.5)
plt.title("Standard Normals z1 vs z2")
plt.xlabel("z1")
plt.ylabel("z2")
plt.grid(True)
plt.show()

# Step 2.1: Transform to uniforms
u = norm.cdf(z1)
v = norm.cdf(z2)

# Plot u vs v (uniforms)
plt.figure(figsize=(6, 4))
plt.scatter(u, v, alpha=0.5)
plt.title("Uniform Variables u = Φ(z1), v = Φ(z2)")
plt.xlabel("u")
plt.ylabel("v")
plt.grid(True)
plt.show()

# Step 2.2: Map to marginals
x_dep = expon.ppf(u)
y_dep = weibull_min.ppf(v, c=2)

# Plot dependent marginals
plt.figure(figsize=(6, 4))
plt.scatter(x_dep, y_dep, alpha=0.5)
plt.title("Dependent X ~ Exponential, Y ~ Weibull")
plt.xlabel("X (Exponential)")
plt.ylabel("Y (Weibull)")
plt.grid(True)
plt.show()


