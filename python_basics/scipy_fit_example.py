import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define a Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

# Generate synthetic data with noise
np.random.seed(42)  # For reproducibility
x = np.linspace(-5, 5, 100)
y = gaussian(x, amplitude=1.0, mean=0.0, stddev=1.0)
noise = 0.1 * np.random.normal(size=x.size)
y_noisy = y + noise

# Fit the noisy data using curve_fit
initial_guess = [1, 0, 1]  # Initial guesses for amplitude, mean, stddev
params, covariance = curve_fit(gaussian, x, y_noisy, p0=initial_guess)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

# Extract fitted parameters
amplitude_fit, mean_fit, stddev_fit = params

print("True parameters: amplitude=1.0, mean=0.0, stddev=1.0")
print("Fitted parameters:")
print(f"  Amplitude: {amplitude_fit:.2f}")
print(f"  Mean: {mean_fit:.2f}")
print(f"  Stddev: {stddev_fit:.2f}")

# Plot the data and the fitted function
plt.figure()
plt.scatter(x, y_noisy, label='Noisy Data', color='blue')
plt.plot(x, y, label='True Gaussian', color='green')
plt.plot(x, gaussian(x, *params), label='Fitted Gaussian', color='red')
plt.title("Gaussian Fit Example")
plt.legend()
plt.show()