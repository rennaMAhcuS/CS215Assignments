import matplotlib.pyplot as plt
import numpy as np


# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data = data

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        u = np.linalg.norm((x - xi) / self.bandwidth, axis=-1)
        return 0.75 * (1 - u ** 2) * (u <= 1)

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        n = len(self.data)
        return np.sum(self.epanechnikov_kernel(x, self.data)) / (n * self.bandwidth)


# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# Initialize the EpanechnikovKDE class
kde = EpanechnikovKDE(bandwidth=1.0)

# Fit the data
kde.fit(data)
# Create a grid of points to evaluate the KDE
x_data = np.linspace(-6, 6, 250)
y_data = np.linspace(-6, 6, 250)
X, Y = np.meshgrid(x_data, y_data)
Z = np.array([kde.evaluate(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
Z = Z.reshape(X.shape)

# Plot the estimated density in a 3D plot
fig = plt.figure(dpi=300)
plt.rcParams['font.family'] = 'Cambria'
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.title('Epanechnikov Kernel Density Estimation')

# Save the plot
plt.savefig('transaction_distribution.png')
plt.close()
