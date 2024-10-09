import matplotlib.pyplot as plt
import numpy as np


# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        # TODO
        self.data = data

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        # TODO
        u = np.linalg.norm((x - xi) / self.bandwidth)
        return 0.75 * (1 - u ** 2) * (np.abs(u) <= 1)

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        # TODO
        n = len(self.data)
        distances = np.linalg.norm((x - self.data) / self.bandwidth, axis=1)
        kernel_values = 0.75 * (1 - distances ** 2) * (np.abs(distances) <= 1)
        return np.sum(kernel_values) / (n * self.bandwidth)


# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
kde = EpanechnikovKDE(bandwidth=1.0)

# TODO: Fit the data
kde.fit(data)
# Create a grid of points to evaluate the KDE
x_data = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
y_data = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
X, Y = np.meshgrid(x_data, y_data)
Z = np.array([kde.evaluate(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
Z = Z.reshape(X.shape)

# TODO: Plot the estimated density in a 3D plot
fig = plt.figure(dpi=300)
plt.rcParams['font.family'] = 'Cambria'
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.title('Epanechnikov Kernel Density Estimation')

# TODO: Save the plot
plt.show()
