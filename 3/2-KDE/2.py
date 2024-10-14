import matplotlib.pyplot as plt
import numpy as np


# Custom 2D Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        """Initialize with given bandwidth."""
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data_to_fit):
        """Fit the KDE model with the provided data."""
        self.data = data_to_fit

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function for 2D using vectorized operations."""
        norm_squared = np.sum(((xi - x) / self.bandwidth) ** 2, axis=-1)
        return ((2 / np.pi) * (1 - norm_squared)) * (norm_squared <= 1)

    def evaluate(self, x):
        """Evaluate the KDE at multiple points x in 2D."""
        return self.epanechnikov_kernel(x, self.data).mean() / (self.bandwidth ** 2)


# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# Initialize the EpanechnikovKDE class with a given bandwidth
kde = EpanechnikovKDE(bandwidth=1.5)

# Fit the data to the KDE model
kde.fit(data)

# Define the range of the grid for plotting the density and evaluate the KDE for each point in the grid
x_range = np.linspace(min(data[:, 0]) - 1, max(data[:, 0]) + 1, 100)
y_range = np.linspace(min(data[:, 1]) - 1, max(data[:, 1]) + 1, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.array([kde.evaluate(np.array([xi, yi])) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

# Plot the estimated density in a 3D plot
fig = plt.figure(dpi=1000)
plt.rcParams['font.family'] = 'Cambria'
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.title('Epanechnikov Kernel Density Estimation')

# Save the plot
plt.savefig('transaction_distribution.png')
plt.close()
