import matplotlib.pyplot as plt
import numpy as np

D = np.loadtxt('3.data')

from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

kde = gaussian_kde(D, bw_method='scott')

x_values = np.linspace(min(D), max(D), 1000)
pdf_values = kde(x_values)

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(D.reshape(-1, 1))

means = gmm.means_.flatten()
variances = gmm.covariances_.flatten()
weights = gmm.weights_

gmm_pdf_values = np.zeros_like(x_values)
for weight, mean, var in zip(weights, means, variances):
    gmm_pdf_values += weight * np.exp(-0.5 * ((x_values - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)

plt.figure(figsize=(8, 6))
plt.hist(D, bins=100, density=True, alpha=0.6, color='b', label='Data')
# plt.plot(x_values, pdf_values, color='blue', lw=2, label='Estimated KDE')
plt.plot(x_values, gmm_pdf_values, color='r', lw=2, label='GMM PDF (6 params)')
plt.xlim(0, 20)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.title('Probability Density Function (PDF) Estimation with GMM')

plt.show()
