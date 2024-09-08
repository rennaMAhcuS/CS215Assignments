import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def sample(loc: float, scale: float, size: int = 100000) -> np.ndarray:
    """
    Samples from a Gaussian using `scipy.stats.norm` and `np.random`
    """

    uniform_randoms = np.random.random(size)

    # PPF - Percent Point Function - Inverse of the CDF
    gaussian_randoms = norm.ppf(uniform_randoms, loc=loc, scale=scale)
    return np.array(gaussian_randoms)


# The given params for which sampling nees to be done:
params = [(0, np.sqrt(0.2)),  # (\mu = 0, \sigma^2 = 0.2)
          (0, np.sqrt(1.0)),  # (\mu = 0, \sigma^2 = 1.0)
          (0, np.sqrt(5.0)),  # (\mu = 0, \sigma^2 = 5.0)
          (-2, np.sqrt(0.5))  # (\mu = -2, \sigma^2 = 0.5)
          ]
samples = [sample(loc, scale) for loc, scale in params]

# Params
plt.figure(figsize=(10, 6), dpi=500)
plt.rcParams['font.family'] = 'Cambria'
# Colors
colors = ['blue', 'red', 'yellow', 'green']
labels = [
    r'$\mu=0, \sigma^2=0.2$',
    r'$\mu=0, \sigma^2=1.0$',
    r'$\mu=0, \sigma^2=5.0$',
    r'$\mu=-2, \sigma^2=0.5$'
]

# Plots
for i, (sample_i, color, label) in enumerate(zip(samples, colors, labels)):
    plt.hist(sample_i, bins=500, density=True, alpha=0.6, color=color, label=label)

# Other params
plt.xlim(-5, 5)
plt.xticks(np.arange(-6, 7, 2))
plt.ylim(0, 1)
plt.title('Gaussian Samples with Different Parameters')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()

plt.show()

# plt.savefig('2c.png')
# plt.close()
