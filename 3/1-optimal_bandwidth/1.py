import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./NED21.11.1-D-5.1.1-20111026-v20120814.csv", header = 12, nrows=1500, usecols={"D (Mpc)"})

filtered_df = df.loc[df["D (Mpc)"] < 4]

df_array = np.array([x for x in filtered_df["D (Mpc)"]], dtype=float)

# (a)

nBins = 10

freq, bins = np.histogram(df_array, nBins)
p_j = freq/df_array.size
# print(freq, p_j)

plt.hist(df_array, bins, density=True)
plt.savefig("10binhistogram.png", dpi=300)
plt.close()

# (b)

# The distribution is oversmoothed.

# (c)
df_range = np.max(df_array) - np.min(df_array)
n = df_array.size
nMax = 1000
crossvalidation = np.zeros(nMax)
h = np.zeros(nMax)

for nBins in range(1, nMax+1):
    h[nBins-1] = df_range/nBins
    freq, bins = np.histogram(df_array, nBins)
    crossvalidation[nBins-1] = (2*nBins)/(df_range*(n-1)) - (n+1)*nBins*np.sum(np.square(freq/n))/(df_range*(n-1))

plt.plot(h, crossvalidation)
plt.savefig("crossvalidation.png", dpi=300)
plt.close()


# (d)

# print(np.min(crossvalidation), crossvalidation[49])
# print(h[49])

# The min crossvalidation is reached for nBins = 50
# The corresponding value of h is 0.06835999999999999

# (e)

nBins = 50

freq, bins = np.histogram(df_array, nBins)
p_j = freq/df_array.size
# print(freq, p_j)

plt.hist(df_array, bins, density=True)
plt.savefig("optimalhistogram.png", dpi=300)
plt.close()
