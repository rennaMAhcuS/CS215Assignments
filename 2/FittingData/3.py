import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import scipy.stats


# Reading data
f = open("./3.data", "r")
D = np.loadtxt(f)
# print(D, D.dtype, D.size)

# Task A

# First moment of D
m1 = np.sum(D)/D.size
# print(m1)

# Second moment of D
m2 = np.sum(np.square(D))/D.size
# print(m2)

# Task B

plt.hist(D, 200, density=True)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.savefig("3b.png", dpi=300)
plt.close()

# Guess of mu: ~6

# Task C

def funcC(x):
    return [x[0]*x[1]-m1, x[0]*x[1]+x[0]*(x[0]-1)*x[1]*x[1]-m2]

est = np.array([1, 0])

root = scipy.optimize.fsolve(funcC, est)
n = round(root[0])
p = root[1]
# print(root, n, p)

var = np.linspace(0, 20, 21)
bin = scipy.stats.binom.pmf(var, n, p)
plt.plot(bin, color=(0.8, 0.2, 0))
plt.hist(D, 200, density=True, histtype="bar", color=(0, 0.3, 0.8))
plt.xlabel("x")
plt.ylabel("p(x)")
plt.savefig("3c.png", dpi=300)
plt.close()


# Task D

def funcD (x):
    return [x[0]*x[1]-m1, x[0]*(x[0]+1)*x[1]*x[1]-m2]

est = np.array([1, 0])

root = scipy.optimize.fsolve(funcD, est)
k = root[0]
theta = root[1]
# print(root, k, theta)

var = np.linspace(0, 20, 200)
print(var)
bin = scipy.stats.gamma.pdf(var, k, 0, theta)
plt.plot(bin, color=(0.8, 0.2, 0))
plt.hist(D, 100, density=True, histtype="bar", color=(0, 0.3, 0.8))
plt.xlabel("x")
plt.ylabel("p(x)")
plt.savefig("3d.png", dpi=300)

