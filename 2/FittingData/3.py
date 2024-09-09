import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import scipy.stats

#
# # Reading data
f = open("3.data", "r")
D = np.loadtxt(f)
# print(D, D.dtype, D.size)

#
# # Task A
#
# # First moment of D
m1 = np.sum(D)/D.size
# # print(m1)
#
# # Second moment of D
m2 = np.sum(np.square(D))/D.size
# # print(m2)
#
#
# # Task B
#
# # plt.hist(D, 200, density=True)
# # plt.xlabel("x")
# # plt.ylabel("p(x)")
# # plt.savefig("3b.png", dpi=300)
# # plt.close()
#
# # Guess of mu: ~7
#
#
# # Task C
#
# # def funcC(x):
# #     return [x[0]*x[1]-m1, x[0]*x[1]+x[0]*(x[0]-1)*x[1]*x[1]-m2]
#
# # est = np.array([1, 0])
#
# # root = scipy.optimize.fsolve(funcC, est)
# # n = round(root[0])
# # p = root[1]
# # # print(root, n, p)
#
# # var = np.linspace(0, 20, 21)
# # bin = scipy.stats.binom.pmf(var, n, p)
# # plt.plot(var, bin, color=(0.8, 0.2, 0))
# # plt.hist(D, 200, density=True, histtype="bar", color=(0, 0.3, 0.8))
# # plt.xlabel("x")
# # plt.ylabel("p(x)")
# # plt.savefig("3c.png", dpi=300)
# # plt.close()
#
#
# # Task D
#
# # def funcD (x):
# #     return [x[0]*x[1]-m1, x[0]*(x[0]+1)*x[1]*x[1]-m2]
#
# # est = np.array([1, 0])
#
# # root = scipy.optimize.fsolve(funcD, est)
# # k = root[0]
# # theta = root[1]
# # print(root, k, theta)
#
# # var_freq = 200
# # var = np.linspace(0, 20, var_freq + 1)
# # gamma = scipy.stats.gamma.pdf(var, k, 0, theta)
# # # gamma_x = np.arange(0, 20 + 20 / var_freq, 20 / var_freq)
# # plt.plot(var, gamma, color=(0.8, 0.2, 0))
# # plt.hist(D, 100, density=True, histtype="bar", color=(0, 0.3, 0.8))
# # plt.xlabel("x")
# # plt.ylabel("p(x)")
# # plt.savefig("3d.png", dpi=300)
# # plt.close()
#
#
# # Task E
#
# # Average log-likelihood of binomial distribution
#
# # rnd = np.round(D)
# # probs = scipy.stats.binom.pmf(rnd, n, p)
# # L_bin = np.sum(np.log(probs))/D.size
# # print("Avg. log-likelihood of binomial distribution:", L_bin)
#
# # # Average log-likelihood of gamma distribution
#
# # probs = scipy.stats.gamma.pdf(D, k, 0, theta)
# # L_gamma = np.sum(np.log(probs))/D.size
# # print("Avg log-likelihood of gamma  distribution:", L_gamma)
#
# # The binomial distribution was a better fit
#
#
# # Task F
#
m3 = np.sum(np.power(D, 3))/D.size
m4 = np.sum(np.power(D, 4))/D.size
# print(m3, m4)
#
# def funcF (x):
#     return [x[0]*x[1] + x[2]*x[3] - m1,
#             x[1]*(np.power(x[0], 2)+1) + x[3]*(np.power(x[2], 2)+1) - m2,
#             x[1]*(np.power(x[0], 3)+3*x[0]) + x[3]*(np.power(x[2], 3)+3*x[2]) - m3,
#             x[1]*(np.power(x[0], 4)+6*np.power(x[0], 2)+3) + x[3]*(np.power(x[2], 4)+6*np.power(x[2], 2)+3) - m4]
#
# est = [5, 0, 7, 0]
# mu1, p1, mu2, p2 = scipy.optimize.fsolve(funcF, est)
# print(mu1, p1, mu2, p2)
#
# # var_freq = 200
# # var = np.linspace(0, 20, var_freq + 1)
# # gmm = p1*scipy.stats.norm.pdf(var, mu1, 1) + p2*scipy.stats.norm.pdf(var, mu2, 1)
# # plt.plot(var, gmm, color=(0.8, 0.2, 0))
# # plt.hist(D, 100, density=True, histtype="bar", color=(0, 0.3, 0.8))
# # plt.xlabel("x")
# # plt.ylabel("p(x)")
# # plt.savefig("3f.png", dpi=300)
# # plt.close()
#
# probs = p1*scipy.stats.norm.pdf(D, mu1, 1) + p2*scipy.stats.norm.pdf(D, mu2, 1)
# neg_L_gmm = -np.sum(np.log(probs))/D.size
# print("Avg. negative log-likelihood of the GMM distribution:", neg_L_gmm)
#
# # The binomial distribution is still the better fit
#
#
# # Ungraded
#
m5 = np.sum(np.power(D, 5))/D.size
m6 = np.sum(np.power(D, 6))/D.size

# assuming GMM

def func_trial (x):
    return [x[0]*x[1] + x[2]*x[3] - m1,
            x[1]*(np.power(x[0], 2)+np.power(x[4], 2)) + x[3]*(np.power(x[2], 2)+np.power(x[5], 2)) - m2,
            x[1]*(np.power(x[0], 3)+3*x[0]*np.power(x[4], 2)) + x[3]*(np.power(x[2], 3)+3*x[2]*np.power(x[5], 2)) - m3,
            x[1]*(np.power(x[0], 4)+6*np.power(x[0], 2)*np.power(x[4], 2)+3*np.power(x[4], 4)) + x[3]*(np.power(x[2], 4)+6*np.power(x[2], 2)*np.power(x[5], 2)+3*np.power(x[5], 4)) - m4,
            x[1]*(np.power(x[0],5)+10*np.power(x[0],3)*np.power(x[4],2)+15*x[0]*np.power(x[4], 4)) + x[3]*(np.power(x[2],5)+10*np.power(x[2],3)*np.power(x[5],2)+15*x[2]*np.power(x[5],4))-m5,
            x[1]*(np.power(x[0],6)+15*np.power(x[0],4)*np.power(x[4],2)+45*np.power(x[0],2)*np.power(x[4],4)+15*np.power(x[4],6)) + x[3]*(np.power(x[2],6)+15*np.power(x[2],4)*np.power(x[5],2)+45*np.power(x[2],2)*np.power(x[5],4)+15*np.power(x[5],6))-m6]

est = [5, 0.5, 8, 0.5, 1, 1] # mu1, p1, m2, p2, s1, s2
mu1, p1, mu2, p2, s1, s2 = scipy.optimize.fsolve(func_trial, est)
# print(mu1, p1, mu2, p2)

var_freq = 200
var = np.linspace(0, 20, var_freq + 1)
gmm = p1*scipy.stats.norm.pdf(var, mu1, s1) + p2*scipy.stats.norm.pdf(var, mu2, s2)
plt.plot(var, gmm, color=(0.8, 0.2, 0))
plt.hist(D, 100, density=True, histtype="bar", color=(0, 0.3, 0.8))
plt.xlabel("x")
plt.ylabel("p(x)")
plt.savefig("3_trial1.png", dpi=300)
plt.close()
#
#
# # Assuming gamma
# # Unable to solve equations
#
# # def func_trial (x):
# #     return [x[0]*x[1]*x[2] + x[3]*x[4]*x[5] - m1,
# #             x[0]*x[1]*(x[1]+1)*np.power(x[2],2) + x[3]*x[4]*(x[4]+1)*np.power(x[5],2) - m2,
# #             x[0]*x[1]*(x[1]+1)*(x[1]+2)*np.power(x[2],3) + x[3]*x[4]*(x[4]+1)*(x[4]+2)*np.power(x[5],3) - m3,
# #             x[0]*x[1]*(x[1]+1)*(x[1]+2)*(x[1]+3)*np.power(x[2],4) + x[3]*x[4]*(x[4]+1)*(x[4]+2)*(x[4]+3)*np.power(x[5],4) - m4,
# #             x[0]*x[1]*(x[1]+1)*(x[1]+2)*(x[1]+3)*(x[1]+4)*np.power(x[2],5) + x[3]*x[4]*(x[4]+1)*(x[4]+2)*(x[4]+3)*(x[4]+4)*np.power(x[5],5) - m5,
# #             x[0]*x[1]*(x[1]+1)*(x[1]+2)*(x[1]+3)*(x[1]+4)*(x[1]+5)*np.power(x[2],6) + x[3]*x[4]*(x[4]+1)*(x[4]+2)*(x[4]+3)*(x[4]+4)*(x[4]+5)*np.power(x[5],6) - m6]
#
# # est = [0.5, 9, 0.9, 0.5, 9, 0.5]
# # p1, k1, theta1, p2, k2, theta2 = scipy.optimize.fsolve(func_trial, est)
#
# # p1, k1, theta1, p2, k2, theta2 = [0.5, 15, 0.2, 0.5, 30, 0.4]
# # var_freq = 200
# # var = np.linspace(0, 20, var_freq + 1)
# # gamma_mm = p1*scipy.stats.gamma.pdf(var, k1, 0, theta1) + p2*scipy.stats.gamma.pdf(var, k2, 0, theta2)
# # plt.plot(var, gamma_mm, color=(0.8, 0.2, 0))
# # plt.hist(D, 100, density=True, histtype="bar", color=(0, 0.3, 0.8))
# # plt.xlabel("x")
# # plt.ylabel("p(x)")
# # plt.savefig("3_trial2.png", dpi=300)
# # plt.close()
#
#
# # Assuming beta
#
#
