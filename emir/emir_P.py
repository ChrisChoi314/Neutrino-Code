import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from emir_func import *

N = 2000
P_prim_k = 2.43e-10
omega_arr = np.array([5e-7, 5e-6, 5e-5, 5e-4, 5e-3])
omega_arr = np.array([5e-7])
k_arr = a_0*np.sqrt(omega_arr**2 - M_GW**2)
fig, (ax1) = plt.subplots(1)  # , figsize=(22, 14))

eta = np.logspace(1, 18, N)
# eta = np.logspace(-3, -1, N)
a = np.vectorize(scale_fac)(eta)
# a = normalize_0(a)
v_0 = 1
v_prime_0 = 0
eta_0_idx = 0
for i in range(len(eta)):
    if eta[i] >= eta_0:
        eta_0_idx = i
        break
for k in k_arr:
    v, v_prime = odeint(diffeqGR, [v_0, v_prime_0], eta, args=(k,)).T
    # print(v[eta_0_idx]/a[eta_0_idx])
    # ax1.plot(eta, v/a, label=f"{k}" + r" $Hz$")
    v, v_prime = odeint(diffeqMG, [v_0, v_prime_0], eta, args=(k,)).T
    # print(v[eta_0_idx]/a[eta_0_idx])
    # ax2.plot(eta, v/a, label=f"{k}" + r" $Hz$")
eta = np.logspace(-7, 1, N)
# k = np.logspace(-5,5, N)
omega_0 = np.linspace(.5*M_GW, 3 * M_GW, N)
omega_0 = np.logspace(-8+ .2, -7)
omega_0 = np.logspace(math.log(M_GW, 10)+.00000000000001, math.log(M_GW, 10) + 1, N)
print(omega_0)
k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
print(k)
a_k = solve(k)  # uses multithreading to run faster
omega_k = np.sqrt((k / a_k) ** 2 + M_GW**2)
k_prime = (
    a_0 * omega_0
) 
beta = H_eq**2 * a_eq**4 / (2)
a_k_prime_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_prime**2 + beta)) / (
    2 * a_eq * k_prime**2
)

# fig, (ax1) = plt.subplots(1, figsize=(22, 14))

eta = np.logspace(-5, 1, N)
a = np.vectorize(scale_fac)(eta)
a = normalize_0(a)
v_0 = 1
v_prime_0 = 0
'''
k = k_prime
v_GR, v_prime_GR = odeint(diffeqGR, [v_0, v_prime_0], eta).T
k_pure = a_0 * np.sqrt(omega_0**2 - M_GW**2)
k = k_pure
v_MG, v_prime_MG = odeint(diffeqMG, [v_0, v_prime_0], eta).T
k_idx = 0
for i in range(0, len(eta)):
    if eta[i] >= eta_0:
        print(eta[i])
        k_idx = i
        break
y_k_GR_0 = v_GR[k_idx]
ax1.axvline(x=M_GW, label='k - GR', color='orange')
y_k_0 = v_MG[k_idx]

'''


def A(k):
    return np.where(k >= 0., np.sqrt(P_prim_k*np.pi**2/(2*k**3)), -1.)


gamma_k_t_0 = A(k)*np.sqrt(omega_k * a_k**3 / (omega_0*a_0**3))
P = np.where(omega_0 <= M_GW, np.nan, omega_0**2 /
             (omega_0**2-M_GW**2)*(2*k**3/np.pi**2)*gamma_k_t_0**2)
print(P)
# P = omega_0**2/(omega_0**2-M_GW**2)*(2*k**3/np.pi**2)#*y_k_0**2
gamma_k_GR_t_0 = A(k_prime)*a_k_prime_GR/a_0
P_GR = (2*k_prime**3/np.pi**2)*gamma_k_GR_t_0**2  # *y_k_0**2
S = np.where(omega_0 <= M_GW, np.nan, k_prime * a_k / (k * a_k_prime_GR)
             * np.sqrt(omega_k * a_k / (omega_0 * a_0)))


# ax1.plot(eta, v_MG, label='MG')
# ax1.plot(eta, v_GR, label='GR')
ax1.plot(omega_0, np.sqrt(P), label='numerical')
ax1.plot(omega_0, np.sqrt(P_GR*S**2), '--', label='fully analytical')
# ax1.plot(omega_0, np.sqrt(P/P_GR), label='S^2')
ax1.set_xlabel(r'$\omega_0$ [Hz]')
ax1.set_ylabel(r'$[P(\omega_0)]^{1/2}$')

ax1.set_xlim(1e-8, 1e-7)
ax1.set_ylim(1e-18, 1e-2)
ax1.legend(loc='best')
ax1.set_xscale("log")
ax1.set_yscale("log")

# ax2.legend(loc='best')
# ax2.set_xscale("log")
plt.title('Power Spectrum')

plt.savefig("emir/emir_P_figs/fig2.pdf")
plt.show()
