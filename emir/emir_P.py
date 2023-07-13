import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from emir_func import *

N = 1000
omega_arr = np.array([5e-7, 5e-6, 5e-5, 5e-4, 5e-3])
k_arr = a_0*np.sqrt(omega_arr**2 - M_GW**2)
fig, (ax1, ax2) = plt.subplots(2, figsize=(22, 14))

eta = np.logspace(1, 18, N)
# eta = np.logspace(-3, -1, N)
a = np.vectorize(scale_fac)(eta)
# a = normalize_0(a)
v_0 = 1
v_prime_0 = 0
for k in k_arr:
    v, v_prime = odeint(diffeqGR, [v_0, v_prime_0], eta, args=(k,)).T
    print(v[-500:]/a[-500:])
    ax1.plot(eta, v, label=f"{k}" + r" $Hz$")
    v, v_prime = odeint(diffeqMG, [v_0, v_prime_0], eta, args=(k,)).T
    print(v[-500:]/a[-500:])
    ax2.plot(eta, v, label=f"{k}" + r" $Hz$")
eta = np.logspace(-7, 1, N)
# k = np.logspace(-5,5, N)
omega_0 = k_0 / a_0
omega_0 = np.logspace(math.log(M_GW, 10), math.log(M_GW, 10) + 1, N)
omega_0 = np.linspace(.5*M_GW, 3 * M_GW, N)
k_prime = (
    a_0 * omega_0
)  # seems a bit circular, but that is how Mukohyama defined it in his paper

a_eq = omega_R/omega_M
H_eq = Hubble_a(a_eq)
k_eq = a_eq*H_eq


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
P = np.where(omega_0 <= M_GW, 0, omega_0**2 /
             (omega_0**2-M_GW**2)*(2*k**3/np.pi**2))

# P = omega_0**2/(omega_0**2-M_GW**2)*(2*k**3/np.pi**2)#*y_k_0**2
P_GR = (2*k_prime**3/np.pi**2)  # *y_k_0**2
# print(y_k_0**2)

# ax1.plot(eta, v_MG, label='MG')
# ax1.plot(eta, v_GR, label='GR')
# ax1.plot(omega_0, P, label='P ')
# ax1.plot(omega_0, P_GR, label='P GR')
# ax1.plot(omega_0, np.sqrt(P/P_GR), label='S^2')
ax1.legend(loc='best')
ax1.set_xscale("log")

ax2.legend(loc='best')
ax2.set_xscale("log")

# plt.savefig("emir/emir_P_figs/fig1.pdf")
plt.show()
