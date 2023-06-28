import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *

N = 2000
omega_0 = k_0/a_0
k = a_0*np.sqrt(omega_0**2-M_GW**2)
print(K)
k = 1e1
eta = np.logspace(-10, 2, N)
H_square = np.vectorize(Hubble)(eta) ** 2
ang_freq_square = np.vectorize(ang_freq)(eta) ** 2
a = np.vectorize(scale_fac)(eta)

k_idx = 0
for i in range(0, len(eta)):
    if a[i] >= a_0:
        print(a[i])
        k_idx = i
        break

# print("Current eta_0 = "+f'{eta[k_idx]}')

time = scipy.integrate.cumtrapz(a, eta, initial=0)

fig, (ax1) = plt.subplots(1)

#ax1.plot(eta, a, label='a')
#ax1.axhline(y=(k/M_GW)**2, linestyle="dashed",
#            linewidth=1, label=r'$k_0/M_{GW}$')
#ax1.plot(eta, time, label='Physical time')
#res, err = scipy.integrate.quad(scale_fac, 0, 2.105)

# print('Time: ' + f'{res}')

#roots = scipy.optimize.fsolve(
#    lambda x: scipy.integrate.quad(scale_fac, 0, x)[0] - 1, 1)
# print('Roots: ', roots[0])



a_k_1 = omega_M*H_0**2+H_0*np.sqrt(4*omega_R*k**2+(H_0*omega_M)**2)/(2*k**2)
print('Using inverse with Wolfalph: ', a_k_1)
idx_1 = 0
H_reg = np.vectorize(Hubble)(eta)
for i in range(0, len(eta)):
    if k >= a[i]*H_reg[i]:
        idx_1 = i
        break
print('Using graph: ', a[idx_1])

ax1.plot(a, a*H_reg,label='aH', color='orange')
ax1.axhline(y=k, label='k')

ax1.legend(loc='best')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Scale Factor')

# plt.savefig("emir/emir_calc_figs/scale_factor_evolution.pdf")
plt.show()
