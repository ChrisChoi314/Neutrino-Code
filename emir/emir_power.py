import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from emir_func import *

N = 100
eta = np.logspace(-7,1,N)
#k = np.logspace(-5,5, N)
omega_0 = k_0/a_0
omega_0 = np.logspace(math.log(M_GW,10)+.00001,math.log(M_GW,10) + 10, N)
k_prime = a_0*omega_0 # seems a bit circular, but that is how Mukohyama defined it in his paper
eq_idx = 0
H_square = np.vectorize(Hubble)(eta)**2
ang_freq_square = np.vectorize(ang_freq)(eta)**2
for i in range(0, len(eta)):
    if H_square[i] <= ang_freq_square[i]:
        eta_k = eta[i]
        eq_idx = i
        break


a_eq = scale_fac(eta_rm)
H_eq = Hubble(eta_rm)

k = a_0*np.sqrt(omega_0**2-M_GW**2)
omega_k = np.sqrt((k/a_0)**2 + M_GW**2)
print(omega_k)
print(omega_k**2 - M_GW**2)
a_k = k / np.sqrt(omega_k**2 - M_GW**2)
beta = H_eq**2*a_eq**4/(2)
a_k_prime_GR = (beta+np.sqrt(beta)*np.sqrt(4*a_eq**2*k_prime**2+beta))/(2*a_eq*k_prime**2)
# a_k_prime_GR = inv_of_H(beta, )
S = k_prime* a_k / (k* a_k_prime_GR)*np.sqrt(omega_k*a_k/(omega_0*a_0))

fig, (ax1) = plt.subplots(1)
ax1.plot(omega_0, S, label = r'$S(\omega_0)$')

ax1.legend(loc='best')
ax1.set_xscale('log')
ax1.set_title('Transfer Function')

plt.savefig("emir/emir_power_figs/fig1.pdf")
plt.show()