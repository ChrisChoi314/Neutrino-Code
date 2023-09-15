from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from blue_func import *

N = 10000

plt.style.use('dark_background')
fig, (ax1) = plt.subplots(1)  # , figsize=(22, 14))

H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14

tau = np.logspace(0, 20, N)
m = .5*H_inf
nu = np.sqrt(9/4 - m**2 / H_inf**2)
k_arr = [1e-18,1e-3,1e9]
colors = ['red', 'green', 'blue']

idx = 0
for k in k_arr:    
    C_1 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(-3/4, m/(2*H_inf)) + (1-2*nu)*scipy.special.jv(1/4, m/(2*H_inf)))
    C_2 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(3/4, m/(2*H_inf)) - (1-2*nu)*scipy.special.jv(-1/4, m/(2*H_inf)))
    lamb = m*tau_m**2 / (2*H_inf*tau_r**2)
    D_1 = -np.sin(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    D_2 = np.cos(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    v_k_reg3 = 2/k*np.sqrt(m*tau_m / (np.pi*H_inf*tau_r**2)) * (D_1*np.cos(k*tau) + D_2*np.sin(k*tau))

    ax1.plot(
        tau, np.abs(v_k_reg3)**2, "--",
        color=colors[idx],
        label=f'k = {k}'
    )
    idx+=1

plt.legend()
plt.show()
