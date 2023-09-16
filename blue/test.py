from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from blue_func import *

N = 5000

plt.style.use('dark_background')
fig, (ax1) = plt.subplots(1)  # , figsize=(22, 14))

H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14

tau = np.logspace(10, 20, N)
k_arr = [1e-18,1e-3,1e9]
colors = ['red', 'green', 'blue']
k_arr = [1e-18]
idx = 0

def P_massless(f,m,tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    C_1 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(-3/4, m/(2*H_inf)) + (1-2*nu)*scipy.special.jv(1/4, m/(2*H_inf)))
    C_2 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(3/4, m/(2*H_inf)) - (1-2*nu)*scipy.special.jv(-1/4, m/(2*H_inf)))
    lamb = m*tau_m**2 / (2*H_inf*tau_r**2)
    D_1 = -np.sin(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    D_2 = np.cos(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    v_k_reg3 = 2/k*np.sqrt(m*tau_m / (np.pi*H_inf*tau_r**2)) * (D_1*np.cos(k*tau) + D_2*np.sin(k*tau))

    return 4*k**3*np.abs(v_k_reg3)**2/(np.pi**2*M_pl**2*(tau/(tau_r**2*H_inf))**2)


def omega_GW_massless(f, m, tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    return np.pi**2/(3*H_0**2)*f**3*P_massless(f,m,tau)

m_arr = [.5*H_inf, .8*H_inf]
f = np.logspace(-18, 5, N)
for m in m_arr:    
    ax1.plot(
        f, omega_GW_massless(f,m,tau_r*1e-10), "--",
        color=colors[idx],
        label=f'm = {m}'
    )
    idx+=1
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(1e-18, 1e6)

plt.legend()
plt.show()
