import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint


c_g = 1  # speed of graviton
hbar = 6.582119569e-25  # GeV / Hz
c = 3e8 / 3.086e25  # m * Hz * (1Gpc / 3.086e25m) = Gpc * Hz
M_GW = 2e-7*hbar  # Hz * GeV / Hz = GeV
M_pl = 1.22089e19  # GeV
H_0 = 1/2997.9*.7*1000 * c*hbar  # GeV, Jacob's val
H_0 = M_GW/1e10  # my val, from page 13 in Emir paper
k_0 = 1e10*H_0
k_c = 1e4*H_0  # both k_c and k_0 defined in same place in Emir paper
omega_M = .3
omega_R = 8.5e-5
omega_L = .7
eta_rm = .1
eta_ml = 12.5
a_0 = 1


def a_rd(eta):
    return H_0*np.sqrt(omega_R)*eta


def a_md(eta):
    return H_0**2*.25*omega_M*eta**2


def a_rd_p(eta):
    return H_0*np.sqrt(omega_R)


def a_md_p(eta):
    return 2*eta*H_0**2*.25*omega_M


k = [1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5]

k = 1
N = 1000
eta = np.logspace(-10, 1, N)

fig, (ax1) = plt.subplots(1)
gamma_k = 4*H_0*a_0*eta_rm**(3/2) / (9*M_pl*k**(3/2)) * eta**2
C = 4*H_0*a_0*eta_rm**(3/2) / (9*M_pl*k**(3/2))
gamma_k = C*3/k**2*(-np.cos(k*eta)+np.sin(k*eta)/(k*eta))
# ax1.plot(eta, gamma_k)
k = 1e1
gamma_k2 = 3*C/k**(3/2)*np.cos(k*eta)
ax1.plot(eta, gamma_k2)


ax1.set_xscale('log')
plt.show()
