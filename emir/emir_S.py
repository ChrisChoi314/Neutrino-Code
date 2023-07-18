import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from sympy import *

from emir_func import *

N = 1000
omega_0 = np.logspace(math.log(M_GW, 10), math.log(M_GW, 10) + 6, N)
omega_0 = np.logspace(math.log(M_GW, 10) - .1, math.log(M_GW, 10) + .4, N)
# omega_0 = np.linspace(M_GW - 10**.04, M_GW + 1.2, N)
k_prime = (
    a_0 * omega_0
)
k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
a_k = ak(k)  # uses multithreading to run faster

# uses the power law approximations from Emir's paper, section 3E


omega_c = np.sqrt((k_c/a_c)**2 + M_GW**2)
def enhance_approx(x):
    if x < M_GW:
        return 0.
    val = a_0 * np.sqrt(x**2 - M_GW**2)
    if k_0 < val:
        return 1.
    elif val <= k_0 and val >= k_c:
        if val >= k_eq:
            output = (x**2 / M_GW**2 - 1)**(-3/4)
            return output
        if val < k_eq and k_eq < k_0:
            output = k_eq/(np.sqrt(2)*k_0)(x**2 / M_GW**2 - 1)**(-5/4)
            return output
        if k_eq > k_0:
            output = (x**2 / M_GW**2 - 1)**(-5/4)
            return output
    elif val <= k_c:
        beta = H_eq**2 * a_eq**4 / (2)
        a_k_0_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_0**2 + beta)) / (
            2. * a_eq * k_0**2
        )
        if abs(x**2 / M_GW**2 - 1) < 1e-25:
            return 0.
        output = a_c/a_k_0_GR*np.sqrt(k_c/k_0)*(x**2 / M_GW**2 - 1)**(-1/2)
        return output

S_approx = np.vectorize(enhance_approx)(omega_0)

omega_k = np.sqrt((k / a_k) ** 2 + M_GW**2)
beta = H_eq**2 * a_eq**4 / (2)
a_k_prime_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_prime**2 + beta)) / (
    2 * a_eq * k_prime**2
)

a_k_prime_GR_approx = a_eq*(k_eq/(np.sqrt(2)*k_prime))

a_k_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k**2 + beta)) / (
    2 * a_eq * k**2
)

S = np.where(omega_0 <= M_GW, 0, k_prime * a_k / (k * a_k_prime_GR)
             * np.sqrt(omega_k * a_k / (omega_0 * a_0)))

fig, (ax1) = plt.subplots(1)
ax1.plot(omega_0, S, label=r"$S(\omega_0)$ completely analytical")
ax1.plot(omega_0, S_approx, label=r"$S(\omega_0)$ semi analytical")

ax1.axvline(x=M_GW, linestyle="dashed", linewidth=1,
            color='green')
# ax1.axvline(x=omega_c, linestyle="dashed", linewidth=1,
            # color='cyan', label=r"$\omega_c$")
# ax1.axvline(x=M_GW*np.sqrt(2), linestyle="dashed", linewidth=1,)
ax1.axhline(y=1, linestyle="dashed", linewidth=1,color='black')
ax1.vlines(M_GW*np.sqrt(2),0, 1.8,linewidth=1,linestyle="dashed",color='black')
ax1.text(2.2e-8, -1.5, r'$M_{GW,0}$', fontsize=11,color='black')
ax1.text(1.64e-8, .65, r'1', fontsize=10,color='black')
ax1.text(3.15e-8, -1.5, r'$\sqrt{2}M_{GW,0}$', fontsize=11,color='black')
ax1.text(5.5e-8, -1.5, r'$\omega_0$', fontsize=10,color='black')

ax1.legend(loc="best")
ax1.set_ylim(0, 20)
ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.set_title(r'Enhancement Factor')
ax1.set_ylabel(r'$S(\omega_0)$')

ax1.get_xaxis().set_visible(False)
ax1.set_xlabel(r'$\omega_0$ [Hz]')
plt.yticks([]) 

plt.savefig("emir/emir_S_figs/fig7.pdf")
plt.show()
