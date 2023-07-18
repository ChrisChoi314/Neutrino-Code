import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *
import math

N = 2000
k_arr = [1e3, 3e3, 5e3]
k_arr = [1e-14, 5e-14, 3e-13]
colors = ['blue', 'red', 'green']
# k_arr = [1e-8, 1e-7, 1e-6]
# fig, (ax1, ax2) = plt.subplots(2, figsize=(22, 14))
fig, (ax1) = plt.subplots(1)
eta = np.logspace(1,math.log(conf_time(a_eq), 10), N)
# eta = np.logspace(-5, -1, N)
eta = np.logspace(10,17, N)
a = np.vectorize(scale_fac)(eta)
v_0 = 1
v_prime_0 = 0
i = 0   
for k in k_arr:
    v, v_prime = odeint(diffeqGR, [v_0, v_prime_0], eta, args=(k,)).T
    # ax1.plot(eta, v/a, label=f"{k}" + " Hz")
    v, v_prime = odeint(diffeqMG, [v_0, v_prime_0], eta, args=(k,)).T
    # ax1.plot(eta, v/a,"--", label=f"{k}" + r" $Hz$")
    v, v_prime = odeint(diffeqMode, [v_0, v_prime_0], eta, args=(k,0)).T
    ax1.plot(eta, v, label=f"{k}" + r" $Hz - GR$", color = colors[i])
    v, v_prime = odeint(diffeqMode, [v_0, v_prime_0], eta, args=(k,M_GW)).T
    ax1.plot(eta, v,"--", label=f"{k}" + r" $Hz - MG$", color = colors[i])
    i+=1
'''
k = 1e3
C = 4 * H_0 * a_0 * eta_rm ** (3 / 2) / (9 * M_pl * k ** (3 / 2))
gamma_k = C * 3 / k**2 * (-np.cos(k * eta) + np.sin(k * eta) / (k * eta))
gamma_k = normalize(gamma_k)
# ax1.plot(eta, gamma_k / a, "--", label="Analyt soln, k = k_c")
p = 0
q = 2 * (p + 3)
gamma_k2_approx = (
    2 ** (1 / 4 + 3 / (2 * q))
    * C
    * (3 / 2) ** (3 / 2 - 3 / q)
    * M_GW ** (-3 / q)
    / (np.sqrt(k))
    * np.cos(k * eta)
)
gamma_k2_approx = normalize(gamma_k2_approx)
# ax2.plot(eta, gamma_k2_approx, "--", label="Analyt soln approx, k = k_c")
gamma_k2 = (
    C
    * (3 / 2) ** (3 / 2 - 3 / q)
    * q ** (3 / q)
    * scipy.special.gamma(1 + 3 / q)
    / M_GW ** (3 / q)
    * eta ** (1 / 2)
    * scipy.special.jv(3 / q, 3 * (2 / 3) ** (q / 2) * M_GW / q * eta ** (q / 2))
)
gamma_k2 = normalize(gamma_k2)
'''
# ax2.plot(eta, gamma_k2 / a, "--", label="Analyt soln exact, k = k_c")


# ax1.title.set_text(r"GR for $k \approx k_c$")
ax1.title.set_text(r"GR vs Massive Gravity in the RD era")
ax1.set_ylabel(r"$\gamma_k(\eta)$")
ax1.set_xlabel(r"$\eta$")
ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.legend(loc="best")
'''
ax2.title.set_text(r"MG for $k \approx k_c$")
ax2.set_ylabel(r"$\gamma_k(\eta)$")
# ax2.set_xlabel(r'$\eta(Gpc)$')
ax2.set_xscale("log")
ax2.legend(loc="best")
'''

ax1.axvline(x=eta_rm, label='k', color = 'orange')
ax1.text(1e15, .6, "Radiation\nMatter\nEquality", fontsize=10)

# plt.savefig("emir/emir_mode_figs/fig8.pdf")
plt.show()
