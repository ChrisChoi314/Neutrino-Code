import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *
import math

N = 2000
k_arr = [1e2, 1e3, 1e4, 1e5]
k_arr = [1e3, 3e3, 5e3]
m_arr = [1e-5, 1e-6, 1e-7, 1e-8]
# k_arr = [1e-8, 1e-7, 1e-6]
# fig, (ax1, ax2) = plt.subplots(2, figsize=(22, 14))
fig, (ax1) = plt.subplots(1)
eta = np.logspace(1,math.log(conf_time(a_eq), 10), N)
eta = np.logspace(1,12, N)
eta = np.logspace(-5, -1, N)
a = np.vectorize(scale_fac)(eta)
v_0 = 1
v_prime_0 = 0
for k in k_arr:
    v, v_prime = odeint(diffeqGR, [v_0, v_prime_0], eta, args=(k,)).T
    # ax1.plot(eta, v/a, label=f"{k}" + " Hz")
    v, v_prime = odeint(diffeqMG, [v_0, v_prime_0], eta, args=(k,)).T
    # ax1.plot(eta, v/a,"--", label=f"{k}" + r" $Hz$")
    v, v_prime = odeint(diffeqMode, [v_0, v_prime_0], eta, args=(k,M_GW)).T
    ax1.plot(eta, v,"--", label=f"{k}" + r" $Hz$")
    #print(f'Horizon crossing is: {conf_time(solve_one(k))}')
for k in k_arr:
    print(f'Horizon crossing is: {conf_time(solve_one(k))}')
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
# ax2.plot(eta, gamma_k2 / a, "--", label="Analyt soln exact, k = k_c")


# ax1.title.set_text(r"GR for $k \approx k_c$")
ax1.title.set_text(r"GR vs Massive Gravity in the RD era")
ax1.set_ylabel(r"$\gamma_k(\eta)$")
# ax1.set_xlabel(r'$\eta(Gpc)$')
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
H_square = np.vectorize(Hubble)(eta) ** 2

for k in k_arr:
    ang_freq_square = np.vectorize(ang_freq)(eta,k) ** 2
    eta_k = 0
    k_idx = 0
    for i in range(0, len(eta)):
        if H_square[i] <= ang_freq_square[i]:
            eta_k = eta[i]
            k_idx = i
            break
    '''
    plt.axvline(
        x=eta_k,
        linestyle="dashed",
        linewidth=1,
        label="Horizon crossing: " + f"{k}" + r" $Gpc^{-1}$",
    )
    '''

# ax1.axvline(x=eta_rm, label='k', color = 'red')
# ax1.text(1e11, 3e34, "Radiation\nMatter\nEquality", fontsize=10)

'''
ax3.plot(eta, H_square, label=r"$H^2$" + f": {k}" + r" $Gpc^{-1}$")
ax3.plot(eta, ang_freq_square, label=r"$\omega^2$: " + f"{k}" + r" $Gpc^{-1}$")

ax3.title.set_text(r"Horizon Crossing")
ax3.set_xlabel(r"$\eta(Gpc)$")
ax3.set_xscale("log")
ax3.legend(loc="best")
'''

plt.savefig("emir/emir_mode_figs/fig6.pdf")
plt.show()
