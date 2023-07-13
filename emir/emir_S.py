import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from emir_func import *

N = 1000
eta = np.logspace(-7, 1, N)
# k = np.logspace(-5,5, N)
omega_0 = k_0 / a_0
omega_0 = np.linspace(M_GW/2, 2*M_GW, N)
omega_0 = np.logspace(math.log(M_GW, 10) , math.log(M_GW, 10) + 6, N)
omega_0 = np.logspace(math.log(M_GW, 10) - .04, math.log(M_GW, 10) + .2, N)
# omega_0 = np.linspace(M_GW - 10**.04, M_GW + 1.2, N)
k_prime = (
    a_0 * omega_0
)
eq_idx = 0
a = np.vectorize(scale_fac)(eta) 
# k = np.array([np.inf if o0 <= M_GW else a_0 * np.sqrt(o0**2 - M_GW**2) for o0 in omega_0])
k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
H_square = np.vectorize(Hubble)(eta) ** 2
ang_freq_square = np.vectorize(ang_freq)(eta,k) ** 2
for i in range(0, len(eta)):
    if H_square[i] <= ang_freq_square[i]:
        eta_k = eta[i]
        eq_idx = i
        break

a_k = solve(k)
# a_k = np.vectorize(solve_one)(k)
a_eq = scale_fac(eta_rm)
print(a_eq)

print( 1/(1+3400))
print(omega_R/omega_M)
a_eq = .000901
a_eq = omega_R/omega_M
H_eq = Hubble_a(a_eq)

k_eq = a_eq*H_eq
omega_c = np.sqrt((k_c/a_c)**2 +M_GW**2)


def enhance_approx(x):
    if x < M_GW:
        return 0.
    val = a_0 * np.sqrt(x**2 - M_GW**2)
    if k_0 < val:
        return 1.

    elif val <= k_0 and val >= k_c:
        if val >= k_eq:
            otpt = (x**2 / M_GW**2 - 1)**(-3/4)
            return otpt
        if val < k_eq and k_eq < k_0:
            otpt = k_eq/(np.sqrt(2)*k_0)(x**2 / M_GW**2 - 1)**(-5/4)
            return otpt
        if k_eq > k_0:
            otpt = (x**2 / M_GW**2 - 1)**(-5/4)
            return otpt
    elif val <= k_c:
        beta = H_eq**2 * a_eq**4 / (2)
        a_k_0_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_0**2 + beta)) / (
            2. * a_eq * k_0**2
        )
        if abs(x**2 / M_GW**2 - 1) < 1e-25:
            return 0.
        otpt = a_c/a_k_0_GR*np.sqrt(k_c/k_0)*(x**2 / M_GW**2 - 1)**(-1/2)
        return otpt


S_approx = np.vectorize(enhance_approx)(omega_0)
# a_k = k / np.sqrt(omega_k**2 - M_GW**2)
omega_k = np.sqrt((k / a_k) ** 2 + M_GW**2)
beta = H_eq**2 * a_eq**4 / (2)
a_k_prime_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_prime**2 + beta)) / (
    2 * a_eq * k_prime**2
)

a_k_prime_GR_approx = a_eq*(k_eq/(np.sqrt(2)*k_prime))

a_k_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k**2 + beta)) / (
    2 * a_eq * k**2
)
# a_k_prime_GR = inv_of_H(beta, )
S = np.where(omega_0 <= M_GW, 0, k_prime * a_k / (k * a_k_prime_GR)
             * np.sqrt(omega_k * a_k / (omega_0 * a_0)))

fig, (ax1) = plt.subplots(1)
ax1.plot(omega_0, S, label=r"$S(\omega_0)$ completely analytical")
ax1.plot(omega_0, S_approx, label=r"$S(\omega_0)$ semi analytical")

#ax1.axvline(x=np.sqrt(k_0**2/a_0**2 + M_GW**2), linestyle="dashed", linewidth=1,
#            color='blue', label=r"$k_0$")
#ax1.axvline(x=np.sqrt(k_eq**2/a_0**2 + M_GW**2), linestyle="dashed", linewidth=1,
#            color='red', label=r"$k_{eq}$")
#print(S)
#print(omega_k[int(N/2):] - k[int(N/2):]/a_k)
#print(omega_0[int(N/2):] - k[int(N/2):]/a_0)
#print(k_prime[int(N/2):] / k[int(N/2):])
#print((omega_0[int(N/2):] - np.sqrt(omega_0[int(N/2):]**2 - M_GW**2))*a_0)
#print(a_k[int(N/2):] / a_k_prime_GR[int(N/2):])

#print((omega_k[int(N/2):]*a_k[int(N/2):]) / (omega_0[int(N/2):]*a_0))
#print(a_k_prime_GR[int(N/2):]/   a_k_prime_GR_approx[int(N/2):])
print(a_k[int(N/2):]/   a_k_prime_GR[int(N/2):])
ax1.axvline(x=M_GW, linestyle="dashed", linewidth=1,
            color='green', label=r"$M_{GW,0}$")
ax1.axvline(x=omega_c, linestyle="dashed", linewidth=1,
            color='cyan', label=r"$\omega_c$")
ax1.axvline(x=M_GW*np.sqrt(2), linestyle="dashed", linewidth=1,
            color='purple', label=r"$\sqrt{2} M_{GW,0}$")
ax1.axhline(y=1, linestyle="dashed", linewidth=1, label=r"1")

ax1.legend(loc="best")
ax1.set_ylim(0, 20)
ax1.set_xscale("log")
ax1.set_title("Transfer Function")
ax1.set_xlabel(r'$\omega_0$ [Hz]')
ax1.set_ylabel(r'$S(\omega_0)$')

plt.savefig("emir/emir_S_figs/fig5.pdf")
plt.show()
