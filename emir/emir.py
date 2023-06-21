import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint


c_g = 1  # speed of graviton
hbar = 6.582119569e-25  # GeV / Hz
c = 3e8 / 3.086e25  # m * Hz * (1Gpc / 3.086e25m) = Gpc * Hz
M_GW = 2e-7*hbar  # Hz * GeV / Hz = GeV
H_0 = M_GW/1e10  # my val, from page 13 in Emir paper
M_pl = 1.22089e19  # GeV
H_0 = 1/2997.9*.7*1000 * c*hbar  # GeV, Jacob's val
k_0 = 1e10*H_0
k_c = 1e4*H_0  # both k_c and k_0 defined in same place in Emir paper
omega_M = .3
omega_R = 8.5e-5
omega_L = .7
eta_rm = .1
eta_ml = 12.5
K = 0
M_pl /= hbar

T = 6.58e-25
L = 1.97e-16
m2Gpc = 3.1e25

MGW = 2e-7


def hz2gpc(hz): return hz*(T/L)*m2Gpc
def gpc2hz(gpc): return gpc*(1/m2Gpc)*L/T


M_pl = hz2gpc(M_pl)
M_GW = hz2gpc(MGW)
H_0 = 1/2997.9*.7*1000  # Gpc^-1
H_0 = M_GW/1e10
k_0 = 1e10*H_0
k_c = 1e4*H_0
eta_rm = .1
a_c = k_c / M_GW
a_0 = k_0 / M_GW


def a_rd(eta):
    return H_0*np.sqrt(omega_R)*eta


def a_md(eta):
    return H_0**2*.25*omega_M*eta**2


def a_rd_p(eta):
    return H_0*np.sqrt(omega_R)


def a_md_p(eta):
    return 2*eta*H_0**2*.25*omega_M


def scale_fac(conf_time):
    if conf_time < eta_rm:
        return H_0*np.sqrt(omega_R)*conf_time
    else:
        return H_0**2*.25*omega_M*conf_time**2


def d_scale_fac_dz(conf_time):
    if conf_time < eta_rm:
        return 0
    else:
        return 2*H_0**2*.25*omega_M


def diffeqMG(M, u):
    r = [M[1], -((c_g*k)**2 + (scale_fac(u)*M_GW)**2 -
                 d_scale_fac_dz(u) / scale_fac(u) + 2*K*c_g**2) * M[0]]
    # print(r)
    return r


def diffeqGR(M, u):
    r = [M[1], -((c_g*k)**2 -
                 d_scale_fac_dz(u) / scale_fac(u) + 2*K*c_g**2) * M[0]]
    # print(r)
    return r


def M_derivs_homoJ(M, u):
    r = [M[1], -((c_g*k)**2 + (a_rd(u)*M_GW)**2 -
                 a_rd_p(u) / a_rd(u) + 2*K*c_g**2) * M[0]]
    print(r)
    return r


def Hubble(conf_t):
    return H_0*np.sqrt(omega_M / scale_fac(conf_t)**3 + omega_R / scale_fac(conf_t)**4 + omega_L)


def ang_freq(conf_t):
    return np.sqrt(k**2/scale_fac(conf_t)**2 + M_GW**2)


N = 5000
k_arr = [1e3, 5e3, 1e4]
'''
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(22, 14))

eta = np.logspace(-5, 0, N)
v_0 = 1
v_prime_0 = 0
for k in k_arr:
    v, v_prime = odeint(diffeqGR, [v_0, v_prime_0], eta).T
    ax1.plot(eta, v, label=f'{k}'+r' $Gpc^{-1}$')
for k in k_arr:
    v, v_prime = odeint(diffeqMG, [v_0, v_prime_0], eta).T
    ax2.plot(eta, v, label=f'{k}'+r' $Gpc^{-1}$')

k = k_c
C = 4*H_0*a_0*eta_rm**(3/2) / (9*M_pl*k**(3/2))
gamma_k = C*3/k**2*(-np.cos(k*eta)+np.sin(k*eta)/(k*eta))
max_k = gamma_k.max()
gamma_k /= max_k
ax1.plot(eta, gamma_k, "--", label="Analyt soln, k = k_c")
p = 0
q = 2*(p+3)
gamma_k2_approx = 2**(1/4 + 3/(2*q))*C*(3/2)**(3/2-3/q) * \
    M_GW**(-3/q)/(np.sqrt(k))*np.cos(k*eta)
max_k2_approx = gamma_k2_approx.max()
gamma_k2_approx /= max_k2_approx
# ax2.plot(eta, gamma_k2_approx, "--", label="Analyt soln approx, k = k_c")
gamma_k2 = C*(3/2)**(3/2-3/q)*q**(3/q)*scipy.special.gamma(1+3/q)/M_GW**(3/q) * \
    eta**(1/2)*scipy.special.jv(3/q, 3*(2/3)**(q/2)*M_GW/q*eta**(q/2))
max_k2 = gamma_k2.max()
gamma_k2 /= max_k2
ax2.plot(eta, gamma_k2, "--", label="Analyt soln exact, k = k_c")


H_square = np.vectorize(Hubble)(eta)**2
ang_freq_square = np.vectorize(ang_freq)(eta)**2
ax3.plot(eta, H_square, label=r'$H^2$'+f': {k}'+r' $Gpc^{-1}$')
ax3.plot(eta, ang_freq_square,
         label=r'$\omega^2$: '+f'{k}'+r' $Gpc^{-1}$')
eta_k = 0
k_idx = 0
for i in range(0, len(eta)):
    if H_square[i] <= ang_freq_square[i]:
        eta_k = eta[i]
        k_idx = i
        break

ax1.title.set_text(r'GR for $k \approx k_c$')
ax1.set_ylabel(r'$\gamma_k(\eta)$')
# ax1.set_xlabel(r'$\eta(Gpc)$')
ax1.set_xscale('log')
ax1.legend(loc='best')

ax2.title.set_text(r'MG for $k \approx k_c$')
ax2.set_ylabel(r'$\gamma_k(\eta)$')
# ax2.set_xlabel(r'$\eta(Gpc)$')
ax2.set_xscale('log')
ax2.legend(loc='best')

plt.axvline(x=eta_k, color="red", linestyle="dashed",
            linewidth=1, label="Horizon crossing")

ax3.title.set_text(r'Horizon Crossing')
ax3.set_xlabel(r'$\eta(Gpc)$')
ax3.set_xscale('log')
ax3.legend(loc='best')
'''

fig, (ax4) = plt.subplots(1, figsize=(22, 14))
eta = np.logspace(-8, .25, N)
a = np.vectorize(scale_fac)(eta)
aH = a*np.vectorize(Hubble)(eta)
ax4.plot(a, aH, label='aH', color='black')
ax4.plot(a, a*M_GW, '-.', label=r'$aM_{GW}$', color='green')
print(k_0)
plt.vlines(x=a_0, ymin=1e0, ymax=k_0, color="black", linestyle="dashed",
           linewidth=1)
ax4.text(a_0, .3, r"$a_0$", fontsize=15, weight="bold")

plt.vlines(x=a_c, ymin=1, ymax=k_c, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(a_c, .3, r"$a_c$", fontsize=15, weight="bold")
plt.hlines(y=k_0, xmin=1e-10,xmax=a_0, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(4e-11, k_0, r"$k_0$", fontsize=15, weight="bold")
plt.hlines(y=k_c, xmin=1e-10, xmax=a_c, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(4e-11, k_c, r"$k_c$", fontsize=15, weight="bold")

ax4.set_xlim(1e-10,5e0)
ax4.set_ylim(1e0, 5e10 )
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend(loc='best')


plt.savefig("emir/fig3.pdf")
plt.show()
