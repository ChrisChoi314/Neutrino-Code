from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from blue_func import *

N = 2000

plt.style.use('dark_background')
fig, (ax1) = plt.subplots(1, figsize=(10, 8))

H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14
a_r = 1/(tau_r*H_inf)


def omega_GW_approx(f, m):
    f_8 = f/(2e8)
    nu = (9/4 - m**2 / H_inf**2)**.5
    k = f*2*np.pi
    # return tau_m/tau_r*(k*tau_r)**(3-2*nu)*1e-15 *H_14**2
    return 1e-15 * tau_m/tau_r * H_14**(nu+1/2)*f_8**(3-2*nu)


def P_T_prim(k):
    n_T = .6
    # from page 3 of https://arxiv.org/pdf/1407.4785.pdf
    k_ref = gpc2hz(0.01*1e3)
    r = .2
    P_gamma_k_ref = 2.2e-9  # also from same page in same paper
    A_T_at_k_ref = r*P_gamma_k_ref
    return A_T_at_k_ref*(k/k_ref)**n_T


def T_1_sq(x):
    return 1+1.57*x+3.42*x**2


def T_2_sq(x):
    return 1/(1-.22*x**1.5+.65*x**2)


def g_star_T_in(k, g_0):
    g_max = 106.75
    A = (-1 - 10.75/g_0)/(-1+10.75/g_0)
    B = (-1 - g_max/10.75)/(-1+g_max/10.75)
    return g_0*((A+np.tanh(-2.5*np.log10(k/(2*np.pi)/(2.5e-12))))/(A+1))*((B+np.tanh(-2.0*np.log10(k/(2*np.pi)/(6e-9))))/(B+1))


def spherical_bess_approx(n, x):
    return 1/(np.sqrt(2)*x)


def T_T_sq(k):
    g_star_0 = 3.36
    g_star_s_0 = 3.91
    tau_0 = eta_0
    T_R = 1e12  # in GeV
    k_mpc = 0.01
    k_eq = 7.1e-2*omega_M*h**2
    g_star_T_R = 106.75*g_star_s_0/g_star_0
    k_R = 1.7e14*(g_star_T_R/106.75)**(1/6)*(T_R/1e7)
    x_eq = k_mpc/k_eq
    x_R = k_mpc/k_R
    func_to_use = scipy.special.spherical_jn
    func_to_use = spherical_bess_approx
    return omega_M**2*(g_star_T_in(k, g_star_0)/g_star_0)*(g_star_s_0/g_star_T_in(k, g_star_s_0))**(4/3)*(3*(func_to_use(1, k*tau_0))/(k*tau_0))**2*T_1_sq(x_eq)*T_2_sq(x_R)


def P_T(k):
    return T_T_sq(k)*P_T_prim(k)


def omega_GW_massless(k):
    return 1/12*(k/(a_0*H_0))**2*P_T(k)


def omega_GW_full(f, m):
    f_8 = f/(2e8)
    nu = (9/4 - m**2 / H_inf**2)**.5
    k = f*2*np.pi
    return omega_GW_massless(k)
    return tau_m/tau_r*(k*tau_r)**(3-2*nu)*omega_GW_massless(k)


# fig 2 of https://arxiv.org/pdf/1001.3161.pdf
linestyle_arr = ['dotted', 'dashdot', 'dashed', 'solid']
M_arr = [2*np.pi*1e-8, 2*np.pi*1e-7, 2*np.pi*1e-6]
M_arr = [4.3e-23*1e-9, 1.2e-22*1e-9, 0]
M_arr = [k / hbar for k in M_arr]

M_arr = [.5*H_inf, .8*H_inf]
tau_m_r_ratio = [1e10, 1e15, 1e21]
colors = ['white', 'cyan', 'yellow']
linestyle_arr = ['solid', 'dashed']
text = ['m = 0.5$H_{inf}$', 'm = 0.8$H_{inf}$']
text2 = ['1e10', '1e15', '1e21']
idx = 0
idx2 = 0
f = np.logspace(-19, 5, N)


for ratio in tau_m_r_ratio:
    for M_GW in M_arr:
        tau_m = tau_m_r_ratio[idx2]
        ax1.plot(f, omega_GW_approx(
            f, M_GW), linestyle=linestyle_arr[idx], color=colors[idx2], label=text[idx] + r', $\frac{\tau_m}{\tau_r} = $'+text2[idx2])
        idx += 1
        ax1.plot(f, omega_GW_full(f, M_GW), color='red', label='With transfer')
    idx = 0
    idx2 += 1
M_GW = 0
tau_m = 1
ax1.plot(f, omega_GW_approx(f, M_GW), color='green', label='GR')

num = 1e-8*tau_r/tau_m
BBN_f = np.logspace(-10, 9)
ax1.fill_between(BBN_f, BBN_f*0+h**2*1e-5, BBN_f *
                 0 + 1e1, alpha=0.5, color='orchid')
ax1.text(1e-12, 1e-5, r"BBN", fontsize=15)

CMB_f = np.logspace(-17, -16)
ax1.fill_between(CMB_f, CMB_f*0+h**2*1e-15, CMB_f *
                 0 + 1e1, alpha=0.5, color='blue')
ax1.text(5e-16, 1e-13, r"CMB", fontsize=15)

ax1.set_xlim(1e-19, 1e9)
# ax1.set_ylim(1e-22, 1e1)
ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$\Omega_{GW}$')
plt.title('Gravitational Energy Density')

ax1.legend(loc='upper right')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend().set_visible(False)
plt.savefig("blue/blue_emir_figs/fig4.pdf")
plt.show()
