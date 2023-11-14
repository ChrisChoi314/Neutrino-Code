from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from nanograv_func import *


def P_T_prim(k):
    n_T = 0
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


def T_3_sq(x):
    return 1+.59*x+.65*x**2


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
    tau_0 = eta_0/1e-3
    T_R = 1e12  # in GeV
    T_R = 1e14
    k_mpc = hz2gpc(k)/1e3
    k_eq = 7.1e-2*omega_M*h**2
    g_star_T_R = 106.75*g_star_s_0/g_star_0
    k_R = 1.7e14*(g_star_T_R/106.75)**(1/6)*(T_R/1e7)
    x_eq = k_mpc/k_eq
    x_R = k_mpc/k_R
    func_to_use = spherical_bess_approx
    extra_fac = 25 # this is to make the results identical to Fig 1 in https://arxiv.org/pdf/1808.02381.pdf
    #extra_fac *= 5.68e11
    return omega_M**2*(g_star_T_in(k, g_star_0)/g_star_0)*(g_star_s_0/g_star_T_in(k, g_star_s_0))**(4/3)*(3*(func_to_use(1, k*tau_0))/(k*tau_0))**2*T_1_sq(x_eq)*T_2_sq(x_R)*extra_fac


def P_T(k):
    return T_T_sq(k)*P_T_prim(k)


def omega_GW_massless(k):
    return 1/12*(k/(a_0*H_0))**2*P_T(k)


def omega_GW_full(f, m, hinf, tr, tm):
    nu = (9/4 - m**2 / hinf**2)**.5
    k = f*2*np.pi
    return tm/tr*(k*tr)**(3-2*nu)*omega_GW_massless(k)