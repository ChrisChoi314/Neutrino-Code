import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
from scipy.integrate import odeint
from math import log10, floor

c_g = 1  # speed of graviton
hbar = 6.582119569e-25  # GeV / Hz
c = 3e8  # m * Hz * (1Gpc / 3.086e25m) = Gpc * Hz
M_pl = 1.22089e19  # GeV
omega_M = .3111
omega_R = 9.182e-5
omega_L = .6889

T = 6.58e-25
L = 1.97e-16
m2Gpc = 3.1e25


def hz2gpc(hz): return hz*(T/L)*m2Gpc
def gpc2hz(gpc): return gpc*(1/m2Gpc)*L/T

def gev2hz(gev): return gev/(1.52e24)

h = .6766 # according to Planck 2018 TT, TE, EE + lowE + lensing + BAO data
#H_0 = 100*h*3.2404407e-20 # in Hz
H_0 = 67.66 
M_GW = H_0*1e10
k_0 = 1e10*H_0  # in Gpc
k_c = 1e4*H_0
a_c = k_c / M_GW
a_0 = k_0 / M_GW
eta_0 = np.sqrt(4/(H_0**2*omega_M))
eta_rm = 4*np.sqrt(omega_R)/omega_M/H_0
eta_rm = 1.353824443067972e+16
eta_ml = 12.5
K = 0
a_lm = .754
G = 6.67e-11

f_BBN = 1.5e-11 # in Hz according to 22.290 of maggiore vol 2 
f_yr = 1/(365*24*3600)

def scale_fac(conf_time):
    H0 = 100*h*3.2404407e-20
    if conf_time < 1e10:
        return H0*np.sqrt(omega_R)*conf_time
    else:
        return (((H0*omega_M)/2 * (conf_time + 2*np.sqrt(omega_R)/(H0*omega_M)))**2 - omega_R) / omega_M

def conf_time(a):
    if a < 1e-8:
        return a/(H_0*np.sqrt(omega_R))
    else:
        return (2*np.sqrt(omega_M*a + omega_R)/(H_0*omega_M)) - 2*np.sqrt(omega_R)/(H_0*omega_M)


def integrand(a):
    return 1 / (a**2*H_0*np.sqrt(omega_M/a**3 + omega_R/a**4 + omega_L))


def conf_time_anal(a):
    return scipy.integrate.quad(integrand, 0, a)[0]


def d_scale_fac_dz(conf_time):
    if conf_time < 1e10:
        return H_0*np.sqrt(omega_R)
    else:
        #   return 2*H_0**2*.25*omega_M*conf_time
        return 2*H_0**2*.25*omega_M * (conf_time + 2*np.sqrt(omega_R)/(H_0*omega_M))


def d_scale_fac_dz2(conf_time):
    if conf_time < 1e10:
        return 0.
    else:
        return 2*H_0**2*.25*omega_M


def diffeqMG(M, u, k):
    r = [M[1], -((c_g*k)**2 + (scale_fac(u)*M_GW)**2 -
                 d_scale_fac_dz2(u) / scale_fac(u) + 2*K*c_g**2) * M[0]]
    return r


def diffeqGR(M, u, k):
    r = [M[1], -((c_g*k)**2 - d_scale_fac_dz2(u) /
                 scale_fac(u) + 2*K*c_g**2) * M[0]]
    return r


def diffeqMode(M, u, k, M_gw):
    r = [M[1], -2*M[1]*d_scale_fac_dz(u)/scale_fac(u) - ((c_g*k)**2 + (scale_fac(u)*M_gw)**2) * M[0]]
    return r


def Hubble(conf_t):
    return H_0*np.sqrt(omega_M / scale_fac(conf_t)**3 + omega_R / scale_fac(conf_t)**4 + omega_L)


def ang_freq(conf_t, k):
    return np.sqrt(k**2/scale_fac(conf_t)**2 + M_GW**2)


def Hubble_a(x):
    return H_0*np.sqrt(omega_M / x**3 + omega_R / x**4 + omega_L)


a_eq = omega_R/omega_M
H_eq = Hubble_a(a_eq)
k_eq = a_eq*H_eq


def ang_freq_a(x, k):
    return np.sqrt(k**2/x**2 + M_GW**2)


def ang_freq_GR(conf_t, k):
    return np.sqrt(k**2/scale_fac(conf_t)**2)


def normalize(array):
    max = array.max()
    array /= max
    return array


def normalize_0(array):
    max = array[0]
    array /= max
    return array


def ak(k):
    n_threads = 16
    with Pool(n_threads) as p:
        return np.where(k >= 0, p.map(solve_one, k), 1.)


def solve_one(k):
    from sympy import nroots, symbols, re, im
    x, y = symbols('x y')
    sols = list(nroots((y**2*x**2 + x**4*M_GW**2 - H_0**2 * (omega_M *
                x + omega_R + x**4*omega_L)).subs({y: k}), n=100, maxsteps=10000))
    for sol in sols:
        if abs(im(sol)) < 1e-20 and re(sol) >= 0:
            return float(re(sol))
    raise RuntimeError("no solution")


def give_eta(a):
    if a >= a_eq:
        return np.sqrt(4*a / (H_0**2 * omega_M))
    else:
        return a/(H_0*np.sqrt(omega_R))
    

def round_it(x, sig):
    if x == 0:
        return 0
    return round(x, sig-int(floor(log10(abs(x))))-1)


def powerlaw(f, log10_A, gamma):
    return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * f_yr**(gamma-3) * f**(-gamma) * f[0])


## From ng_blue_func.py from nanograv folder 

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