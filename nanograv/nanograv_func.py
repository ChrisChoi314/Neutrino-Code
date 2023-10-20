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
omega_M = .3
omega_R = 8.5e-5
omega_L = .7

T = 6.58e-25
L = 1.97e-16
m2Gpc = 3.1e25


def hz2gpc(hz): return hz*(T/L)*m2Gpc
def gpc2hz(gpc): return gpc*(1/m2Gpc)*L/T


H_0 = 2.18e-18  # in Hz according to astronomy stack exchange https://astronomy.stackexchange.com/questions/49248/interpretation-of-hubble-constant-in-si-units
h = H_0/(100 / 3.085678e+19)
M_GW = H_0*1e10
k_0 = 1e10*H_0  # in Gpc
k_c = 1e4*H_0
eta_rm = .1
a_c = k_c / M_GW
a_0 = k_0 / M_GW
eta_0 = np.sqrt(4/(H_0**2*omega_M))
eta_rm = 4*np.sqrt(omega_R)/omega_M/H_0
eta_rm = 1.353824443067972e+16
eta_ml = 12.5
K = 0
a_lm = .754


def scale_fac(conf_time):
    if conf_time < 1e10:
        return H_0*np.sqrt(omega_R)*conf_time
    else:
        return (((H_0*omega_M)/2 * (conf_time + 2*np.sqrt(omega_R)/(H_0*omega_M)))**2 - omega_R) / omega_M

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
