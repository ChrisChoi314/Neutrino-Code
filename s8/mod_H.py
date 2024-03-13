import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from scipy.integrate import quad
from s8_func import *


fs = 11
plt.rcParams.update({'font.size': fs})
N = 5000

r_rec = 144.57 # in Mpc, from Table 2 in Planck 2018 paper
theta_rec = 1.04119/100 # same as above, in radians presumably
a_rec = 1/(1 + 1089.80) # same as above
c_km = 299792458/1000 # in km/s

def integrand(a):
    return 1/np.sqrt(omega_R+omega_M*a+omega_L*a**4)

val, err = quad(integrand, a_rec, 1, args=())

calc_H0 = theta_rec/r_rec*val*c_km
print(f'val = {val}, err = {err}')
print(f"H_0 = {calc_H0}")
# Much of this code is taken from the NANOGrav collaboration's github page, where they have code that generates certain plots from their set of 4 (or 5?) papers.
hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]


#plt.figure(figsize=(13.7, 4.2))
plt.figure(figsize=(10, 4))

f = np.linspace(-9, math.log(3e-7, 10), N)
f = np.logspace(-8.6, -7, 30)
freqs_30 = f



def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)


num_freqs = 30
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
# Finally realized the log10_A and gamma I needed were in https://zenodo.org/records/8067506 in the
# NANOGrav15yr_CW-Analysis_v1.0.0/15yr_cw_analysis-main/data/15yr_quickCW_detection.h5 file.
A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))


for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))
#plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0)), color='orange', alpha=0.7)
#plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0)), color='orange', alpha=0.5)
#plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0)), color='orange', alpha=0.3)
#plt.plot(np.log10(freqs), np.log10(h**2*omega_GW(freqs, -15.6, 4.7)),
#         linestyle='dashed', color='black', label='SMBHB spectrum')

# This part plots the energy densities of massive gravitons from the Mukohyama Blue tilted paper https://arxiv.org/pdf/1808.02381.pdf
num_freqs = 1000
H_inf = 1e8  # in GeV
#a_r = 1/(tau_r*H_inf)
tau_r = 5.494456683825391e-7  # calculated from equation (19)

#f_UV = a_r*H_inf/(2*np.pi)

H_inf = .47
f_UV = 2e8*(H_inf/1e14)**.5
freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
M_arr = np.array([1.298])*H_inf
idx = 0
for M_GW in M_arr:
    Omega = h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)
    #plt.plot(freqs, (h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)),
    #     color='red', label=r"$M_{GW}$ = 1.298$H_{inf}$" + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    idx+=1

M_GW = 1.298*H_inf
omega_GW_0_tot, err = quad(omega_GW_full, 1e-19, f_UV , args=(M_GW, H_inf, tau_r, tau_m))
print(f'omega_GW_0 = {omega_GW_0_tot}, err = {err} ')

H_inf = 1e8
f_UV = 2e8*(H_inf/1e14)**.5
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
tau_m = 1e23*tau_r
M_GW = 0.8*H_inf
omega_GW0, err = quad(omega_GW_full, 1e-19, f_UV , args=(M_GW, H_inf, tau_r, tau_m))
print(f'omega_GW_0 = {omega_GW_0_tot}, err = {err} ')

omega_GW0 = 3.8e-6 # from https://physics.stackexchange.com/questions/661039/which-is-the-total-energy-density-constraint-for-the-gravitational-wave-backgrou

a_m = scale_fac(tau_m)

def D_M1(a):
    return 1/np.sqrt((omega_R+omega_GW0)+omega_M*a+omega_L*a**4)
def D_M2(a):
    return 1/np.sqrt(omega_R+(omega_M + omega_GW0)*a+omega_L*a**4)

k_eq = 0.010339
k_eq_sig = 0.000063
    

print(f'a_eq = {a_eq}, a_rec = {a_rec}, a_m = {a_m}')

z_eq = 3387 
z_eq_sig = 21
a_sig = z_eq_sig*(1/(1+z_eq)**2)

def Hub_fun1(a, omegaR, omegaM, omegaL, omegaGW):
    if (a < a_m):
        return H_0*np.sqrt((omegaR)/a**4+(omegaM+omegaGW)/a**3+omegaL - (omegaR + omegaM + omegaL + omegaGW - 1)/a**2)
    else: 
        return H_0*np.sqrt((omegaR + omegaGW)/a**4+omegaM/a**3+omegaL - (omegaR + omegaM + omegaL + omegaGW - 1)/a**2)
    
def Hub_fun(a, omegaR, omegaM, omegaL, omegaGW):
    if (a < a_m):
        return H_0*np.sqrt((omegaR)/a**4+(omegaM+omegaGW)/a**3+omegaL )
    else: 
        return H_0*np.sqrt((omegaR+omegaGW)/a**4+(omegaM)/a**3+omegaL )

y_upper = 1e8
y_lower = 1e0
plt.vlines(x=a_eq, ymin=y_lower, ymax=y_upper, color="black", linestyle="dashed",
            linewidth=1)
plt.text(a_eq,y_lower*100, r"$a_{eq}$", fontsize=15, weight="bold")

plt.vlines(x=a_rec, ymin=y_lower, ymax=y_upper, color="blue", linestyle="dashed",
            linewidth=1)
plt.text(a_rec,y_lower*100, r"$a_{CMB}$", fontsize=15, weight="bold")

plt.vlines(x=a_m, ymin=y_lower, ymax=y_upper, color="red", linestyle="dashed",
            linewidth=1)
plt.text(a_m,y_lower*100, r"$a_{m}$", fontsize=15, weight="bold")

a_l = (omega_M/omega_L)**(1/3)

plt.vlines(x=a_l, ymin=y_lower, ymax=y_upper, color="green", linestyle="dashed",
            linewidth=1)
plt.text(a_l,y_lower*100, r"$a_{M,\Lambda}$", fontsize=15, weight="bold")

a = np.logspace(-5,0,N)
H = Hubble_a(a)
plt.xscale('log')
plt.yscale('log')
plt.plot(a, a*np.vectorize(Hub_fun)(a, omega_R, omega_M, omega_L, 0), label='$\Lambda CDM$')
print(f'omega_GW from paper = {omega_GW0}')
quad_a = 1
quad_b = (omega_R + omega_M) - (a_sig*omega_R)
quad_c = 2*omega_M*omega_R - a_sig*omega_R**2
#omega_GW0 = (-quad_b + np.sqrt(quad_b**2 - 4*quad_a*quad_c))/(2*quad_a)
#omega_GW0 = z_eq_sig*omega_R
a_eq = 1/(z_eq + z_eq_sig)
omega_GW0 = omega_R*(1 + z_eq + z_eq_sig) - omega_M
print(f'aH(a_m) on left = {(a_m-.000000000000001)*Hub_fun((a_m-.000000000000001),omega_R, omega_M, omega_L, omega_GW0)}')
print(f'aH(a_m) in middle = {a_m*Hub_fun(a_m,omega_R, omega_M, omega_L, omega_GW0)}')
print(f'aH(a_m) on right = {(a_m+.000000000000001)*Hub_fun((a_m+.000000000000001),omega_R, omega_M, omega_L, omega_GW0)}')
Hmass = np.vectorize(Hub_fun)(a, omega_R, omega_M, omega_L, omega_GW0)

print(f'{(omega_R+omega_GW0)/a_m**4}')
print(f'{(omega_R)/a_m**4}')
plt.plot(a, a*Hmass, label='Massive Gravity')

plt.ylim(4e1,1e5)
plt.legend()
plt.show()
plt.savefig('s8/mod_H/fig0.pdf')


val1, err1 = quad(D_M1, a_rec, a_m, args=())
val2, err2 = quad(D_M2, a_m, 1, args=())

calc_H0 = theta_rec/r_rec*(val1 + val2)*c_km
print(f"H_0 = {calc_H0}")