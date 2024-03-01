import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from scipy.integrate import quad
from s8_func import *

fs = 12
plt.rcParams.update({'font.size': fs})

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

N = 1000
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
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0)), color='orange', alpha=0.7)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0)), color='orange', alpha=0.5)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0)), color='orange', alpha=0.3)
plt.plot(np.log10(freqs), np.log10(h**2*omega_GW(freqs, -15.6, 4.7)),
         linestyle='dashed', color='black', label='SMBHB spectrum')

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
    plt.plot(freqs, (h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)),
         color='red', label=r"$M_{GW}$ = 1.298$H_{inf}$" + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    idx+=1

M_GW = 1.298*H_inf
omega_GW_0_tot, err = quad(omega_GW_full, 1e-19, f_UV , args=(M_GW, H_inf, tau_r, tau_m))
print(f'omega_GW_0 = {omega_GW_0_tot}, err = {err} ')

H_inf = 1e8
f_UV = 2e8*(H_inf/1e14)**.5
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
M_GW = 0.8*H_inf
omega_GW0, err = quad(omega_GW_full, 1e-19, f_UV , args=(M_GW, H_inf, tau_r, tau_m))
print(f'omega_GW_0 = {omega_GW_0_tot}, err = {err} ')

omega_GW0 = 3.8e-6 # from https://physics.stackexchange.com/questions/661039/which-is-the-total-energy-density-constraint-for-the-gravitational-wave-backgrou

a_m = scale_fac(tau_m)

def D_M1(a):
    return 1/np.sqrt((omega_R+omega_GW0)+omega_M*a+omega_L*a**4)
def D_M2(a):
    return 1/np.sqrt(omega_R+(omega_M + omega_GW0)*a+omega_L*a**4)


val1, err1 = quad(D_M1, a_rec, a_m, args=())
val2, err2 = quad(D_M2, a_m, 1, args=())

calc_H0 = theta_rec/r_rec*(val1 + val2)*c_km
print(f"H_0 = {calc_H0}")