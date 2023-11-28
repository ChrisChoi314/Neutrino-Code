import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

fs = 12
plt.rcParams.update({'font.size': fs})
# Much of this code is taken from the NANOGrav collaboration's github page, where they have code that generates certain plots from their set of 4 (or 5?) papers.
f_yr = 1/(365*24*3600)
gamma_12p5 = 13/3
gamma_15 = 3.2  # from page 4 of https://iopscience.iop.org/article/10.3847/2041-8213/acdac6/pdf
A_12p5 = -16
A_15 = -14.19382002601611

gamma_cp = gamma_15
A_cp = A_15

# # Common Process Spectral Model Comparison Plot (Figure 1) # #

# Definition for powerlaw and broken powerlaw for left side of Figure 1


def powerlaw_vec(f, f_0, log10_A=A_cp, gamma=gamma_cp):
    return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma-3) * f**(-gamma) * f_0)


def powerlaw(f, log10_A=A_cp, gamma=gamma_cp):
    return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma-3) * f**(-gamma) * f[0])

hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]
    log_likelihood = f['log_likelihood'][:, burnin::extra_thin]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]
burnin = 0
thin = 1

plt.figure(figsize=(10,4.5))
#plt.figure(figsize=(20,8))

N = 1000
f = np.linspace(-9, math.log(3e-7, 10), N)
f = np.logspace(-8.6, -7, 30)
freqs_30 = f


def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)

num_freqs = 30
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

PL = np.zeros((67, num_freqs))
for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0)), color='orange', alpha=0.7)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0)), color='orange', alpha=0.5)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0)), color='orange', alpha=0.3)

plt.plot(freqs, h**2*omega_GW(freqs, -15.6, 4.7),
         linestyle='dashed', color='black')

# This part plots the energy densities of massive gravitons from the Mukohyama Blue tilted paper https://arxiv.org/pdf/1808.02381.pdf
tau_r = 5.494456683825391e-7

H_inf_arr = [.47, 5.2, 5e1]
M_arr = [1.298, 1.251, 1.201]
color_arr = ['red', 'blue', 'green']
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
idx = 0
for M_GW in M_arr:
    H_inf = H_inf_arr[idx]
    M_GW *= H_inf
    tau_m = 1e10*(H_inf/1e14)**-2*tau_r
    plt.plot(freqs, h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m),
         color=color_arr[idx], label=r"$M_{GW}$ = "+f'{M_arr[idx]}'+r'$H_{inf}$' + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    idx+=1

BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between(BBN_f, BBN_f*0+h**2*1e-5,
                 BBN_f * 0 + 1e1, alpha=0.5, color='orchid')
plt.text(10**(-7.5), 1e-6, r"BBN Bound", fontsize=15)


# Plot Labels
plt.ylim(1e-11, 1e-5)
plt.xlim(1e-9,1e-7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$f$ [Hz]')
plt.ylabel(r'$h_0^2\Omega_{GW}$')
plt.legend(loc='upper left')
plt.grid(alpha=.2)
plt.savefig('nanograv/SFM_compare_figs/fig0.pdf')
plt.show()
