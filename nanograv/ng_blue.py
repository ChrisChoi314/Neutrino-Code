import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from blue_func import *

# Much of this code is taken from the NANOGrav collaboration's github page, where they have code that generates certain plots from their set of 4 (or 5?) papers.
hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]


plt.figure(figsize=(13.7, 4.2))

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
plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0), OMG_15.mean(axis=0) +
                 2*OMG_15.std(axis=0), color='orange', label='2$\sigma$ posterior of GWB', alpha=0.5)
plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 4*OMG_15.std(axis=0), OMG_15.mean(axis=0) +
                 4*OMG_15.std(axis=0), color='orange', label='4$\sigma$ posterior of GWB', alpha=0.3)

plt.plot(np.log10(freqs), np.log10(h**2*omega_GW(freqs, -15.6, 4.7)),
         linestyle='dashed', color='black', label='SMBHB spectrum')

# This part plots the energy densities of massive gravitons from the Mukohyama Blue tilted paper https://arxiv.org/pdf/1808.02381.pdf
H_inf = 1e8  # in GeV
tau_r = 5.494456683825391e-7  # calculated from equation (19)
a_r = 1/(tau_r*H_inf)


freqs = np.logspace(-19,5,num_freqs)
M_arr = np.logspace(-2, np.log10(1.5), 10)*H_inf
#M_GW = 1e-5*H_inf
tau_m = 6.6e21*tau_r
M_arr = np.array([.5])*H_inf
for M_GW in M_arr:
    plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)),
         color='blue', label=r'MG - Blue-tilted, $m = 0.3H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')
    print(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)[-1], M_GW)
plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_full(freqs, 0, H_inf, tau_r, tau_m)),
         color='red', label=r'MG - Blue-tilted, $m = 0.3H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')
print(h**2*omega_GW_full(freqs, 0, H_inf, tau_r, tau_m)[-1], 0)
BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between(np.log10(BBN_f), np.log10(BBN_f*0+h**2*1e-5),
                 np.log10(BBN_f * 0 + 1e1), alpha=0.5, color='orchid')
plt.text(-8.5, -5, r"BBN Bound", fontsize=15)

print(h**2)
# Plot Labels
plt.title(r'NANOGrav 15-year data and Time Dependent model')
plt.xlabel(r'log$_{10}(f$ Hz)')
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$')

plt.xlim(-9, -7)
plt.ylim(-13, -4)

plt.legend(loc='lower left').set_visible(False)
plt.grid(alpha=.2)

'''plt.clf()
idx_BBN = 0
for f in freqs:
    if f < f_BBN:
        idx_BBN +=1 


for M_GW in M_arr:
    plt.plot(freqs[idx_BBN:], np.trapz(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)[idx_BBN:] / freqs[idx_BBN:]),
         color='blue', label=r'MG - Blue-tilted, $m = 0.3H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')
    
BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between(BBN_f, BBN_f*0+1e-6,BBN_f * 0 + 1e1, alpha=0.5, color='orchid')

plt.title(r'BBN Bound on $\int_{f_{BBN}}^f df \frac{1}{f} h^2\Omega_{GW,0}(f)$')
plt.xlabel(r'log$_{10}(f$ Hz)')
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$')

plt.xscale('log')
plt.yscale('log')'''

plt.savefig('nanograv/ng_blue_figs/fig2.pdf')
plt.show()
