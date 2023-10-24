import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

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
plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0), OMG_15.mean(axis=0) +
                 2*OMG_15.std(axis=0), color='orange', label='2$\sigma$ posterior of GWB', alpha=0.5)
plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 4*OMG_15.std(axis=0), OMG_15.mean(axis=0) +
                 4*OMG_15.std(axis=0), color='orange', label='4$\sigma$ posterior of GWB', alpha=0.3)

plt.plot(np.log10(freqs), np.log10(h**2*omega_GW(freqs, -15.6, 4.7)),
         linestyle='dashed', color='black', label='SMBHB spectrum')

# This part plots the energy densities of massive gravitons from the Mukohyama Blue tilted paper https://arxiv.org/pdf/1808.02381.pdf
num_freqs = 1000
H_inf = 1e8  # in GeV
#a_r = 1/(tau_r*H_inf)
tau_r = 5.494456683825391e-7  # calculated from equation (19)

#f_UV = a_r*H_inf/(2*np.pi)

#freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
#M_GW = 1e-5*H_inf
tau_m = 6.6e21*tau_r
#tau_m = 2e22*tau_r
#tau_m = 1e24*tau_r

H_inf = 5e2
f_UV = 2e8*(H_inf/1e14)**.5
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
M_arr = np.logspace(-6, np.log10(1.5), 10)*H_inf # this is for generating the plot in fig1 and fig2 in nanograv/ng_blue_figs/
M_arr_coeff = np.linspace(.35, .75, 20) + .5642
print(M_arr_coeff)
M_arr = M_arr_coeff*H_inf 
M_arr = np.array([1.1422])*H_inf
idx = 0
for M_GW in M_arr:
    plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)),
         color='red', label=r"$M_{GW}$ = 1.14$H_{inf}$" + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    #plt.text(np.log10(freqs)[int(num_freqs/2)], np.log10(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m))[int(num_freqs/2)], r"$M_{GW}$ = "+f'{M_arr_coeff[idx]}' + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$ = "+f'{H_inf} GeV', fontsize=8)
    idx+=1

H_inf = 5e0
f_UV = 2e8*(H_inf/1e14)**.5
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
M_arr = np.logspace(-6, np.log10(1.5), 10)*H_inf # this is for generating the plot in fig1 and fig2 in nanograv/ng_blue_figs/
M_arr_coeff = np.linspace(.35, .75, 20) + .5642
print(M_arr_coeff)
M_arr = M_arr_coeff*H_inf 
M_arr = np.array([1.251042105263158])*H_inf
idx = 0
for M_GW in M_arr:
    plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)),
         color='blue', label=r"$M_{GW}$ = 1.25$H_{inf}$" + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    #plt.text(np.log10(freqs)[int(num_freqs/2)], np.log10(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m))[int(num_freqs/2)], r"$M_{GW}$ = "+f'{M_arr_coeff[idx]}' + r", $H_{inf}$ = "+f'{H_inf} GeV', fontsize=8)
    idx+=1
#plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_full(freqs, 0, H_inf, tau_r, tau_m)),
#         color='red', label=r'MG - Blue-tilted, $m = 0.3H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')
BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between(np.log10(BBN_f), np.log10(BBN_f*0+h**2*1e-5),
                 np.log10(BBN_f * 0 + 1e1), alpha=0.5, color='orchid')
plt.text(-7.5, -5, r"BBN Bound", fontsize=15)

# Plot Labels
# because apparently, no papers title their plots so I won't title mine because I like to be a conformist
# plt.title(r'NANOGrav 15-year data and SFM $\frac{\tau_m}{\tau_r} =$ 6.6e21')
plt.xlabel(r'log$_{10}(f/$Hz)')
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$')

#plt.xlim(-9, -7)
#plt.ylim(-11, -4)

plt.legend(loc='upper left')
plt.grid(alpha=.2)

#plt.ylim(-80,1)

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

plt.savefig('nanograv/ng_blue_figs/fig4.pdf')
plt.show()
