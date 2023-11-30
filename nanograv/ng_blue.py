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

H_inf = 5.2
f_UV = 2e8*(H_inf/1e14)**.5
freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
M_arr = np.array([1.251042105263158])*H_inf
idx = 0
for M_GW in M_arr:
    Omega = h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)
    plt.plot(freqs, (h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)),
         color='blue', label=r"$M_{GW}$ = 1.251$H_{inf}$" + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    idx+=1

H_inf = 5e1
f_UV = 2e8*(H_inf/1e14)**.5
freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
M_arr = np.array([1.201])*H_inf
idx = 0
for M_GW in M_arr:
    Omega = h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)
    plt.plot(freqs, (h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)),
         color='green', label=r"$M_{GW}$ = 1.201$H_{inf}$" + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    idx+=1

BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between(BBN_f, BBN_f*0+h**2*1e-5, BBN_f*0 + 1e1,alpha=0.5, color='orchid')
plt.text(10**-7.5, 1e-5, r"BBN Bound", fontsize=15)

plt.xlabel(r'$f$ [Hz]')
plt.ylabel(r'$h_0^2\Omega_{GW}$')

plt.xlim(1e-9, 1e-7)
plt.ylim(1e-11, 1e-4)

plt.legend(loc='upper left')
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

#plt.savefig('nanograv/ng_blue_figs/fig3.pdf')

plt.clf()

plt.figure(figsize=(8,6))

BBN = h**2*1e-5

CMB_f = np.logspace(-16.7, -16)
plt.fill_between(CMB_f,CMB_f*0+h**2*1e-15, CMB_f *
                 0 + 1e6, alpha=0.5, color='blue')
plt.text(10**-18.5, 1e-13, r"CMB", fontsize=15)

num_freqs = 30
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)

for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0)), color='orange', alpha=0.7)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0)), color='orange', alpha=0.5)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0)), color='orange', alpha=0.3)

num_freqs = 1000
tau_r = 5.494456683825391e-7  # calculated from equation (19)

H_inf_arr = [.47, 5.2, 5e1]
M_arr = [1.298, 1.251, 1.201]
color_arr = ['red', 'blue', 'green']
idx = 0
for M_GW in M_arr:
    H_inf = H_inf_arr[idx]
    M_GW *= H_inf
    f_UV = 2e8*(H_inf/1e14)**.5
    freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
    tau_m = 1e10*(H_inf/1e14)**-2*tau_r
    Omega = np.where(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)< BBN, np.nan, BBN)
    plt.plot(freqs, h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m),
         color=color_arr[idx], label=r"$M_{GW}$ = "+f'{M_arr[idx]}'+r'$H_{inf}$' + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    plt.plot(freqs, Omega, color=color_arr[idx], linestyle='dashed')
    idx+=1
plt.text(1e-4, 1e-7, r"With suppression", fontsize=15)
plt.text(1e-9, 1e3, r"Without suppression", fontsize=15)

BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between((BBN_f), (BBN_f*0+h**2*1e-5),
                 (BBN_f * 0 + 1e10), alpha=0.5, color='orchid')
plt.text(10**-13.5, 1e0, "BBN\nBound", fontsize=15)

outfile = np.load('emir/emir_hasasia/nanograv_sens_full.npz')

freq_NG = []
omega_GW_NG = []
idx = 0
with open('blue/data/sensitivity_curves_NG15yr_fullPTA.txt', 'r') as file:
    for line in file:
        if idx != 0:
            elems = line.strip("\r\n").split(",")
            freq_NG.append(float(elems[0]))
            omega_GW_NG.append(float(elems[3]))
        idx +=1

f_nanoGrav = outfile['freqs']
nanoGrav_sens = outfile['sens']
plt.plot(f_nanoGrav, nanoGrav_sens, color='darkturquoise')

plt.text(5e-7, 1e-14, "NANOGrav\nSensitivity", fontsize=15)

plt.xlabel(r'$f$ [Hz]')
plt.ylabel(r'$h_0^2\Omega_{GW}$')
plt.xscale('log')
plt.yscale('log')
# plt.legend(loc='lower right')
plt.grid(alpha=.2)
plt.xlim(1e-20,1e3)
plt.ylim(1e-22,1e6)
plt.savefig('nanograv/ng_blue_figs/fig6.pdf')
plt.show()