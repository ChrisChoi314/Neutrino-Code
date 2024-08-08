import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

fs = 11.5
plt.rcParams.update({'font.size': fs})

# Much of this code is taken from the NANOGrav collaboration's github page, where they have code that generates certain plots from their set of 4 (or 5?) papers.
hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]

N = 1000
f = np.linspace(-9, math.log(3e-7, 10), N)
f = np.logspace(-8.6, -7, 30)
freqs_30 = f


def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)

#plt.figure(figsize=(10,4))

fig, axs = plt.subplots(2, 1, figsize = (10,7), sharex=True)
fig.subplots_adjust(hspace=0)

BBN = h**2*1e-5

CMB_f = np.logspace(-16.7, -16)
plt.fill_between(CMB_f,CMB_f*0+h**2*1e-15, CMB_f *
                 0 + 1e15, alpha=0.5, color='blue')
plt.text(10**-18.5, 1e-13, r"CMB", fontsize=15)

dir = '30f_fs{hd}_ceffyl'
freqs = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/freqs.npy')
num_freqs = 30

A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

slope = (14.8-7.5)/(np.log10(6e-8) - np.log10(2e-9))
b = -7.6 - slope*np.log10(6e-8)
lower = slope*np.log10(freqs) + b
lower = 10**lower

upper = np.array([-8.8, -8.3, -8.1, -7.7, -7.5, -7., -6.6, -6.5, -6.7, -6.2, -6, -6.05, -5.8, -5.6, -5.1, -3, -4.3, -4.85, -4.8, -5.1, -5.15, -4.85, -4.5, -4.6, -4.4, -4.3, -4.6, -4.3, -4.5, -4])
upper = 10**upper

#plt.fill_between(freqs, lower, upper, color='black', alpha=0.1)
for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))
for i in range(2):
    axs[i].fill_between(freqs, lower, upper, color='black', alpha=0.1)
    axs[i].fill_between(freqs, 10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0)), color='orange', alpha=0.7)
    axs[i].fill_between(freqs, 10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0)), color='orange', alpha=0.5)
    axs[i].fill_between(freqs, 10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0)), color='orange', alpha=0.3)

    axs[i].plot(freqs,h**2*omega_GW(freqs, -15.6, 4.7),
            linestyle='dashed', color='black')

num_freqs = 1000
tau_r = 5.494456683825391e-7  # calculated from equation (19)

a_r = 1/(tau_r*1e8)

color_arr = ['red', 'blue', 'green','m', 'y']
H_inf_arr = [1.7, 8, 5e1, 1e8,1e8]
M_arr = [1.298, 1.251, 1.201, .3, .8]
H_inf_arr = np.array(H_inf_arr)
tau_r_arr = 1/(a_r*H_inf_arr)
tau_m_arr = [1e10*(H_inf_arr[0]/1e14)**-2,1e10*(H_inf_arr[1]/1e14)**-2,1e10*(H_inf_arr[2]/1e14)**-2,1e21, 1e23] *tau_r_arr
tau_m_arr = np.array([1e27,1e27,1e27,2e21,1e21])*tau_r_arr
idx = 0
for M_GW in M_arr:
    H_inf = H_inf_arr[idx]
    M_GW *= H_inf
    tau_r = tau_r_arr[idx]
    freqs = np.logspace(np.log10(freqs[0]),np.log10(freqs[-1]),num_freqs)
    tau_m = tau_m_arr[idx]
    print(f'a_r*H_inf: {a_r*H_inf}')
    print(f'a_r: {a_r}')
    print(f'H_inf: {H_inf}')

    Omega = np.where(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)< BBN, np.nan, BBN)
    if idx <= 2:
        axs[0].plot(freqs, h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m),color=color_arr[idx], label=r"$M_{\mathrm{GW}}$ = "+f'{M_arr[idx]}'+r'$H_{\mathrm{inf}}$' + r", $H_{\mathrm{inf}}$ = "+f'{H_inf} GeV' + r", $\tau_m = 10^{27}\tau_r$")
    elif idx == 3:
        axs[1].plot(freqs, h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m),color=color_arr[idx], label=r"$M_{\mathrm{GW}}$ = "+f'{M_arr[idx]}'+r'$H_{\mathrm{inf}}$' + r", $H_{\mathrm{inf}} = 10^8$ GeV" + r", $\tau_m = 2\times 10^{21}\tau_r$")
    else:
        axs[1].plot(freqs, h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m),color=color_arr[idx], label=r"$M_{\mathrm{GW}}$ = "+f'{M_arr[idx]}'+r'$H_{\mathrm{inf}}$' + r", $H_{\mathrm{inf}} = 10^8$ GeV" + r", $\tau_m = 10^{21}\tau_r$")
    idx+=1
axs[0].text(1e-4, 1e-7, r"With suppression", fontsize=15)
axs[0].text(1e-9, 1e3, r"Without suppression", fontsize=15)

BBN_f = np.logspace(np.log10(f_BBN), 9)
for i in range(2):
    axs[i].fill_between((BBN_f), (BBN_f*0+h**2*1e-5),
                    (BBN_f * 0 + 1e15), alpha=0.5, color='orchid')
    axs[i].text(6e-8, 1e-6, "BBN\nBound", fontsize=15)

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
for i in range(2):
    #plt.plot(f_nanoGrav, nanoGrav_sens, color='darkturquoise')
    #plt.text(5e-7, 1e-14, "NANOGrav\nSensitivity", fontsize=15)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_ylabel(r'$h_0^2\Omega_{\mathrm{GW}}$')
    axs[i].grid(which='major', alpha=.2)
    axs[i].grid(which='minor', alpha=.2)
    axs[i].legend(loc='upper left')
axs[0].set_ylim(1e-11, 1e-5)
axs[1].set_ylim(1e-14, 1e-4)    
axs[0].set_yticks(np.array([1e-10,1e-8, 1e-6 ]))
axs[1].set_yticks(np.array([1e-12,1e-10,1e-8,1e-6,1e-4]))

plt.xlabel(r'$f$ [Hz]')
plt.xlim(1e-9,1e-7)
#plt.savefig('nanograv/bbn_both_figs/fig0.pdf')
plt.show()