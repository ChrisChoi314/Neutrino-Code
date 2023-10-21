import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from blue_func import *

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


# determine placement of frequency components
Tspan = 12.893438736619137 * (365 * 86400)  # psr.toas.max() - psr.toas.min() #
freqs_30 = 1.0 * np.arange(1, 31) / Tspan


chain_DE438_30f_vary = np.loadtxt(
    './blue/data/12p5yr_DE438_model2a_cRN30freq_gammaVary_chain.gz', usecols=[90, 91, 92], skiprows=25000)
chain_DE438_30f_vary = chain_DE438_30f_vary[::4]

# Pull MLV params
DE438_vary_30cRN_idx = np.argmax(chain_DE438_30f_vary[:, -1])

# Make MLV Curves
PL_30freq = powerlaw(
    freqs_30, log10_A=chain_DE438_30f_vary[:, 1][DE438_vary_30cRN_idx], gamma=chain_DE438_30f_vary[:, 0][DE438_vary_30cRN_idx])
PL_30freq_num = int(chain_DE438_30f_vary[:, 0].shape[0] / 5.)
PL_30freq_array = np.zeros((PL_30freq_num, 30))

gamma_arr = np.zeros((PL_30freq_num, 30))
A_arr = np.zeros((PL_30freq_num, 30))
for ii in range(PL_30freq_num):
    PL_30freq_array[ii] = np.log10(powerlaw(
        freqs_30, log10_A=chain_DE438_30f_vary[ii*5, 1], gamma=chain_DE438_30f_vary[ii*5, 0]))
    A_arr[ii] = chain_DE438_30f_vary[ii*5, 1]
    gamma_arr[ii] = chain_DE438_30f_vary[ii*5, 0]


hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]
    # samples = f['samples_cold'][...]
    log_likelihood = f['log_likelihood'][:, burnin::extra_thin]
    # log_likelihood = f['log_likelihood'][...]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]
burnin = 0
thin = 1


# Make Figure
# plt.style.use('dark_background')
plt.figure(figsize=(12, 5))

# Left Hand Side Of Plot

# plt.semilogx(freqs_30, (PL_30freq_array.mean(axis=0)), color='C2', label='PL (30 freq.)', ls='dashdot')
# plt.fill_between(freqs_30, (PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0)), (PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0)), color='C2', alpha=0.15)


# This commented out portion was my attempt to get the violin plots for the 12.5 and 15 year data set for the HD_correlated free spectral process from
# Fig 1 of https://iopscience.iop.org/article/10.3847/2041-8213/abd401/pdf and Fig 3 of https://iopscience.iop.org/article/10.3847/2041-8213/acdc91/pdf#bm_apjlacdc91eqn73
# respectively. I was able to successfully get the 12.5 one, as seen in blue/nanograv_masses_figs/fig2.pdf
'''
chain_DE438_FreeSpec = np.loadtxt('blue/data/12p5yr_DE438_model2a_PSDspectrum_chain.gz', usecols=np.arange(90,120), skiprows=30000)
print(chain_DE438_FreeSpec.shape)
chain_DE438_FreeSpec = chain_DE438_FreeSpec[::5]
print(chain_DE438_FreeSpec.shape)
vpt = plt.violinplot(chain_DE438_FreeSpec, positions=(freqs_30), widths=0.05*freqs_30, showextrema=False)
for pc in vpt['bodies']:
    pc.set_facecolor('k')
    pc.set_alpha(0.3)

# this is with the 15 year data set

dir = '30f_fs{hd}_ceffyl'
dir = '30f_fs{hd+mp+dp}_ceffyl_hd-only'
dir = '30f_fs{cp}_ceffyl'
dir = '30f_fs{hd+mp+dp+cp}_ceffyl_hd-only'
freqs = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/freqs.npy')
rho = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/log10rhogrid.npy')
density = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/density.npy')
bandwidth = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/bandwidths.npy')

density = np.transpose(density[0])

vpt = plt.violinplot(density,
               positions=(freqs),
                widths=0.05*freqs_30, showextrema=False)
for pc in vpt['bodies']:
    pc.set_facecolor('k')
    pc.set_alpha(0.3)
'''

N = 1000
f = np.linspace(-9, math.log(3e-7, 10), N)
f = np.logspace(-8.6, -7, 30)
freqs_30 = f


def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)


OMG_30freq_array = np.zeros((PL_30freq_num, 30))
for ii in range(PL_30freq_num):
    OMG_30freq_array[ii] = np.log10(
        h**2*omega_GW(freqs_30, A_arr[ii], gamma_arr[ii]))

num = 67
num_freqs = 30
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
with open('blue/data/v1p1_all_dict.json', 'r') as f:
    data = json.load(f)

# Finally realized the log10_A and gamma I needed were in https://zenodo.org/records/8067506 in the
# NANOGrav15yr_CW-Analysis_v1.0.0/15yr_cw_analysis-main/data/15yr_quickCW_detection.h5 file.
A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

# plt.scatter(gamma_arr, A_arr, color='black')
# gamma_arr = np.zeros((PL_30freq_num,30))
# A_arr = np.zeros((PL_30freq_num,30))
PL = np.zeros((67, num_freqs))
for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))
    PL[ii] = np.log10(powerlaw(freqs, A_arr[ii], gamma_arr[ii]))
plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0), OMG_15.mean(axis=0) +
                 2*OMG_15.std(axis=0), color='orange', label='2$\sigma$ posterior of GWB', alpha=0.5)

plt.plot(np.log10(freqs), np.log10(h**2*omega_GW(freqs, -15.6, 4.7)),
         linestyle='dashed', color='black', label='SMBHB spectrum')
# plt.fill_between(np.log10(freqs), PL.mean(axis=0) - 2*PL.std(axis=0), PL.mean(axis=0) + 2*PL.std(axis=0), color='orange', alpha=0.5)
# plt.fill_between(log10_f, omega_15.mean(axis=0) - 2*omega_15.std(axis=0), omega_15.mean(axis=0) + 2*omega_15.std(axis=0), color='orange', alpha=0.75)

# trying to reproduce Emma's fig 1 in https://arxiv.org/pdf/2102.12428.pdf
# plt.fill_between(np.log10(freqs_30), OMG_30freq_array.mean(axis=0) - 2*OMG_30freq_array.std(axis=0), OMG_30freq_array.mean(axis=0) + 2*OMG_30freq_array.std(axis=0), color='pink', alpha=0.75)
# plt.fill_between(freqs_30, PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0), PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0), color='pink', alpha=0.55)

# This part plots the energy densities of massive gravitons from the Mukohyama Blue tilted paper https://arxiv.org/pdf/1808.02381.pdf
H_inf = 1e8
tau_r = 5.494456683825391e-7  # calculated from equation (19)
a_r = 1/(tau_r*H_inf)

M_GW = .5*H_inf
tau_m = 1e21*tau_r

plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)), color='blue', label=r'MG - Blue-tilted, $m = 0.3H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')


BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between(np.log10(BBN_f), np.log10(BBN_f*0+h**2*1e-5),
                 np.log10(BBN_f * 0 + 1e1), alpha=0.5, color='orchid')
plt.text(-8.5, -5.4, r"BBN Bound", fontsize=15)


# Plot Labels
plt.title(r'NANOGrav 15-year data and Time Dependent model')
# plt.xlabel('$\gamma_{cp}$')
plt.xlabel(r'log$_{10}(f$ Hz)')

# plt.ylabel(r'log$_{10}(A_{GWB})$')
# plt.ylabel('log$_{10}A_{cp}$')
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$')

plt.xlim(-9, -7)
plt.ylim(-12, -4)

# plt.xscale("log")
# plt.yscale("log")
plt.legend(loc='lower left')



plt.savefig('nanograv/ng_blue_figs/fig1.pdf')
plt.show()
