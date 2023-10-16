import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
from blue_func import *

f_yr = 1/(365*24*3600)
gamma_cp = 13/3

# # Common Process Spectral Model Comparison Plot (Figure 1) # #

## Definition for powerlaw and broken powerlaw for left side of Figure 1
def powerlaw(f, log10_A=-16, gamma=gamma_cp):
    return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma-3) * f**(-gamma) * f[0])

# determine placement of frequency components
Tspan = 12.893438736619137 * (365 * 86400) #psr.toas.max() - psr.toas.min() #
freqs_30 = 1.0 * np.arange(1, 31) / Tspan


chain_DE438_30f_vary = np.loadtxt('./blue/data/12p5yr_DE438_model2a_cRN30freq_gammaVary_chain.gz', usecols=[90,91,92], skiprows=25000)
chain_DE438_30f_vary = chain_DE438_30f_vary[::4]

# Pull MLV params
DE438_vary_30cRN_idx = np.argmax(chain_DE438_30f_vary[:,-1])

# Make MLV Curves
PL_30freq = powerlaw(freqs_30, log10_A=chain_DE438_30f_vary[:,1][DE438_vary_30cRN_idx], gamma=chain_DE438_30f_vary[:,0][DE438_vary_30cRN_idx])

PL_30freq_num = int(chain_DE438_30f_vary[:,0].shape[0] / 5.)
PL_30freq_array = np.zeros((PL_30freq_num,30))
for ii in range(PL_30freq_num):
    PL_30freq_array[ii] = np.log10(powerlaw(freqs_30, log10_A=chain_DE438_30f_vary[ii*5,1], gamma=chain_DE438_30f_vary[ii*5,0]))


# Make Figure
plt.style.use('dark_background')
plt.figure()

# Left Hand Side Of Plot

#plt.semilogx(freqs_30, (PL_30freq_array.mean(axis=0)), color='C2', label='PL (30 freq.)', ls='dashdot')
#plt.fill_between(freqs_30, (PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0)), (PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0)), color='C2', alpha=0.15)


def omega_GW(f,A_cp):
    return 2*np.pi**2*A_cp**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma_cp)

#trying to reproduce Emma's fig 1 in https://arxiv.org/pdf/2102.12428.pdf
plt.fill_between(freqs_30, omega_GW(freqs_30,PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0)), omega_GW(freqs_30,PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0)), color='pink', alpha=0.75)

# Plot Labels
plt.title('NANOGrav 12.5 2$\sigma$ contours, Emma fig 1')
plt.xlabel(r'Frequency [Hz]')
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$') 
#plt.ylim(-9, -6)
plt.xlim(10**-9,10**-7)

plt.xscale("log")
plt.yscale("log")

plt.savefig('blue/nanograv_masses_figs/fig1.pdf')
plt.show()