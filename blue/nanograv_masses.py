import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
from blue_func import *


f_yr = 1/(365*24*3600)
gamma_12p5 = 13/3
gamma_15 = 3.2 # from page 4 of https://iopscience.iop.org/article/10.3847/2041-8213/acdac6/pdf 
A_12p5 = -16
A_15 = -14.19382002601611

gamma_cp = gamma_15
A_cp = A_15

# # Common Process Spectral Model Comparison Plot (Figure 1) # #

## Definition for powerlaw and broken powerlaw for left side of Figure 1
def powerlaw_vec(f, f_0, log10_A=A_cp, gamma=gamma_cp):
    return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma-3) * f**(-gamma) * f_0)


def powerlaw(f, log10_A=A_cp, gamma=gamma_cp):
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
gamma_arr = np.zeros((PL_30freq_num,30))
for ii in range(PL_30freq_num):
    PL_30freq_array[ii] = np.log10(powerlaw(freqs_30,log10_A=chain_DE438_30f_vary[ii*5,1], gamma=chain_DE438_30f_vary[ii*5,0]))
    gamma_arr[ii] = chain_DE438_30f_vary[ii*5,0]



# Make Figure
#plt.style.use('dark_background')
plt.figure()

# Left Hand Side Of Plot

#plt.semilogx(freqs_30, (PL_30freq_array.mean(axis=0)), color='C2', label='PL (30 freq.)', ls='dashdot')
#plt.fill_between(freqs_30, (PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0)), (PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0)), color='C2', alpha=0.15)


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

# plt.scatter(log10_f, log10_A, color='orange')
N = 1000
f = np.linspace(-9, math.log(3e-7,10),N)
#A = np.vectorize(powerlaw_vec)(f,f[0], np.linspace(-18,-11,N ), np.ones(N)*gamma_12p5)

# plt.plot(f, A, color='orange')

def omega_GW(f,A_cp):
    return 2*np.pi**2*(10**(A_cp))**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma_arr.mean(axis=0))
print(gamma_arr.mean(axis=0))
f_test = freqs_30[0]
A_test = PL_30freq_array.mean(axis=0)[0] - PL_30freq_array.std(axis=0)[0]
print(f_test, A_test)
print(omega_GW(f_test, A_test))

#trying to reproduce Emma's fig 1 in https://arxiv.org/pdf/2102.12428.pdf
plt.fill_between(freqs_30, h**2*omega_GW(freqs_30,PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0)), h**2*omega_GW(freqs_30,PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0)), color='pink', alpha=0.75)

# plt.fill_between(freqs_30, PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0), PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0), color='pink', alpha=0.55)

# Plot Labels
plt.title('NANOGrav 12.5 2$\sigma$ contours, Emma fig 1')
plt.xlabel(r'Frequency [Hz]')

# plt.ylabel(r'log$_{10}(A_{GWB})$') 
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$') 
# plt.ylim(-9, -6)

# plt.ylim(-13, -4)
plt.xlim(10**-9,10**-7)
#plt.ylim(10**-12,10**-4)

plt.xscale("log")
plt.yscale("log")

plt.savefig('blue/nanograv_masses_figs/fig1.pdf')
plt.show()