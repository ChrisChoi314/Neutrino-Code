import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

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
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]

# Make Figure
# plt.style.use('dark_background')
plt.figure(figsize=(10, 4))

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
N = 200
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
                 2*OMG_15.std(axis=0), color='orange', label='2$\sigma$ posterior of GWB', alpha=0.5,interpolate=True)
plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 4*OMG_15.std(axis=0), OMG_15.mean(axis=0) +
                 4*OMG_15.std(axis=0), color='orange', label='4$\sigma$ posterior of GWB', alpha=0.3,interpolate=True)

plt.plot(np.log10(freqs), np.log10(h**2*omega_GW(freqs, -15.6, 4.7)),
         linestyle='dashed', color='black', label='SMBHB spectrum')
freqs = np.logspace(-19, 9, N)

BBN_f = np.logspace(-10, 9)
plt.fill_between(np.log10(BBN_f), np.log10(BBN_f*0+1e-5),
                 np.log10(BBN_f * 0 + 1e1), alpha=0.5, color='orchid',interpolate=True)
plt.text(-8.5, -5.4, r"BBN", fontsize=15)

plt.vlines(x=np.log10(2e-9/(2*np.pi)),ymin=-15, ymax=-4)

P_prim_k = 2.43e-10
beta = H_eq**2 * a_eq**4 / (2)


def A(k):
    return np.where(k >= 0., np.sqrt(P_prim_k*np.pi**2/(2*k**3)), -1.)


def enhance_approx(x):
    if x < M_GW:
        return 0.
    val = a_0 * np.sqrt(x**2 - M_GW**2)
    if k_0 < val:
        return 1.
    elif val <= k_0 and val >= k_c:
        if val >= k_eq:
            output = (x**2 / M_GW**2 - 1)**(-3/4)
            return output
        if val < k_eq and k_eq < k_0:
            output = k_eq/(np.sqrt(2)*k_0)(x**2 / M_GW**2 - 1)**(-5/4)
            return output
        if k_eq > k_0:
            output = (x**2 / M_GW**2 - 1)**(-5/4)
            return output
    elif val <= k_c:
        beta = H_eq**2 * a_eq**4 / (2)
        a_k_0_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_0**2 + beta)) / (
            2. * a_eq * k_0**2
        )
        if abs(x**2 / M_GW**2 - 1) < 1e-25:
            return 0.
        output = a_c/a_k_0_GR*np.sqrt(k_c/k_0)*(x**2 / M_GW**2 - 1)**(-1/2)
        return output
    

'''plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_massless(freqs*2*np.pi)),
         color='salmon', label=r'GR - Blue-tilted paper')'''

linestyle_arr = ['solid', 'dashed', 'solid']
M_arr = [4.3e-23*1e-9, 1.2e-22*1e-9, 0]
M_arr = [4.3e-23*1e-9, 1.2e-22*1e-9]
M_arr = [8.6e-24*1e-9, 8.2e-24*1e-9]
M_arr = [k / hbar for k in M_arr] + [2e-9*2*np.pi]
print(M_arr)
#linestyle_arr = ['solid', 'dashed', 'solid']
color_arr = ['red', 'blue', 'green']
text = ['2023 Wang et al.', '2023 Wu et al.', 'NG15 Freq Bound']
idx = 0


for M_GW in M_arr:
    if M_GW == 0:
        omega_0 = np.logspace(-10, -1, N)
        omega_0 = np.logspace(-18, 11, N)
    else:
        omega_0 = np.logspace(math.log(M_GW, 10), math.log(.1*2*np.pi, 10), N)
        omega_0 = np.logspace(math.log(M_GW, 10), np.log10(6e-8*2*np.pi), N)
    k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
    a_k = ak(k)  # uses multithreading to run faster
    omega_k = np.sqrt((k / a_k) ** 2 + M_GW**2)
    k_prime = (
        a_0 * omega_0
    )
    a_k_prime_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_prime**2 + beta)) / (
        2 * a_eq * k_prime**2
    )   

    gamma_k_t_0 = A(k)*np.sqrt(omega_k * a_k**3 / (omega_0*a_0**3))
    P = np.where(omega_0 <= M_GW, np.nan, omega_0**2 /
                 (omega_0**2-M_GW**2)*(2*k**3/np.pi**2)*gamma_k_t_0**2)
    P = omega_0**2/(omega_0**2-M_GW**2)*(2*k**3/np.pi**2)  # *y_k_0**2
    gamma_k_GR_t_0 = A(k_prime)*a_k_prime_GR/a_0
    P_GR = (2*k_prime**3/np.pi**2)*gamma_k_GR_t_0**2  # *y_k_0**2
    S = np.where(omega_0 <= M_GW, np.nan, k_prime * a_k / (k * a_k_prime_GR)
                 * np.sqrt(omega_k * a_k / (omega_0 * a_0)))
    f = omega_0/(2*np.pi)
    
    if idx < 3:
        al = 1
        plt.plot(np.log10(f), np.log10(h**2*2*np.pi**2*(P_GR*S**2 / (4*f))/(3*H_0**2)*(f)**(3)),
                 color=color_arr[idx], label=r'$M_{GW}=$' + f'{round_it(M_GW*hbar, 2)}'+r' GeV/$c^2$' + ' ('+text[idx] + ')')
        
    #comment this back in and the plot part in the if statement above out if you want figure fig0.pdf
    '''else:
        plt.plot(np.log10(f), np.log10(h**2*2*np.pi**2*(P_GR/(4*f))/(3*H_0**2)  
                 * (f)**(3)), color='palegreen', label=r'GR - Emir Gum. et al Paper')
        print((h**2*2*np.pi**2*(P_GR/(4*f))/(3*H_0**2)  
                 * (f)**(3)) / (h**2*omega_GW_massless(freqs*2*np.pi)))'''
    N_extra = math.log(10)
    a_k_0_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_0**2 + beta)) / (
        2 * a_eq * k_0**2
    )
    S_peak_anal = a_c*np.sqrt(k_0*k_c)/(a_k_0_GR*a_0*H_0)*np.e**N_extra
    S_peak_num = 0
    var = False
    for s in S:
        # print(s)
        if s != np.nan and var == False:
            S_peak_num = s
            var = True
    #print(f'M_GW (in Hz): {M_GW}, S_peak_anal: {S_peak_anal}, S_peak_num: {S_peak_num}')
    T_obs = 15/f_yr
    #if M_GW == 0:
    #    break
        #print(f'amplif factor: {1e-4*(T_obs/H_0)**(-4)*(M_GW/H_0)**(-3) *math.log(np.e**N_extra*M_GW/H_0)}')
    idx += 1
    

plt.xlim(-9, -7)
plt.ylim(-16, -4)

# Plot Labels
# plt.xlabel('$\gamma_{cp}$')
plt.xlabel(r'log$_{10}(f/$Hz)')

# plt.ylabel(r'log$_{10}(A_{GWB})$')
# plt.ylabel('log$_{10}A_{cp}$')
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$')

#plt.xlim(-9, -7)
#plt.ylim(-24, -4)
#plt.xlim(np.log10(2e-9), np.log10(6e-8))
# plt.xscale("log")
# plt.yscale("log")
plt.legend(loc='lower right')

plt.savefig('nanograv/ng_emir_figs/fig1.pdf')
plt.show()
