import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
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
A_arr = np.zeros((PL_30freq_num,30))
for ii in range(PL_30freq_num):
    PL_30freq_array[ii] = np.log10(powerlaw(freqs_30,log10_A=chain_DE438_30f_vary[ii*5,1], gamma=chain_DE438_30f_vary[ii*5,0]))
    A_arr[ii] = chain_DE438_30f_vary[ii*5,1]
    gamma_arr[ii] = chain_DE438_30f_vary[ii*5,0]


hdf_file = "blue/data/15yr_quickCW_detection.h5"

#specify how much of the first samples to discard
#(no need to discard any for provided samples as that has already been done)
#and how much more to thin the samples in addition to what we already thinned when we saved the samples
#burnin = 100_000
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    print(list(f.keys()))
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][0,burnin::extra_thin,:]
    print(f['samples_cold'].dtype)
    #samples = f['samples_cold'][...]
    log_likelihood = f['log_likelihood'][:,burnin::extra_thin]
    print(f['log_likelihood'].dtype)
    #log_likelihood = f['log_likelihood'][...]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]
burnin = 0
thin = 1



# Make Figure
plt.style.use('dark_background')
plt.figure(figsize=(22, 14))

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

N = 1000
f = np.linspace(-9, math.log(3e-7,10),N)
f= np.logspace(-8.6, -7,30)
freqs_30 = f
#A = np.vectorize(powerlaw_vec)(f,f[0], np.linspace(-18,-11,N ), np.ones(N)*gamma_12p5)

# plt.plot(f, A, color='orange')
def omega_GW(f,A_cp,gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)

OMG_30freq_array = np.zeros((PL_30freq_num,30))
for ii in range(PL_30freq_num):
    OMG_30freq_array[ii] = np.log10(h**2*omega_GW(freqs_30,A_arr[ii],gamma_arr[ii]))

for ii in range(PL_30freq_num):
    break
    #plt.scatter(gamma_arr[ii], A_arr[ii])
log10_f = samples_cold[burnin::thin,3]
log10_A = samples_cold[burnin::thin,4]
gam = samples_cold[burnin::thin,2]

num = 67
num_freqs = 30
freqs = np.logspace(np.log10(2e-9),np.log10(6e-8), num_freqs)
with open('blue/data/v1p1_all_dict.json', 'r') as f:
    data= json.load(f)
A_arr = []
gamma_arr = []
i = 0
for key in data.keys():
    if 'log10_A' in key:
        #print(key)
        A_arr.append(data[key])
    if 'gamma' in key:
        #print(key)
        gamma_arr.append(data[key])
A_arr = np.array(A_arr)
gamma_arr = np.array(gamma_arr)

A_arr = samples_cold[:,-1]
gamma_arr = samples_cold[:,-2]
#A_arr = np.linspace(-18,-11,67)
#gamma_arr = np.linspace(0,7,67)
#PL_30freq = powerlaw(freqs_30, log10_A=chain_DE438_30f_vary[:,1][DE438_vary_30cRN_idx], gamma=chain_DE438_30f_vary[:,0][DE438_vary_30cRN_idx])
#PL_30freq_num = int(chain_DE438_30f_vary[:,0].shape[0] / 5.)
OMG_15 = np.zeros((67,num_freqs))

# plt.scatter(gamma_arr, A_arr, color='black')
#gamma_arr = np.zeros((PL_30freq_num,30))
#A_arr = np.zeros((PL_30freq_num,30))
PL = np.zeros((67,num_freqs))
for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs,A_arr[ii], gamma_arr[ii]))
    PL[ii] = np.log10(powerlaw(freqs,A_arr[ii], gamma_arr[ii]))
plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0), OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0), color='orange', label='2$\sigma$ posterior of GWB', alpha=0.5)

plt.plot(np.log10(freqs),np.log10(h**2*omega_GW(freqs,-15.6,4.7)), linestyle='dashed', color='white', label='SMBHB spectrum' )
#plt.fill_between(np.log10(freqs), PL.mean(axis=0) - 2*PL.std(axis=0), PL.mean(axis=0) + 2*PL.std(axis=0), color='orange', alpha=0.5)
# omega_15 = np.log10(h**2*omega_GW(10**log10_f,log10_A,gam))
# plt.fill_between(log10_f, omega_15.mean(axis=0) - 2*omega_15.std(axis=0), omega_15.mean(axis=0) + 2*omega_15.std(axis=0), color='orange', alpha=0.75)
#trying to reproduce Emma's fig 1 in https://arxiv.org/pdf/2102.12428.pdf
# plt.fill_between(np.log10(freqs_30), OMG_30freq_array.mean(axis=0) - 2*OMG_30freq_array.std(axis=0), OMG_30freq_array.mean(axis=0) + 2*OMG_30freq_array.std(axis=0), color='pink', alpha=0.75)
# plt.fill_between(freqs_30, PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0), PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0), color='pink', alpha=0.55)


H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14
a_r = 1/(tau_r*H_inf)

def omega_GW_approx(f, m):
    f_8 = f/(2e8)
    nu = (9/4 - m**2 / H_inf**2)**.5
    k = f*2*np.pi
    #return tau_m/tau_r*(k*tau_r)**(3-2*nu)*1e-15 *H_14**2
    return 1e-15 * tau_m/tau_r * H_14**(nu+1/2)*f_8**(3-2*nu)


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
#plt.text(np.log10(2e-10),np.log10(1e-14), "Nano Grav \n Sensitivity ", fontsize=15)
#plt.plot(np.log10(freq_NG), np.log10(omega_GW_NG), color='dodgerblue')
M_arr = [.5*H_inf, .8*H_inf]
tau_m_r_ratio = [1e10, 1e15, 1e21]
colors = ['white', 'cyan', 'yellow']
linestyle_arr = ['solid', 'dashed']
text = ['m = 0.5$H_{inf}$', 'm = 0.8$H_{inf}$']
text2 =  ['1e10', '1e15', '1e21']
idx = 0
idx2 = 0

for ratio in tau_m_r_ratio: 
    for M_GW in M_arr:
        tau_m = tau_m_r_ratio[idx2]

        #plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_approx(freqs, M_GW)), linestyle=linestyle_arr[idx], color=colors[idx2])
        
        #ax1.plot(f, omega_GW_massive2(f, M_GW,tau_r), linestyle=linestyle_arr[idx], color='cyan',
        #        label=text[idx] + ', exact expression v2')
        #ax1.plot(f, omega_GW_massive1(f*2e8, M_GW,tau_r), linestyle=linestyle_arr[idx], color='green',
        #        label=text[idx] + ', exact expression v1')
        idx+=1
    idx =0
    idx2+=1
tau_m = 1e21*tau_r
M_GW = .5*H_inf
plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_approx(freqs, M_GW)), color='blue', label = r'MG - Blue-tilted, $m = 0.5H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')
tau_m = 1e25*tau_r
M_GW = .8*H_inf
plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_approx(freqs, M_GW)), linestyle= 'dashed', color='blue', label = r'MG - Blue-tilted, $m = 0.8H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{25}$')

BBN_f = np.logspace(-10, 9)
plt.fill_between(np.log10(BBN_f),np.log10(BBN_f*0+1e-5), np.log10(BBN_f *0 + 1e1), alpha=0.5, color='orchid')
plt.text(-8.5,-5.4, r"BBN", fontsize=15)


N = 1000
P_prim_k = 2.43e-10

omega_0 = np.logspace(-8 + .2, -7)
omega_0 = np.logspace(math.log(M_GW, 10), -2, N)
k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
# a_k = solve(k)  # uses multithreading to run faster
# omega_k = np.sqrt((k / a_k) ** 2 + M_GW**2)
k_prime = a_0 * omega_0
beta = H_eq**2 * a_eq**4 / (2)
a_k_prime_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_prime**2 + beta)) / (
    2 * a_eq * k_prime**2
)


def A(k):
    return np.where(k >= 0., np.sqrt(P_prim_k*np.pi**2/(2*k**3)), -1.)


# gamma_k_t_0 = A(k)*np.sqrt(omega_k * a_k**3 / (omega_0*a_0**3))
# P = np.where(omega_0 <= M_GW, np.nan, omega_0**2 /
#             (omega_0**2-M_GW**2)*(2*k**3/np.pi**2)*gamma_k_t_0**2)
# P = omega_0**2/(omega_0**2-M_GW**2)*(2*k**3/np.pi**2)#*y_k_0**2
gamma_k_GR_t_0 = A(k_prime)*a_k_prime_GR/a_0
P_GR = (2*k_prime**3/np.pi**2)*gamma_k_GR_t_0**2  # *y_k_0**2
# S = np.where(omega_0 <= M_GW, np.nan, k_prime * a_k / (k * a_k_prime_GR)
#             * np.sqrt(omega_k * a_k / (omega_0 * a_0)))

omega_c = np.sqrt((k_c/a_c)**2 + M_GW**2)


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


S_approx = np.vectorize(enhance_approx)(omega_0)


linestyle_arr = ['dotted', 'dashdot',  'dashed', 'solid']
M_arr = [4.3e-23*1e-9, 1.2e-22*1e-9, 0]
M_arr = [k / hbar for k in M_arr]
linestyle_arr = ['solid', 'dashed', 'solid']
color_arr = ['red', 'red', 'green']
text = ['Upper bound 2023 NANOGrav','Upper bound 2016 LIGO', 'GR']
idx = 0
for M_GW in M_arr:
    if M_GW == 0: 
        omega_0 = np.logspace(-10, -1)
    else: omega_0 = np.logspace(math.log(M_GW, 10), math.log(.1*2*np.pi, 10), N)
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
    # ax1.plot(f, np.sqrt(P_GR*S**2),
             # linestyle=linestyle_arr[idx], color='black', label=r'$M_{GW}/2\pi=$' + f'{round_it(M_GW/(2*np.pi), 1)} Hz')
    if idx < 2:
        plt.plot(np.log10(f),np.log10(h**2* 2*np.pi**2* np.sqrt(P_GR*S**2)**2/(3*H_0**2)*(f)**(3)),linestyle=linestyle_arr[idx], color='red', label=r'MG - Emir Gum. - $M_{GW}=$' + f'{round_it(M_GW*hbar, 2)}'+r' GeV/$c^2$'+ ' ('+text[idx] +')')
    else: 
        plt.plot(np.log10(f),np.log10(h**2* 2*np.pi**2* np.sqrt(P_GR*S**2)**2/(3*H_0**2)*(f)**(3)), color='green', label=r'GR - massless graviton')
    idx += 1



# Plot Labels
plt.title(r'NANOGrav 15-year data and Massive Gravity Models')
plt.xlabel('$\gamma_{cp}$')
plt.xlabel(r'log$_{10}(f$ Hz)')

# plt.ylabel(r'log$_{10}(A_{GWB})$') 
plt.ylabel('log$_{10}A_{cp}$')
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$')
# plt.ylim(-9, -6)

# plt.ylim(-9, -5.5)
#plt.xlim(10**-9,10**-7)

plt.xlim(-9,-7)
plt.ylim(-23,-4)
#plt.ylim(10**-12,10**-4)

#plt.xscale("log")
#plt.yscale("log")
plt.legend()




plt.savefig('blue/nanograv_masses_figs/fig6.pdf')
plt.show()