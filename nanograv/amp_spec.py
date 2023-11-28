import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

from scipy.spatial import ConvexHull

def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)

hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]

num_freqs = 30
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
# Finally realized the log10_A and gamma I needed were in https://zenodo.org/records/8067506 in the
# NANOGrav15yr_CW-Analysis_v1.0.0/15yr_cw_analysis-main/data/15yr_quickCW_detection.h5 file.
A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

num_freqs = 30
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)

for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))

i = 0
points = []
while i < len(A_arr):
    points += [[gamma_arr[i], A_arr[i]]]
    i+=1
points = np.array(points)
print(points)
hull = ConvexHull(points)

# plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', color = 'orange', alpha=.8)

plt.vlines(13/3, -16,-13, colors='black', linestyle='dashed')
plt.text(4.23, -14,'SMBHB', rotation='vertical')

#plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
#plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')  

#plt.scatter(gamma_arr,A_arr, edgecolor='orange')

num_freqs = 1000
tau_r = 5.494456683825391e-7  # calculated from equation (19)

H_inf_arr = [.47, 5.2, 5e1]
M_arr = [1.298, 1.251, 1.201]
color_arr = ['red', 'blue', 'green']
idx = 0
for M_GW in M_arr:
    H_inf = H_inf_arr[idx]
    M_GW *= H_inf
    freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
    tau_m = 1e10*(H_inf/1e14)**-2*tau_r
    Omega = h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m)
    Omega = np.log10(Omega)
    freqs = np.log10(freqs)
    for i in range(len(Omega)):
        grad = np.gradient(Omega, freqs)
        gamma_gw = 5 - grad
        A_gw = (Omega - grad*(freqs - np.log10(f_yr)) - np.log10(2*np.pi**2*f_yr**2/(3*H_0**2)) )/2
    plt.plot(gamma_gw,A_gw, color = color_arr[idx],label=r"$M_{GW}$ = "+f'{M_arr[idx]}'+r'$H_{inf}$' + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    '''plt.plot(freqs, h**2*omega_GW_full(freqs, M_GW, H_inf, tau_r, tau_m),
         color=color_arr[idx], label=r"$M_{GW}$ = "+f'{M_arr[idx]}'+r'$H_{inf}$' + r", $H_{inf}$ = "+f'{H_inf} GeV' + r", $\frac{\tau_m}{\tau_r} = 10^{10}H_{14}^{-2}$")
    plt.plot(freqs, Omega, color=color_arr[idx], linestyle='dashed')'''
    idx+=1



plt.xlabel(r'$\gamma_{GW}$')
plt.ylabel(r'log$_{10}$A$_{GW}$')
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='lower left')
plt.grid(alpha=.2)
#plt.xlim(1.5,5)
#plt.ylim(-15.0,-13.6)
plt.savefig('nanograv/amp_spec_figs/fig0.pdf')
plt.show()