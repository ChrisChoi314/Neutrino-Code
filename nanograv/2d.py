import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

fs = 13
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


dir = '30f_fs{hd}_ceffyl'
freqs = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/freqs.npy')
num_freqs = 30

A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

freqs_boundary = [freqs[0], freqs[-1]]

Omega_sig_f_low = []
Omega_sig_f_high = []

for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))

Omega_sig_f_low += [10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0))[0], 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0))[0]]
Omega_sig_f_low += [10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0))[0], 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0))[0]]
Omega_sig_f_low += [10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0))[0], 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0))[0]]

Omega_sig_f_high += [10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0))[-1], 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0))[-1]]
Omega_sig_f_high += [10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0))[-1], 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0))[-1]]
Omega_sig_f_high += [10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0))[-1], 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0))[-1]]

fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(8,12))

N = 2000
num_freqs = N
H_inf = 1  # in GeV
tau_r = 5.494456683825391e-7  # calculated from equation (19)
a_r = 1/(tau_r*H_inf)

f_UV = 1/tau_r / (2*np.pi)
tau_m = 1e27*tau_r

freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
 
M_arr = np.linspace(0.00000001, 1.499999, N)
#M_arr = np.logspace(-6,np.log10(1.499999), N)
colors = ['black', 'red']

ax1.plot(M_arr, omega_GW_full(freqs_boundary[0], M_arr*H_inf, H_inf, tau_r, tau_m), color = colors[0],linewidth = 2)
ax1.plot(M_arr, omega_GW_full(freqs_boundary[1], M_arr*H_inf, H_inf, tau_r, tau_m), color = colors[1])


#ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$m$')
ax1.set_xlim(0,1.5)

#ax1.annotate(r'$\tau_m = 10^{21}\tau_r, H_{\mathrm{inf}} = 10^8$ GeV', xy=(0.4,0.9),xycoords='axes fraction',fontsize=fs)

tau_m_arr = np.logspace(5,30, N)

M_GW = H_inf

ax2.plot(tau_m_arr, omega_GW_full(freqs_boundary[0], M_GW, H_inf, tau_r, tau_m_arr*tau_r), color = colors[0])
ax2.plot(tau_m_arr, omega_GW_full(freqs_boundary[1], M_GW, H_inf, tau_r, tau_m_arr*tau_r), color = colors[1])

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'$\tau_m / \tau_r$')

H_inf = np.logspace(-24,14, N) # in GeV

tau_r = 1/(a_r*H_inf)

tau_m = 1e27*tau_r 
M_GW = H_inf

ax3.plot(H_inf, omega_GW_full(freqs_boundary[0], M_GW, H_inf, tau_r, tau_m), color = colors[0])
ax3.plot(H_inf, omega_GW_full(freqs_boundary[1], M_GW, H_inf, tau_r, tau_m), color = colors[1])

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel(r'$H_{\mathrm{inf}}$')

axs = [ax1,ax2,ax3]

for i in range(3):
    axs[i].set_ylabel(r'$h_0^2\Omega_{\mathrm{GW}}$')
    axs[i].grid(which='major', alpha=.2)
    axs[i].grid(which='minor', alpha=.2)
    axs[i].legend(loc='upper left')
# for the second figure 
'''
plt.gca()
ax.clear()

tau_m_arr = np.logspace(0,np.log10(1e10*(H_inf/1e14)**-2), N)

M_GW = H_inf

def f(x, y):
    return omega_GW_full(x, M_GW, H_inf, tau_r, y)

X, Y = np.meshgrid(freqs, tau_m_arr*tau_r)
Z = f(X, Y)

plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
surf = ax.plot_surface( np.log10(X),np.log10(Y/tau_r), np.log10(Z),linestyles="solid",)
ax.set_xlabel('f [Hz]')
ax.set_ylabel(r'$\tau_m \ [\tau_r]$')
ax.set_zlabel('$h_0^2\Omega_{\mathrm{GW}}$')

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

#ax.view_init(elev=20, azim=120)
ax.annotate(r'$m = H_{\mathrm{inf}}, H_{\mathrm{inf}} = 10^8$ GeV', xy=(0.4,0.9),xycoords='axes fraction',
             fontsize=fs)

plt.savefig('nanograv/contour_figs/fig4b.pdf')

# for the third figure 

plt.gca()
ax.clear()


H_inf = np.logspace(-24,14, N) # in GeV

tau_r = 1/(a_r*H_inf)

tau_m = 1e21*tau_r 
M_GW = H_inf

def f(x, y):
    return omega_GW_full(x, M_GW, y, tau_r, tau_m)

X, Y = np.meshgrid(freqs, H_inf)
Z = f(X, Y)

plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
surf = ax.plot_surface( np.log10(X),np.log10(Y), np.log10(Z),linestyles="solid")
ax.set_xlabel('f [Hz]')
ax.set_ylabel(r'$H_{\mathrm{inf}}$')
ax.set_zlabel('$h_0^2\Omega_{\mathrm{GW}}$')

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))


#ax.view_init(elev=20, azim=120)
ax.annotate(r'$m = H_{\mathrm{inf}}, \tau_m = 10^{21}\tau_r$', xy=(0.4,0.9),xycoords='axes fraction',
             fontsize=fs)

'''
plt.savefig('nanograv/2d_figs/fig0.pdf')
plt.show()