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

N=500
num_freqs = N
H_inf = 1e8  # in GeV
tau_r = 5.494456683825391e-7  # calculated from equation (19)

tau_m = 6.6e21*tau_r
tau_m = 1e10*(H_inf/1e14)**-2*tau_r
f_UV = 2e8*(H_inf/1e14)**.5

tau_m = 1e10*(H_inf/1e14)**-2*tau_r

freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
M_arr = np.linspace(0.00001,1.499999, N)*H_inf
M_arr = np.logspace(-6,np.log10(1.5), N)*H_inf

def f(x, y):
    return omega_GW_full(x, y, H_inf, tau_r, tau_m)

def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

X, Y = np.meshgrid(freqs, M_arr)
Z = f(X, Y)


plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
surf = ax.plot_surface( np.log10(X), np.log10(Y/H_inf), np.log10(Z),linestyles="solid", cmap='autumn')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('$M_{GW}$ [$H_{inf}$]')
ax.set_zlabel('$\log(h_0\Omega_{GW})$')

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))


surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d

ax.view_init(elev=20, azim=120)
ax.annotate(r'$\tau_m = 10^{10}*(H_{inf}/10^{14})^{-2}\tau_r, H_{inf} = 10^8$ GeV', xy=(0.4,0.9),xycoords='axes fraction',
             fontsize=fs)

plt.savefig('nanograv/contour_figs/fig0.pdf')


# for the second figure 

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
surf = ax.plot_surface( np.log10(X),np.log10(Y/tau_r), np.log10(Z),linestyles="solid", cmap='autumn')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'$\tau_m \ [\tau_r]$')
ax.set_zlabel('$\log(h_0\Omega_{GW})$')

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d

#ax.view_init(elev=20, azim=120)
ax.annotate(r'$M_{GW} = H_{inf}, H_{inf} = 10^8$ GeV', xy=(0.4,0.9),xycoords='axes fraction',
             fontsize=fs)

plt.savefig('nanograv/contour_figs/fig1.pdf')

# for the third figure 

plt.gca()
ax.clear()


H_inf = np.logspace(-24,14, N) # in GeV

tau_m = 1e10*(H_inf/1e14)**-2*tau_r 
M_GW = H_inf

def f(x, y):
    return omega_GW_full(x, M_GW, y, tau_r, tau_m)

X, Y = np.meshgrid(freqs, H_inf)
Z = f(X, Y)

plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
surf = ax.plot_surface( np.log10(X),np.log10(Y), np.log10(Z),linestyles="solid", cmap='autumn')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'$H_{inf}$')
ax.set_zlabel('$\log(h_0\Omega_{GW})$')

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d

#ax.view_init(elev=20, azim=120)
ax.annotate(r'$M_{GW} = H_{inf}, 10^{10}*(H_{inf}/10^{14})^{-2}\tau_r$', xy=(0.4,0.9),xycoords='axes fraction',
             fontsize=fs)

plt.savefig('nanograv/contour_figs/fig2.pdf')

plt.show()