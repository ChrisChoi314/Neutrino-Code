import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *


N=500
num_freqs = N
H_inf = 1e8  # in GeV
tau_r = 5.494456683825391e-7  # calculated from equation (19)

tau_m = 6.6e21*tau_r
f_UV = 2e8*(H_inf/1e14)**.5

tau_m = 1e10*(H_inf/1e14)**-2*tau_r

freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
M_arr = np.linspace(0.00001,1.499999, N)*H_inf

def f(x, y):
    return omega_GW_full(x, y, H_inf, tau_r, tau_m)

def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

X, Y = np.meshgrid(freqs, M_arr)
Z = f(X, Y)

ax = plt.axes(projection='3d')
surf = ax.plot_surface( np.log10(X), Y, np.log10(Z),linestyles="solid", cmap='hot')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('$M_{GW}$ [$H_{inf}$]')
ax.set_zlabel('$\log(h_0\Omega_{GW})$')

ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d

ax.view_init(elev=20, azim=120)
ax.annotate(r'$\tau_m = 6.6\times 10^{21}\tau_r$', xy=(0.6,0.9),xycoords='axes fraction',
             fontsize=10)

# plt.savefig('nanograv/contour_figs/fig1.pdf')


# for the second figure 

plt.gca()
ax.clear()

tau_m_arr = np.logspace(0,np.log10(1e10*(H_inf/1e14)**-2), N)*tau_r

M_GW = H_inf

def f(x, y):
    return omega_GW_full(x, M_GW, H_inf, tau_r, tau_m_arr)

X, Y = np.meshgrid(freqs, M_arr)
Z = f(X, Y)

plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
surf = ax.plot_surface( np.log10(X), Y, np.log10(Z),linestyles="solid", cmap='hot')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'$\tau_m$ [s]')
ax.set_zlabel('$\log(h_0\Omega_{GW})$')

ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d

ax.view_init(elev=20, azim=120)
ax.annotate(r'$M_{GW} = H_{inf}$', xy=(0.6,0.9),xycoords='axes fraction',
             fontsize=10)

plt.savefig('nanograv/contour_figs/fig1.pdf')
plt.show()