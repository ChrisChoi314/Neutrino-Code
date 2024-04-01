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

N = 2000
num_freqs = N
H_inf = 1e8  # in GeV
tau_r = 5.494456683825391e-7  # calculated from equation (19)
a_r = 1/(tau_r*H_inf)

f_UV = 1/tau_r / (2*np.pi)
tau_m = 1e21*tau_r

freqs = np.logspace(-19,np.log10(f_UV),num_freqs)
M_arr = np.linspace(0.00001,1.499999, N)*H_inf
M_arr = np.logspace(-6,np.log10(1.499999), N)*H_inf

def f(x, y):
    return omega_GW_full(x, y, H_inf, tau_r, tau_m)

def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

X, Y = np.meshgrid(freqs, M_arr)
Z = f(X, Y)

print((Y/H_inf))

plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
surf = ax.plot_surface( np.log10(X), np.log10(Y/H_inf), np.log10(Z),linestyles="solid", cmap=color)
ax.set_xlabel('f [Hz]')
ax.set_ylabel('$m$ [$H_{\mathrm{inf}}$]')
ax.set_zlabel('$h_0^2\Omega_{\mathrm{GW}}$')

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.view_init(elev=20, azim=120)
ax.annotate(r'$\tau_m = 10^{21}\tau_r, H_{\mathrm{inf}} = 10^8$ GeV', xy=(0.4,0.9),xycoords='axes fraction',fontsize=fs)

plt.savefig('nanograv/contour_figs/fig4a.pdf')

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
surf = ax.plot_surface( np.log10(X),np.log10(Y/tau_r), np.log10(Z),linestyles="solid", cmap=color)
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
surf = ax.plot_surface( np.log10(X),np.log10(Y), np.log10(Z),linestyles="solid", cmap=color)
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

plt.savefig('nanograv/contour_figs/fig4c.pdf')

plt.show()