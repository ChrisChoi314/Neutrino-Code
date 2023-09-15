from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from blue_func import *

N = 1000

plt.style.use('dark_background')
fig, (ax1) = plt.subplots(1)  # , figsize=(22, 14))

H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14

def omega_GW(f, m):
    f_8 = f/(2e8)
    return 1e-15 * tau_m/tau_r * H_14**(v+1/2)*f_8**(3-2*v)


# range from https://arxiv.org/pdf/1201.3621.pdf after eq (5) 3x10^-5 Hz to 1 Hz
f_elisa = np.logspace(math.log(3e-5, 10), -1, N)
# eq (2), in Hz m^2 Hz^-1
S_x_acc = 1.37e-32*(1+(1e-4)/f_elisa)/f_elisa**4
# eq (3), in m^2 Hz^-1
S_x_sn = 5.25e-23
# eq (4), in m^2 Hz^-1
S_x_omn = 6.28e-23
# from caption of figure 1, in m
L = 1e9
eLisa_sensitivity = np.sqrt(
    (20/3)*(4*S_x_acc+S_x_sn + S_x_omn)/L**2*(1+(f_elisa/(0.41*(c/(2*L))))**2))
ax1.plot(f_elisa, eLisa_sensitivity, color='lime')
ax1.text(1e-4, 1e-20, r"eLISA", fontsize=15)

f_ss = np.logspace(-9, -5, N)
h_0 = 1.46e-15
f_0 = 3.72e-8
gamma = -1.08
stoc_sig = h_0*(f_ss/f_0)**(-2/3)*(1+f_ss/f_0)**gamma

# fig 2 of https://arxiv.org/pdf/1001.3161.pdf
f_ska = np.logspace(math.log(2.9e-9), -5, N)
G = 6.67e-11
M_sun = 1.989e30
M_c = ((1.35*M_sun)**(3/5))**2/(2*M_sun)**(1/5)
d_c = 9.461e18
ska_sensitivity = np.where(f_ska < 3.1e-9, 10**((((-8+16)/(math.log(2.9e-9, 10)-math.log(3.1e-9, 10))*(-math.log(2.9e-9, 10)) - 8)) * np.log(f_ska)**((-8+16)/(math.log(1e-5, 10)-math.log(3.1e-9)))), 10**(
    ((-12.5+16)/(math.log(1e-5, 10)-math.log(3.1e-9, 10))*(math.log(2.9e-9, 10)) - 12.5) * np.log(f_ska)**((-12.5+16)/(math.log(1e-5, 10) - math.log(3.1e-9)))))  # 2*(G*M_c)**(5/3)*(np.pi*f)**(2/3)/(c**4*d_c)

f_ska_1 = np.linspace(2.9e-9, 3.1e-9, N)
plt.vlines(3.1e-9, 1e-16, 1e-8, colors='red')

f_ska_2 = np.linspace(3.1e-9, 1e-5, N)
ska_sen_2 = 10**((-12.5+16)/(math.log(1e-5, 10)-math.log(3.1e-9, 10))*(-math.log(
    2.9e-9, 10)) - 16)*f_ska_2**((-12.5+16)/(math.log(1e-5, 10)-math.log(3.1e-9, 10)))
ax1.loglog(f_ska_2, ska_sen_2, color='red')
# ax1.plot(f_ska,ska_sensitivity, color='red')
ax1.text(6e-6, 1e-12, r"SKA", fontsize=15)


outfile = np.load('emir/emir_hasasia/nanograv_sens_full.npz')

f_nanoGrav = outfile['freqs']
nanoGrav_sens = outfile['sens']
ax1.plot(f_nanoGrav, nanoGrav_sens, color='dodgerblue')

ax1.text(2e-10, 1e-14, "Nano\nGrav", fontsize=15)
ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$[P(f)]^{1/2}$')  # (r'$[P(f)]^{1/2}$')

linestyle_arr = ['dotted', 'dashdot',  'dashed', 'solid']
M_arr = [2*np.pi*1e-8, 2*np.pi*1e-7, 2*np.pi*1e-6]
M_arr = [4.3e-23*1e-9, 1.2e-22*1e-9, 0]
M_arr = [k / hbar for k in M_arr]

M_arr = [.5*H_inf, .8*H_inf]
linestyle_arr = ['solid', 'dashed']
text = ['m = 0.5$H_{inf}$', 'm = 0.8$H_{inf}$']
idx = 0

f = f_nanoGrav
for M_GW in M_arr:
    f = np.logspace(-18, 5, N)
    v = (9/4-M_GW**2/H_inf**2)**.5
    ax1.plot(f, omega_GW(f, M_GW), linestyle=linestyle_arr[idx], color='gold',
             label=text[idx])
    idx+=1

ax1.set_xlim(1e-18, 1e9)
ax1.set_ylim(1e-22, 1e1)
ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$\Omega_{GW}$')
plt.title('Gravitational Energy Density')

ax1.legend(loc='best')
ax1.set_xscale("log")
ax1.set_yscale("log")

plt.savefig("nano/blue_emir_figs/fig0.pdf")
plt.show()