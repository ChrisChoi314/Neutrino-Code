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
a_r = 1/(tau_r*H_inf)

def scale_fac(conf_time):
    if conf_time < tau_r:
        return -1/(H_inf*conf_time)
    else:
        return a_r * conf_time / tau_r
def omega_GW_approx(f, m):
    f_8 = f/(2e8)
    nu = (9/4 - m**2 / H_inf**2)**.5
    return 1e-15 * tau_m/tau_r * H_14**(nu+1/2)*f_8**(3-2*nu)

def P_massless(f,m,tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    C_1 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(-3/4, m/(2*H_inf)) + (1-2*nu)*scipy.special.jv(1/4, m/(2*H_inf)))
    C_2 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(3/4, m/(2*H_inf)) - (1-2*nu)*scipy.special.jv(-1/4, m/(2*H_inf)))
    lamb = m*tau_m**2 / (2*H_inf*tau_r**2)
    D_1 = -np.sin(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    D_2 = np.cos(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    v_k_reg3 = 2/k*np.sqrt(m*tau_m / (np.pi*H_inf*tau_r**2)) * (D_1*np.cos(k*tau) + D_2*np.sin(k*tau))

    return 4*k**3*np.abs(v_k_reg3)**2/(np.pi**2*M_pl**2*(tau/(tau_r**2*H_inf))**2)


def omega_GW_massless(f, m, tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    return np.pi**2/(3*H_0**2)*f**2*P_massless(f,m,tau)

def omega_GW_massive(f,m,tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    return tau_m/tau_r*(k*tau_r)**(3-2*nu)*omega_GW_massless1(f,m,tau)

def P_T(f, tau):
    k = f*2*np.pi
    return k**2/(2*np.pi*scale_fac(tau)**2*M_pl**2)*(-k*tau)*np.abs(scipy.special.hankel1(3/2, tau))

def omega_GW_massless1(f,m,tau):
    k = f*2*np.pi
    return np.pi**2/(3*H_0**2)*f**2*P_T(f, tau)

def omega_GW_massive1(f,m,tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    return tau_m/tau_r*(k*tau_r)**(3-2*nu)*omega_GW_massless1(f,m,tau)

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

f = np.logspace(-18, 5, N)
for M_GW in M_arr:
    ax1.plot(f, omega_GW_approx(f, M_GW ), linestyle=linestyle_arr[idx], color='white',
             label=text[idx]+ ', approximation')
    
    ax1.plot(f, omega_GW_massless(f, M_GW,tau_m*1e10), linestyle=linestyle_arr[idx], color='cyan',
             label=text[idx] + ', exact expression')
    idx+=1

ax1.set_xlim(1e-18, 1e9)
ax1.set_ylim(1e-22, 1e1)
ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$\Omega_{GW}$')
plt.title('Gravitational Energy Density')

ax1.legend(loc='best')
ax1.set_xscale("log")
ax1.set_yscale("log")

plt.savefig("blue/blue_emir_figs/fig3.pdf")
plt.show()
