from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from blue_func import *

N = 20000    

plt.style.use('dark_background')
fig, (ax1) = plt.subplots(1, figsize=(10, 8))

H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14
a_r = 1/(tau_r*H_inf)
print(1e-15*H_14**2)
print(3/16/np.pi**3/M_pl**2*H_inf**2)

def scale_fac(conf_time):
    if conf_time < tau_r:
        return -1/(H_inf*conf_time)
    else:
        return a_r * conf_time / tau_r
def omega_GW_approx(f, m):
    f_8 = f/(2e8)
    nu = (9/4 - m**2 / H_inf**2)**.5
    k = f*2*np.pi
    #return tau_m/tau_r*(k*tau_r)**(3-2*nu)*1e-15 *H_14**2
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
    # print(scipy.special.hankel1(3/2, tau))
    return k**2/(2*np.pi*scale_fac(tau)**2*(M_pl/1e14)**2)*(k*tau)*np.abs(scipy.special.hankel1(3/2, tau))**2

def omega_GW_massless1(f,m,tau):
    k = f*2*np.pi
    return np.pi**2/(3*(H_0*1e14)**2)*f**2*P_massless(f, m, tau)

def omega_GW_massive1(f,m,tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    return tau_m/tau_r*(k*tau_r)**(3-2*nu)*omega_GW_massless1(f,m,tau)

def omega_GW_massless2(f,tau):
    k = f*2*np.pi
    return (np.pi**2/(3*(H_0/1e14)**2)*f**2*P_T(f,tau))

def omega_GW_massless3(f,tau):
    k = f*2*np.pi
    H_k = k/scale_fac(tau)
    return (np.pi**2/(3*(H_0)**2)*(f)**2*8*(H_k**2)/(M_pl**2 * (2*np.pi)**2))


def omega_GW_massive2(f,m,tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    return tau_m/tau_r*(k*tau_r)**(3-2*nu)*(np.pi**2/(3*H_0**2)*f**2*P_T(f,tau_r ))

v_0 = 1
v_prime_0 = 0
eta = np.logspace(1,math.log(conf_time(a_0), 10), N*10)

def emir_omega(f):
    k = f*2*np.pi
    v, v_prime = odeint(diffeqMode, [v_0, v_prime_0], eta, args=(k,0)).T
    return k**3 / (12*np.pi*Hubble(eta_0)**2)*(v_prime[-1]**2 + (k**2/a_0**2 + M_GW**2)*v[-1]**2)

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


f_nanoGrav = outfile['freqs']
nanoGrav_sens = outfile['sens']
ax1.plot(f_nanoGrav, nanoGrav_sens, color='dodgerblue')
ax1.plot(freq_NG, omega_GW_NG, color='blue')

ax1.text(2e-10, 1e-14, "Nano\nGrav", fontsize=15)
ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$[P(f)]^{1/2}$')  # (r'$[P(f)]^{1/2}$')

linestyle_arr = ['dotted', 'dashdot', 'dashed', 'solid']
M_arr = [2*np.pi*1e-8, 2*np.pi*1e-7, 2*np.pi*1e-6]
M_arr = [4.3e-23*1e-9, 1.2e-22*1e-9, 0]
M_arr = [k / hbar for k in M_arr]

M_arr = [.5*H_inf, .8*H_inf]
tau_m_r_ratio = [1e10, 1e15, 1e21]
colors = ['white', 'cyan', 'yellow']
linestyle_arr = ['solid', 'dashed']
text = ['m = 0.5$H_{inf}$', 'm = 0.8$H_{inf}$']
text2 =  ['1e10', '1e15', '1e21']
idx = 0
idx2 = 0
f = np.logspace(-19, 5, N)

'''def average(arr):
    outpt = []
    i= 2
    for i in range(len(arr)-1):
        outpt += [(arr[i-2]+arr[i-1]+ arr[i])/3]
    outpt += [arr[len(arr) -1]]
    return outpt
'''

for ratio in tau_m_r_ratio: 
    for M_GW in M_arr:
        tau_m = tau_m_r_ratio[idx2]
        ax1.plot(f, omega_GW_approx(f, M_GW), linestyle=linestyle_arr[idx], color=colors[idx2],label=text[idx]+ r', $\frac{\tau_m}{\tau_r} = $'+text2[idx2])
        #ax1.plot(f, omega_GW_massive2(f, M_GW,tau_r), linestyle=linestyle_arr[idx], color='cyan',
        #        label=text[idx] + ', exact expression v2')
        #ax1.plot(f, omega_GW_massive1(f*2e8, M_GW,tau_r), linestyle=linestyle_arr[idx], color='green',
        #        label=text[idx] + ', exact expression v1')
        idx+=1
    idx =0
    idx2+=1

# ax1.plot(f, np.vectorize(emir_omega)(f), color = 'orange',label="Emir's MG model")
num = 1e-8*tau_r/tau_m
#ax1.plot(f, omega_GW_massless2(f, num*tau_m), color='red',
#             label='massless')
#ax1.plot(f, omega_GW_massless3(f, tau_m), color='magenta',
#             label='massless v3')
#print(omega_GW_massless2(f, num*tau_m))
#ax1.plot(f, f*0+1e-15*H_14**2, color='yellow',
#             label=r'$\Omega_{GW,0}^{massless}$')
BBN_f = np.logspace(-10, 9)
ax1.fill_between(BBN_f, BBN_f*0+1e-5, BBN_f *0 + 1e1, alpha=0.5, color='orchid')
ax1.text(1e-12, 1e-5, r"BBN", fontsize=15)

CMB_f = np.logspace(-17, -16)
ax1.fill_between(CMB_f, CMB_f*0+1e-15, CMB_f *0 + 1e1, alpha=0.5, color='blue')
ax1.text(5e-16, 1e-13, r"CMB", fontsize=15)

ax1.set_xlim(1e-19, 1e9)
ax1.set_ylim(1e-22, 1e1)
ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$\Omega_{GW}$')
plt.title('Gravitational Energy Density')

ax1.legend(loc='upper right')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend().set_visible(False)
plt.savefig("blue/blue_emir_figs/fig4.pdf")
plt.show()
