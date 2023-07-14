import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.integrate import odeint
from emir_func import *

N = 1000
P_prim_k = 2.43e-10
fig, (ax1) = plt.subplots(1)  # , figsize=(22, 14))

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

'''
# Figure 4 from Emir Paper
# use omega_0 = np.logspace(math.log(M_GW, 10)+ 0.00000000001, -7, N)
ax1.plot(omega_0, np.sqrt(P), label='numerical')
ax1.plot(omega_0, np.sqrt(P_GR*S**2), '--', label='fully analytical')
# ax1.plot(omega_0, np.sqrt(P/P_GR), label='S^2')
ax1.set_xlabel(r'$\omega_0$ [Hz]')
ax1.set_ylabel(r'$[P(\omega_0)]^{1/2}$')

ax1.set_xlim(1e-8, 1e-7)
ax1.set_ylim(1e-18, 1e-2)
plt.title('Power Spectrum')
'''

'''
# Figure 5 from Emir Paper
# use omega_0 = np.logspace(math.log(M_GW, 10), -2, N)
ax1.plot(k/(a_0*H_0), np.sqrt(P), label = 'numerical' )
ax1.plot(k/(a_0*H_0), np.sqrt(P_GR*S**2), '--', label='fully analytical')
ax1.plot(k/(a_0*H_0), np.sqrt(P_GR*S_approx**2), '-.', label='semi-analytical')
ax1.set_xlabel(r'$k / a_0 H_0$')
ax1.set_ylabel(r'$[P(\omega_0)]^{1/2}$')
ax1.set_xlim(1e-8, 1e-7)
ax1.set_ylim(1e-18, 1e-2)
plt.title('Power Spectrum vs k')
'''

# Figure 6 from Emir Paper
# use omega_0 = np.logspace(math.log(M_GW, 10), -2, N)
M_arr = [2*np.pi*1e-8, 2*np.pi*1e-7, 2*np.pi*1e-6]
linestyle_arr = ['dotted', 'dashed', 'solid']
idx = 0
for M_GW in M_arr:
    # a_c = k_c / M_GW
    # a_0 = k_0 / M_GW

    omega_0 = np.logspace(math.log(M_GW, 10), math.log(.1*2*np.pi, 10), N)
    k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
    a_k = solve(k)  # uses multithreading to run faster
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
    P = omega_0**2/(omega_0**2-M_GW**2)*(2*k**3/np.pi**2)#*y_k_0**2
    gamma_k_GR_t_0 = A(k_prime)*a_k_prime_GR/a_0
    P_GR = (2*k_prime**3/np.pi**2)*gamma_k_GR_t_0**2  # *y_k_0**2
    S = np.where(omega_0 <= M_GW, np.nan, k_prime * a_k / (k * a_k_prime_GR)
                 * np.sqrt(omega_k * a_k / (omega_0 * a_0)))
    f = omega_0/(2*np.pi)
    ax1.plot(f, np.sqrt(P_GR*S**2),
             linestyle=linestyle_arr[idx], color='black', label=r'$M_{GW}/2\pi=1e$' + f'{idx-8} Hz')
    idx += 1

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
eLisa_sensitivity = np.sqrt((20/3)*(4*S_x_acc+S_x_sn + S_x_omn)/L**2*(1+(f_elisa/(0.41*(c/(2*L))))**2))
ax1.plot(f_elisa,eLisa_sensitivity, color='lime')
ax1.text(1e-4, 1e-20, r"eLISA", fontsize=15)

f_ss = np.logspace(-9, -5, N)
h_0 = 1.46e-15
f_0 = 3.72e-8
gamma = -1.08
stoc_sig = h_0*(f_ss/f_0)**(-2/3)*(1+f_ss/f_0)**gamma
#ax1.plot(f_ss,stoc_sig, color='red')
#ax1.text(1e-4, 1e-20, r"Predicted Stochastic Signal", fontsize=15)

f_ska = np.logspace(math.log(2.9e-9),-5,N)
G = 6.67e-11
M_sun = 1.989e30 
M_c = ((1.35*M_sun)**(3/5))**2/(2*M_sun)**(1/5)
d_c = 9.461e18
ska_sensitivity = np.where(f_ska <3.1e-9, 10**((((-8+16)/(math.log(2.9e-9,10)-math.log(3.1e-9,10))*(-math.log(2.9e-9,10)) - 8)) * np.log(f_ska)**((-8+16)/(math.log(1e-5,10)-math.log(3.1e-9)))), 10**(((-12.5+16)/(math.log(1e-5,10)-math.log(3.1e-9,10))*(math.log(2.9e-9,10)) - 12.5) * np.log(f_ska)**((-12.5+16)/(math.log(1e-5,10) - math.log(3.1e-9)))))  #2*(G*M_c)**(5/3)*(np.pi*f)**(2/3)/(c**4*d_c)
#print(ska_sensitivity)

f_ska_1 = np.linspace(2.9e-9, 3.1e-9, N)
plt.vlines(3.1e-9, 1e-16, 1e-8,colors='red')
ax1.text(3e-9, 1e-19, r"SKA", fontsize=15)
#ska_sen_1 = 10**((((-8+16)/(math.log(2.9e-9,10)-math.log(3.1e-9,10))*(-math.log(2.9e-9,10)) - 8)))*f_ska_1**((-8+16)/(math.log(2.9e-9,10)-math.log(3.1e-9,10)))

#ax1.loglog(f_ska_1, ska_sen_1, '-', color='red')

f_ska_2 = np.linspace(3.1e-9, 1e-5, N)
ska_sen_2 = 10**((-12.5+16)/(math.log(1e-5,10)-math.log(3.1e-9,10))*(-math.log(2.9e-9,10)) - 16)*f_ska_2**((-12.5+16)/(math.log(1e-5,10)-math.log(3.1e-9,10)))
print(((-12.5+16)/(math.log(1e-5,10)-math.log(3.1e-9,10))*(-math.log(2.9e-9,10)) - 12.5))
ax1.loglog(f_ska_2, ska_sen_2, color='red')
# ax1.plot(f_ska,ska_sensitivity, color='red')

ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$[P(f)]^{1/2}$')

ax1.set_xlim(1e-9, 1e-1)
ax1.set_ylim(1e-25, 1e-2)
plt.title('Gravitational Power Spectra')

ax1.legend(loc='best')
ax1.set_xscale("log")
ax1.set_yscale("log")

plt.savefig("emir/emir_P_figs/fig5.pdf")
plt.show()
