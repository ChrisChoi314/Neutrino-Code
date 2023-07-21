import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *
import math

print(f'Mass in kg: {M_GW*hbar/c**2}')
print(eta_0)
print(conf_time(a_0))
print(conf_time_anal(a_0))
N = 10
omega_0 = k_0/a_0
k = a_0*np.sqrt(omega_0**2-M_GW**2)
k = 1e-5
eta = np.logspace(17, 18, N)

H_square = np.vectorize(Hubble)(eta) ** 2
ang_freq_square = np.vectorize(ang_freq)(eta, k) ** 2
a = np.vectorize(scale_fac)(eta)
k_idx = 0
#print(a_0)
for i in range(0, len(eta)):
    if a[i] >= a_0:
        #print(a[i])
        k_idx = i
        break

omega_0 = np.logspace(math.log(M_GW, 10), -2, N)
omega_0 = np.logspace(math.log(M_GW, 10) - .04, math.log(M_GW, 10) + .2, N)
k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
print(k)
a_k = ak(k)
print(a_k)

# print("Current eta_0 = "+f'{eta[k_idx]}')
# print(f'eta_0 analytical = {np.sqrt(4/(H_0**2*omega_M))}')

time = scipy.integrate.cumtrapz(a, eta, initial=0)



#fig, (ax1) = plt.subplots(1)

#ax1.plot(eta, a, label='a')
#ax1.axhline(y=(k/M_GW)**2, linestyle="dashed",
#            linewidth=1, label=r'$k_0/M_{GW}$')
#ax1.plot(eta, time, label='Physical time')
#res, err = scipy.integrate.quad(scale_fac, 0, 2.105)

# print('Time: ' + f'{res}')

#roots = scipy.optimize.fsolve(
#    lambda x: scipy.integrate.quad(scale_fac, 0, x)[0] - 1, 1)
# print('Roots: ', roots[0])


'''

eta = np.logspace(-20, 20, N*1000)
a_k_1 = omega_M*H_0**2+H_0*np.sqrt(4*omega_R*k**2+(H_0*omega_M)**2)/(2*k**2)
#print('Using inverse with Wolpha: ', a_k_1)
idx_1 = 0
a = np.linspace(1e-30, 1e0,N)
#N = 2000
eta = np.logspace(-7, 1, N)
# k = np.logspace(-5,5, N)
omega_0 = k_0 / a_0
omega_0 = np.logspace(math.log(M_GW, 10) - .04, math.log(M_GW, 10) + .2, N)
omega_0 = np.linspace(M_GW/2, 2*M_GW, N)
omega_0 = np.logspace(math.log(M_GW, 10) - .04, math.log(M_GW, 10) + 6, N)
# omega_0 = np.linspace(M_GW - 10**.04, M_GW + 1.2, N)
k_prime = (
    a_0 * omega_0
)
eq_idx = 0
a = np.vectorize(scale_fac)(eta) 
H_square = np.vectorize(Hubble)(eta) ** 2
ang_freq_square = np.vectorize(ang_freq)(eta,k) ** 2
for i in range(0, len(eta)):
    if H_square[i] <= ang_freq_square[i]:
        eta_k = eta[i]
        eq_idx = i
        break

a = np.logspace(-30, 0,N)
k_arr = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
H_square = np.vectorize(Hubble_a)(a)**2
#print(Hubble_a(a))
#print(H_square)
for k in k_arr:
    ang_freq_square = np.vectorize(ang_freq_a)(a,k) ** 2
    eta_k = 0
    for i in range(len(a)):
        #print(a[i])
        if H_square[i] <= ang_freq_square[i]:
            a_k = a[i]
            #print(a_k)
            eq_idx = i
            break
    #if k == 1e-1:
        #ax1.plot(a, H_square, label = 'H')
        #ax1.plot(a, ang_freq_square, label = 'omega')
    #print('k = ', k)
    #print('a_k = ', a_k)
    #print('a_k using wolpha= ', inverse_H_omega(k))

#print('eta_k = ', eta_k)
a = np.vectorize(scale_fac)(eta)
#ax1.plot(a, H_square,label='H', color='orange')
#ax1.plot(a, ang_freq_square,label='omega', color='blue')
#ax1.axhline(y=k, label='k')
eta = np.linspace(.1, 10, N)
a = np.linspace(1e-22, 1e-6,N)
a = np.logspace(-22,-6 ,N)
#print(H_0**2*(omega_M/a**3 + omega_R/a**4 + omega_L) )
#ax1.plot(a,a*np.sqrt(H_0**2*(omega_M/a**3 + omega_R/a**4 + omega_L) - M_GW**2), label ='k(a)')
lim = 10
a = np.linspace(.01, lim ,N)
#ax1.plot(a, H_omega(a), label ='regular')
#ax1.plot(a, sympy1(a), label='inverse sympy1')
#ax1.plot(a, inverse_H_omega(a), label='inverse')
#ax1.plot([0, lim], [0, lim], '--', color='gray')
#print(sympy1(a))
#print(inverse_H_omega2(a*np.sqrt(H_0**2*(omega_M/a**3 + omega_R/a**4 + omega_L) - M_GW**2)) - a)

#ax1.legend(loc='best')
#ax1.set_xscale('log')
#ax1.set_yscale('log')
#ax1.set_title('Scale Factor')

''' 

'''
ax2.plot([0, 10], [0, 10], '--', color='gray')

a = H_0**2*omega_M
b = H_0**2*omega_R
c = H_0**2*omega_L
x = np.linspace(.1,10,10000)
y = reg_N(a,b,c,x)
y_min = y.min()
ax2.plot(x,y, label="regular")

inverse1 = inv_of_H(a,b,c,x, c1=1, c2=1)
inverse1 = np.where(x >= y_min, inverse1, np.nan)
ax2.plot(x,inverse1, label="inverse1")

inverse2 = inv_of_H(a,b,c,x, c1=1, c2=-1)
inverse2 = np.where(x >= y_min, inverse2, np.nan)
ax2.plot(x,inverse2, label="inverse2")

inverse3 = inv_approx(a,b,x, c1=1)
inverse3 = np.where(x >= y_min, inverse3, np.nan)
ax2.plot(x,inverse3, label="inverse3")

inverse4 = inv_approx(a,b,x, c1=-1)
inverse4 = np.where(x >= y_min, inverse4, np.nan)
ax2.plot(x,inverse4, label="inverse4")

ax2.legend(loc="best")

'''
N = 100

'''
omega_arr = np.array([5e-7])
omega_arr = np.array([5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-1])
k_arr = a_0*np.sqrt(omega_arr**2 - M_GW**2)
fig, (ax1, ax2) = plt.subplots(2, figsize=(22, 14))

eta = np.logspace(1, 18, N)
# eta = np.logspace(-3, -1, N)
a = np.logspace(-25,1, N)
# a = normalize_0(a)
v_0 = 1
v_prime_0 = 0
eta_0_idx = 0 
for i in range(len(eta)):
    if eta[i] >= eta_0:
        eta_0_idx = i
        break

for k in k_arr:
    v, v_prime = odeint(diffeqGR, [v_0, v_prime_0], eta, args=(k,)).T
    # print(v[eta_0_idx]/a[eta_0_idx])
    print()
    ax1.plot(eta, v, label=f"{k}" + r" $Hz$")
    v, v_prime = odeint(diffeqMG, [v_0, v_prime_0], eta, args=(k,)).T
    # print(v[eta_0_idx]/a[eta_0_idx])
    ax2.plot(eta, v, label=f"{k}" + r" $Hz$")
    a_k = solve_one(k)
    ax1.axvline(x=give_eta(a_k), label='eta_k', color='orange')
    ax2.axvline(x=give_eta(a_k), label='eta_k', color='orange')

ax1.legend(loc='best')
ax1.set_xscale("log")

ax2.legend(loc='best')
ax2.set_xscale("log")
'''
fig, (ax1) = plt.subplots(1)

# range from https://arxiv.org/pdf/1201.3621.pdf after eq (5) 3x10^-5 Hz to 1 Hz
f_elisa = np.logspace(math.log(1e-5, 10), 0, N)
# eq (1), in m^2 s^-4 Hz^-1
S_x_acc = 2.13e-29*(1+(1e-4)/f_elisa)
S_x_acc = 1.37e-32*(1+(1e-4)/f_elisa)/f_elisa**4
# eq (3), in m^2 Hz^-1
S_x_sn = 5.25e-23
# eq (4), in m^2 Hz^-1
S_x_omn = 6.28e-23
# from caption of figure 1, in m
L = 1e9
eLisa_sensitivity = np.sqrt((20/3)*(4*S_x_acc+S_x_sn + S_x_omn)/L**2*(1+(f_elisa/(0.41*(c/(2*L))))**2))
#print(eLisa_sensitivity)

f_ska = np.logspace(-5,0,N)
eLisa_sensitivity = ska_sensitivity = ((((-17+24)/(math.log(1e-5,10)-math.log(1e-0,10))*(17 + math.log(1e-5,10)))) * np.log(f_ska)**((-17+24)/(math.log(1e-5,10)-math.log(1e-0,10))))
# print(eLisa_sensitivity)
# print((-17+24)/(math.log(1e-5,10)-math.log(1e-0,10)))
# ax1.plot(f_elisa,eLisa_sensitivity, color='lime')
# ax1.text(1e-4, 1e-20, r"eLISA", fontsize=15)

x = np.linspace(1e-5, 1e0, 100)
y = 1e-17*x**(1.4)

# ax1.loglog(x, y, '-')
# ax1.plot([x[0], x[-1]], [y[0], y[-1]], '--', label='with plot')
ax1.legend()


ax1.set_xlabel(r'f [Hz]')
ax1.set_ylabel(r'$[P(f)]^{1/2}$')

# ax1.set_xlim(1e-5, 1e0)
# ax1.set_ylim(1e-24, 1e-17)
plt.title('Gravitational Power Spectra')

N = 100
a = np.logspace(-24,0, N)
eta = np.logspace(-20,18, N)
# a = np.linspace(1e-30, 1, N)
def conf_time(a):
    if a < scale_fac(eta_rm):
        return a/(H_0*np.sqrt(omega_R))
    else:
        return (a/(H_0**2*.25*omega_M))**(1/2)
def conf_time2(a):
    a_lm = 1.465
    if a < a_eq:
        return a/(H_0*np.sqrt(omega_R))
    elif a < a_lm:
        return (a/(H_0**2*.25*omega_M))**(1/2) - ((a_eq/(H_0**2*.25*omega_M))**(1/2) - a_eq/(H_0*np.sqrt(omega_R)))
    else:       
        #return np.arcsinh((1 / ((1/omega_L - 1)**(1/3)))**(-3/2))*2 /(3 * H_0 * np.sqrt(omega_L))
        #return 1/(a*H_0*np.sqrt(omega_L))
        return 1.9323e18
        return (a_lm/(H_0**2*.25*omega_M))**(1/2) - ((a_eq/(H_0**2*.25*omega_M))**(1/2) - a_eq/(H_0*np.sqrt(omega_R)))
def integrand(a):
    return 1 / (a**2*H_0*np.sqrt(omega_M/a**3 + omega_R/a**4 + omega_L))
def integral(x):
    return scipy.integrate.quad(integrand, 0, x)
def conf_time3(a):
    return (2*np.sqrt(omega_M*a + omega_R)/(H_0*omega_M))- 2*np.sqrt(omega_R)/(H_0*omega_M) 
#ax1.plot(a, np.vectorize(conf_time)(a),"-.", label='approximation method')
ax1.plot(a, np.vectorize(conf_time)(a),"-.", label='approximation method 1' )
ax1.plot(a, np.vectorize(conf_time3)(a),"-.", label='approximation method 2' )
ax1.plot(a, np.vectorize(integral)(a)[0], label='integral meth' )

# ax1.plot(eta, np.vectorize(scale_fac)(eta),"-.", label='old')
# ax1.plot(eta, np.vectorize(scale_fac2)(eta),"-.", label='new')

# ax1.plot(a, scipy.integrate.cumtrapz(1 / (a**2*H_0*np.sqrt(omega_M/a**3 + omega_R/a**4 + omega_L)), initial=0), "--", label = 'integration method 2' )

# print( np.vectorize(scale_fac)(eta))
# print( np.vectorize(scale_fac2)(eta))
ax1.legend(loc='best')
ax1.set_xscale("log")
ax1.set_yscale("log")

print(H_0)

plt.savefig("emir/emir_calc_figs/fig2.pdf")
plt.show()