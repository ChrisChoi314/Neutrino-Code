import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *
import math

N = 1000
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

#print("Current eta_0 = "+f'{eta[k_idx]}')

time = scipy.integrate.cumtrapz(a, eta, initial=0)



fig, (ax1) = plt.subplots(1)

#ax1.plot(eta, a, label='a')
#ax1.axhline(y=(k/M_GW)**2, linestyle="dashed",
#            linewidth=1, label=r'$k_0/M_{GW}$')
#ax1.plot(eta, time, label='Physical time')
#res, err = scipy.integrate.quad(scale_fac, 0, 2.105)

# print('Time: ' + f'{res}')

#roots = scipy.optimize.fsolve(
#    lambda x: scipy.integrate.quad(scale_fac, 0, x)[0] - 1, 1)
# print('Roots: ', roots[0])


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
    if k == 1e-1:
        ax1.plot(a, H_square, label = 'H')
        ax1.plot(a, ang_freq_square, label = 'omega')
    print('k = ', k)
    print('a_k = ', a_k)
    print('a_k using solve= ', solve(k))
    print('a_k using sympy1= ', sympy1(k))
    print('a_k using sympy2= ', sympy2(k))
    print('a_k using sympy3= ', sympy3(k))
    print('a_k using sympy4= ', sympy4(k))
    print('a_k using wolpha= ', inverse_H_omega(k))

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

ax1.legend(loc='best')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Scale Factor')

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
#plt.savefig("emir/emir_calc_figs/inverse_of_hubble.pdf")
plt.show()
