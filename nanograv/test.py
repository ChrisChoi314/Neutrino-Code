import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *

# Much of this code is taken from the NANOGrav collaboration's github page, where they have code that generates certain plots from their set of 4 (or 5?) papers.

print(conf_time(1/(1+1e-2)))
print(conf_time(1/(1+1e-2))/5.5e-7)

f_yr = 1/(365*24*3600)
gamma_12p5 = 13/3
gamma_15 = 3.2  # from page 4 of https://iopscience.iop.org/article/10.3847/2041-8213/acdac6/pdf
A_12p5 = -16
A_15 = -14.19382002601611

gamma_cp = gamma_15
A_cp = A_15

plt.figure()

N = 1000

H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14
a_r = 1/(tau_r*H_inf)


def omega_GW_approx(f, m):
    f_8 = f/(2e8)
    nu = (9/4 - m**2 / H_inf**2)**.5
    return 1e-15 * tau_m/tau_r * H_14**(nu+1/2)*f_8**(3-2*nu)


M_GW = .3*H_inf

def f(x,y):
    return np.log10(1e-15*H_14**(y+(1/2))*((10**x)/2e8)**(3-2*y))

nu = np.linspace(0,3/2, N)
freqs = np.linspace(-9,-7, N)
X,Y = np.meshgrid(freqs,nu)
func = f(X,Y)
plt.contourf(X,Y,func)
plt.colorbar()

# Plot Labels
plt.title(r'Contour of $\nu$ values and freq values for $\Omega_{GW,massive}$ for $\frac{\tau_m}{\tau_r} = 1$')
# plt.xlabel('$\gamma_{cp}$')
plt.xlabel(r'log$_{10}(f$ Hz)')

# plt.ylabel(r'log$_{10}(A_{GWB})$')
# plt.ylabel('log$_{10}A_{cp}$')
plt.ylabel(r'$\nu = \sqrt{\frac{9}{4} - \frac{m^2}{H_{inf}^2}}$')

plt.legend(loc='lower left')



# plt.savefig('nanograv/test_figs/fig0.pdf')
plt.show()
