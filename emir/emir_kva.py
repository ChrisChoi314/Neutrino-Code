import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *


fig, (ax4) = plt.subplots(1) #, figsize=(22, 14))
N=1000
eta = np.logspace(-14, .25, N)
a = np.vectorize(scale_fac)(eta)
aH = a*np.vectorize(Hubble)(eta)
ax4.plot(a, aH, label='aH', color='black')
ax4.plot(a, a*M_GW, '-.', label=r'$aM_{GW}$', color='green')
print(k_0)
plt.vlines(x=a_0, ymin=1e0, ymax=k_0, color="black", linestyle="dashed",
           linewidth=1) 
ax4.text(a_0, .3, r"$a_0$", fontsize=15, weight="bold")

plt.vlines(x=a_c, ymin=1, ymax=k_c, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(a_c, .3, r"$a_c$", fontsize=15, weight="bold")
plt.hlines(y=k_0, xmin=1e-14,xmax=a_0, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(4e-15, k_0, r"$k_0$", fontsize=15, weight="bold")
plt.hlines(y=k_c, xmin=1e-14, xmax=a_c, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(4e-15, k_c, r"$k_c$", fontsize=15, weight="bold")

ax4.set_xlim(1e-14,5e0)
ax4.set_ylim(1e0, 5e10)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend(loc='best')


plt.savefig("emir/emir_kva_figs/fig1.pdf")
plt.show()