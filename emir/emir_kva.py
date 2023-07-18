import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *
import math


fig, (ax4) = plt.subplots(1) #, figsize=(22, 14))
N=1000
eta = np.logspace(-14, .25, N)
a = np.vectorize(scale_fac)(eta)
a = np.logspace(-14,0, N)
aH = a*np.vectorize(Hubble_a)(a)
ax4.plot(a, aH, label='aH', color='black')
ax4.plot(a, a*M_GW, '-.', label=r'$aM_{GW}$', color='green')
print(k_0)
plt.vlines(x=a_0, ymin=1e-24, ymax=k_0, color="black", linestyle="dashed",
           linewidth=1) 
ax4.text(a_0, 4e-25, r"$a_0$", fontsize=15, weight="bold")

plt.vlines(x=a_c, ymin=1e-24, ymax=k_c, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(a_c,4e-25, r"$a_c$", fontsize=15, weight="bold")
plt.hlines(y=k_0, xmin=1e-14,xmax=a_0, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(2e-15, k_0, r"$k_0$", fontsize=15, weight="bold")
plt.hlines(y=k_c, xmin=1e-14, xmax=a_c, color="black", linestyle="dashed",
            linewidth=1)
ax4.text(2e-15, k_c, r"$k_c$", fontsize=15, weight="bold")

ax4.text(1e-14, 1e-24, r"$aM_{GW}$", fontsize=15, weight="bold", color = 'green')

ax4.text(1e-13, 1e-6, r"aH", fontsize=15)

#ax4.set_xlim(1e-14,5e0)
#ax4.set_ylim(1e0, 5e10)
ax4.set_xscale('log')
ax4.set_yscale('log')
# ax4.legend(loc='best')


ax4.set_xlabel(r"$a$")
ax4.title.set_text(r'Evolution of aH and a$M_{GW}$')

plt.savefig("emir/emir_kva_figs/fig2.pdf")
plt.show()