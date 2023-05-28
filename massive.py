import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

# Define the parameters
H_inf = 1  # 1e14  # in gev, according to https://cds.cern.ch/record/1420368/files/207.pdf
a_r = 1  # 1e-32 # preliminary value, will change later
tau_r = 1/(a_r*H_inf)
k = 1
tau_m = 2*tau_r  # tau_m will be before matter-rad equality, but after inflation ends https://arxiv.org/pdf/1808.02381.pdf
m = 1  # 1e-30 # from Claude de Rham paper https://arxiv.org/pdf/1401.4173.pdf

N = 1000
tau = np.linspace(.000001, tau_m + tau_r, N)


def scale_fac(conf_time):
    if conf_time < tau_r:
        return -1/(H_inf*conf_time)
    else:
        return a_r * conf_time / tau_r


def d_scale_fac_dz(conf_time):
    if conf_time < tau_r:
        return -2/(H_inf*conf_time**3)
    else:
        return 0


def mu(conf_time):
    if conf_time < tau_m:
        return m
    else:
        return 0


N = 1000


# The input for the homogeneous version of the equation, just chi''(u) + 2/u chi'(u) + chi(u) = 0
def M_derivs_homo(M, u):
    return [M[1], -(k**2 + (scale_fac(u)*m)**2 - d_scale_fac_dz(u) / scale_fac(u)) * M[0]]


# Get the homogeneous solution using scipy.integrate.odeint
v, v_prime = odeint(M_derivs_homo, [1, 0], tau).T


# Plot the solutions
fig, (ax1) = plt.subplots(1)
ax1.set_xlabel(r"$\tau$ (conformal time)")
ax1.plot(
    tau, v_prime, label=r"Mode function $v_k(\tau)$", color="black"
)
tau_up_to_r = np.empty([0])
for val in tau:
    if val < tau_r:
        tau_up_to_r = np.append(tau_up_to_r, val)
print(tau_up_to_r)
print(np.sqrt(9/4 - m**2 / H_inf**2))
ax1.plot(
    tau_up_to_r, np.sqrt(np.pi*tau_up_to_r) / 2 * scipy.special.hankel1(np.sqrt(9/4 - m**2 / H_inf**2), tau_up_to_r), label=r"Analytical Sol for mode func, in reg 1", color="orange"
)

#np.sqrt(np.pi*tau_up_to_r) / 2

ax1.set_ylabel(r"$v_k(\tau)$")

plt.axvline(x=tau_r, color='red', linestyle='dashed',
            linewidth=1, label='Tau_r')
plt.axvline(x=tau_m, color='blue', linestyle='dashed',
            linewidth=1, label='Tau_m')

plt.title(r"Solns. to the Diff eq of $v_k(\tau)$")
plt.legend()
# plt.savefig("mode_function_blue_tilt.pdf")
plt.show()
