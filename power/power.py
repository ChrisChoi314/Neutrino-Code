import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

# Define the parameters
H_inf = (
    1  # 1e14  # in gev, according to https://cds.cern.ch/record/1420368/files/207.pdf
)
a_r = 1  # 1e-32 # preliminary value, will change later
tau_r = 1 / (a_r * H_inf)
M_pl = 1
k = .01
tau_m = (   
    10 * tau_r
)  # tau_m will be before matter-rad equality, but after inflation ends https://arxiv.org/pdf/1808.02381.pdf
# 1e-30 # from Claude de Rham paper https://arxiv.org/pdf/1401.4173.pdf
# m = 1*np.sqrt(2) - .1
# m = 1*np.sqrt(2) + .00001

# For the k = 0.01 graph, i used tau_m = 10tau_r, tau_r = 10 / ..., tau_init = -2000*tau_r, and tau_end = 100_tau_m 


tau_end = 10 * tau_m
m = 0.8
nu = np.sqrt(9 / 4 - m**2 / H_inf**2)
N = 50000
tau_init = -200 * tau_r
# tau = np.linspace(tau_init, tau_m + tau_r, N)
tau = np.linspace(tau_init, tau_end, N)
N_neg_r = 0
N_pos_r = 0
N_m = 0
N_end = 0
for val in tau:
    if val <= -tau_r:
        N_neg_r += 1
    if val <= tau_r:
        N_pos_r += 1
    if val <= tau_m:
        N_m += 1
    if val <= -tau_end:
        N_end += 1

isReal = False


def scale_fac(conf_time):
    if conf_time < tau_r:
        return -1 / (H_inf * conf_time)
    else:
        return a_r * conf_time / tau_r


def d_scale_fac_dz(conf_time):
    if conf_time < tau_r:
        return -2 / (H_inf * conf_time**3)
    else:
        return 0


def mu(conf_time):
    if conf_time < tau_m:
        return m
    else:
        return 0


# The input for the homogeneous version of the equation, just chi''(u) + 2/u chi'(u) + chi(u) = 0
def M_derivs_homo(M, u):
    return [
        M[1],
        -(k**2 + (scale_fac(u) * mu(u)) ** 2 - d_scale_fac_dz(u) / scale_fac(u))
        * M[0],
    ]


fig, (ax1, ax2) = plt.subplots(2, figsize=(22, 14))


# horrors beyond my comprehension for the central difference method
xxx = np.array([tau[0] - 0.0000001, tau[0] + 0.0000001])
xx = (np.sqrt(-np.pi * xxx) / 2 * scipy.special.hankel1(nu, -k * xxx))[:2]
xx = xx[1] - xx[0]
xx /= xxx[1] - xxx[0]
print(xx)


# Get the homogeneous solution using scipy.integrate.odeint
v_0 = np.sqrt(-np.pi * tau[0]) / 2 * scipy.special.hankel1(nu, -k * tau[0]).imag
v_prime_0 = xx.imag


v, v_prime = odeint(M_derivs_homo, [v_0, v_prime_0], tau[0:N_neg_r]).T


ax1.plot(
    tau[:N_neg_r],
    4
    * k**3
    * abs(v) ** 2
    / (np.pi * M_pl * np.vectorize(scale_fac)(tau[:N_neg_r])) ** 2,
    label=r"Numerical solution",
    color="black",
)

v_k_reg1 = (
    -1j
    * np.sqrt(-np.pi * tau[0:N_neg_r])
    / 2
    * scipy.special.hankel1(nu, -k * tau[0:N_neg_r])
)
ax1.plot(
    tau[:N_neg_r],
    4
    * k**3
    * abs(v_k_reg1) ** 2
    / (np.pi * M_pl * np.vectorize(scale_fac)(tau[:N_neg_r])) ** 2,
    "--",
    color="orange",
    label=r"Analyt Soln: $\frac{\sqrt{-\pi \tau}}{2} H_\nu^{(1)} (-k\tau)$",
)


v_0_rest = v[N_neg_r - 1]
xxxxx = np.array([tau[N_neg_r - 1] - 0.0000001, tau[N_neg_r - 1] + 0.0000001])
xxxx = (-1j*np.sqrt(-np.pi * xxxxx) / 2 * scipy.special.hankel1(nu, -k * xxxxx))[:2]
xxxx = xxxx[1] - xxxx[0]
xxxx /= xxxxx[1] - xxxxx[0]
v_prime_0_rest = xxxx
print(xxxx)
v_rest, v_prime_rest = odeint(
    M_derivs_homo, [v_0_rest, v_prime_0_rest], tau[N_pos_r:]
).T


ax2.plot(
    tau[N_pos_r:],
    4
    * k**3
    * abs(v_rest) ** 2
    / (np.pi * M_pl * np.vectorize(scale_fac)(tau[N_pos_r:])) ** 2,
    color="black",
)

ax1.set_xlabel(r"$\tau$ (conformal time)")

C_1 = (
    -1j
    * np.sqrt(np.pi)
    * 2 ** (-7 / 2 + nu)
    * (k * tau_r) ** (-nu)
    * scipy.special.gamma(nu)
    * (
        2 * m / H_inf * scipy.special.jv(-3 / 4, m / (2 * H_inf))
        + (1 - 2 * nu) * scipy.special.jv(1 / 4, m / (2 * H_inf))
    )
)
C_2 = (
    -1j
    * np.sqrt(np.pi)
    * 2 ** (-7 / 2 + nu)
    * (k * tau_r) ** (-nu)
    * scipy.special.gamma(nu)
    * (
        2 * m / H_inf * scipy.special.jv(3 / 4, m / (2 * H_inf))
        - (1 - 2 * nu) * scipy.special.jv(-1 / 4, m / (2 * H_inf))
    )
)
v_k_reg2 = (
    2
    / np.sqrt(np.pi * a_r * tau[N_pos_r:N_m] / tau_r * m)
    * (
        C_1
        * np.cos(a_r * tau[N_pos_r:N_m] / tau_r * m * tau[N_pos_r:N_m] / 2 - np.pi / 8)
        + C_2
        * np.sin(a_r * tau[N_pos_r:N_m] / tau_r * m * tau[N_pos_r:N_m] / 2 + np.pi / 8)
    )
)


ax2.plot(
    tau[N_pos_r:N_m],
    4
    * k**3
    * abs(v_k_reg2) ** 2
    / (np.pi * M_pl * np.vectorize(scale_fac)(tau[N_pos_r:N_m])) ** 2,
    "--",
    color="cyan",
    label=r"Analyt Solution: $\frac{2}{\sqrt{\pi am}}[C_1 \cos(\frac{am\tau}{2} - \frac{\pi}{8}) + C_2 \sin(\frac{am\tau}{2} + \frac{\pi}{8})]$",
)

lamb = m * tau_m**2 / (2 * H_inf * tau_r**2)
D_1 = -np.sin(k * tau_m) * (
    C_2 * np.cos(lamb + np.pi / 8) - C_1 * np.sin(lamb - np.pi / 8)
)
D_2 = np.cos(k * tau_m) * (
    C_2 * np.cos(lamb + np.pi / 8) - C_1 * np.sin(lamb - np.pi / 8)
)
v_k_reg3 = (
    2
    / k
    * np.sqrt(m * tau_m / (np.pi * H_inf * tau_r**2))
    * (
        D_1 * np.cos(k * tau[N_m:])
        + D_2 * np.sin(k * tau[N_m:])
    )
)
ax2.plot(
    tau[N_m:],
    4
    * k**3
    * abs(v_k_reg3) ** 2
    / (np.pi * M_pl * np.vectorize(scale_fac)(tau[N_m:])) ** 2,
    "--",
    color="magenta",
    label=r"Analyt Solution: $\frac{2}{k}\sqrt{\frac{m\tau_m}{\pi H_{inf} \tau_r^2}}[D_1 \cos(k\tau) + D_2 \sin(k\tau)]$",
)

ax1.set_xlim(-tau_end, tau_end)
ax2.set_xlim(-tau_end, tau_end)
ax1.set_ylim(0, np.max(4
    * k**3
    * abs(v[N_end:]) ** 2
    / (np.pi * M_pl * np.vectorize(scale_fac)(tau[N_end:N_neg_r])) ** 2))
ax1.set_ylabel(r"$v_k(\tau)$")


plt.axvline(x=-tau_r, color="red", linestyle="dashed", linewidth=1)
plt.axvline(
    x=tau_r, color="red", linestyle="dashed", linewidth=1, label=r"$\pm \tau_r$"
)
plt.axvline(x=tau_m, color="blue", linestyle="dashed", linewidth=1, label=r"$\tau_m$")

plt.title(r"Power Spectrum of Graviton$")

plt.legend()
#plt.savefig("power-spectrum-k0-01.pdf")
plt.show()
