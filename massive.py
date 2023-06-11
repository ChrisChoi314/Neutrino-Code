import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

# Define the parameters
H_inf = 1  # 1e14  # in gev, according to https://cds.cern.ch/record/1420368/files/207.pdf
a_r = 1  # 1e-32 # preliminary value, will change later
tau_r = 1/(a_r*H_inf)
k = 10
tau_m = 1e10*tau_r  # 5*tau_r tau_m will be before matter-rad equality, but after inflation ends https://arxiv.org/pdf/1808.02381.pdf
# 1e-30 # from Claude de Rham paper https://arxiv.org/pdf/1401.4173.pdf
# m = 1*np.sqrt(2) - .1
# m = 1*np.sqrt(2) + .00001
m = .8
nu = np.sqrt(9/4 - m**2 / H_inf**2)
N = 10000
tau_init = -2000*tau_r
# tau = np.linspace(tau_init, tau_m + tau_r, N)
tau = np.logspace(tau_init, -tau_r, N)

isReal = False


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


# The input for the homogeneous version of the equation, just chi''(u) + 2/u chi'(u) + chi(u) = 0
def M_derivs_homo(M, u):
    return [M[1], -(k**2 + (scale_fac(u)*mu(u))**2 - d_scale_fac_dz(u) / scale_fac(u)) * M[0]]


fig, (ax1) = plt.subplots(1, figsize=(22, 14))

tau_up_to_r = np.empty([0])
for val in tau:
    if val < tau_r:
        tau_up_to_r = np.append(tau_up_to_r, val)


# horrors beyond my comprehension for the central difference method
xxx = np.array([tau_up_to_r[0] - 0.0000001, tau_up_to_r[0] + 0.0000001])
xx = (np.sqrt(-np.pi*xxx) / 2 * scipy.special.hankel1(nu, -k*xxx))[:2]
xx = xx[1] - xx[0]
xx /= xxx[1] - xxx[0]
print(xx)
tau_up_to_r = np.logspace(tau_init, -tau_r, N)


# Get the homogeneous solution using scipy.integrate.odeint
if isReal:
    v_0 = np.sqrt(-np.pi*tau_up_to_r[0]) / 2 * \
        scipy.special.hankel1(nu, -k*tau_up_to_r[0])
    v_prime_0 = xx
else:
    v_0 = np.sqrt(-np.pi*tau_up_to_r[0]) / 2 * \
        scipy.special.hankel1(nu, -k*tau_up_to_r[0]).imag
    v_prime_0 = xx.imag


v, v_prime = odeint(M_derivs_homo, [v_0, v_prime_0], tau).T

'''
ax1.plot(
    tau, v, label=r"Numerical solution", color="black"
)

if isReal:
    ax1.plot(
        tau_up_to_r, np.sqrt(-np.pi*tau_up_to_r) / 2 *
        scipy.special.hankel1(nu, -k*tau_up_to_r), "--", color="orange",
        label=r"Analyt Soln: $\frac{\sqrt{-\pi \tau}}{2} H_\nu^{(1)} (-k\tau)$"
    )
else:
    ax1.plot(
        tau_up_to_r, -1j*np.sqrt(-np.pi*tau_up_to_r) / 2 *
        scipy.special.hankel1(nu, -k*tau_up_to_r), "--", color="orange",
        label=r"Analyt Soln: $\frac{\sqrt{-\pi \tau}}{2} H_\nu^{(1)} (-k\tau)$"
    )
'''

tau_end = 1e10*tau_m
tau_rest = np.logspace(tau_r, tau_end, N)
v_0_rest = v[N-1]
xxxxx = np.array([tau_up_to_r[N-1] - 0.0000001, tau_up_to_r[N-1] + 0.0000001])
if isReal:
    xxxx = (np.sqrt(-np.pi*xxxxx) / 2 *
            scipy.special.hankel1(nu, -k*xxxxx))[:2]
else:
    xxxx = -1j*(np.sqrt(-np.pi*xxxxx) / 2 *
                scipy.special.hankel1(nu, -k*xxxxx))[:2]
xxxx = xxxx[1] - xxxx[0]
xxxx /= xxxxx[1] - xxxxx[0]
v_prime_0_rest = xxxx
print(xxxx)
v_rest, v_prime_rest = odeint(
    M_derivs_homo, [v_0_rest, v_prime_0_rest], tau_rest).T


ax1.plot(
    tau_rest, v_rest, color="black"
)

ax1.set_xlabel(r"$\tau$ (conformal time)")

tau_r_to_m = np.empty([0])
for val in tau_rest:
    if val <= tau_m:
        tau_r_to_m = np.append(tau_r_to_m, val)

C_1 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /
                                                                                  H_inf * scipy.special.jv(-3/4, m/(2*H_inf)) + (1-2*nu)*scipy.special.jv(1/4, m/(2*H_inf)))
C_2 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /
                                                                                  H_inf * scipy.special.jv(3/4, m/(2*H_inf)) - (1-2*nu)*scipy.special.jv(-1/4, m/(2*H_inf)))
v_k_reg2 = 2/np.sqrt(np.pi*a_r * tau_r_to_m / tau_r*m)*(C_1*np.cos(a_r * tau_r_to_m / tau_r *
                                                                   m*tau_r_to_m/2 - np.pi/8) + C_2*np.sin(a_r * tau_r_to_m / tau_r*m*tau_r_to_m/2 + np.pi/8))


ax1.plot(
    tau_r_to_m, -1j*v_k_reg2, "--", color="cyan",
    label=r"Analyt Solution: $\frac{2}{\sqrt{\pi am}}[C_1 \cos(\frac{am\tau}{2} - \frac{\pi}{8}) + C_2 \sin(\frac{am\tau}{2} + \frac{\pi}{8})]$"
)

tau_m_to_end = np.empty([0])
for val in tau_rest:
    if val >= tau_m:
        tau_m_to_end = np.append(tau_m_to_end, val)
lamb = m*tau_m**2 / (2*H_inf*tau_r**2)
D_1 = -np.sin(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
D_2 = np.cos(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
v_k_reg3 = 2/k*np.sqrt(m*tau_m / (np.pi*H_inf*tau_r**2)) * \
    (D_1*np.cos(k*tau_m_to_end) + D_2*np.sin(k*tau_m_to_end))
ax1.plot(
    tau_m_to_end, -1j*v_k_reg3, "--",
    color="magenta",
    label=r"Analyt Solution: $\frac{2}{k}\sqrt{\frac{m\tau_m}{\pi H_{inf} \tau_r^2}}[D_1 \cos(k\tau) + D_2 \sin(k\tau)]$"
)

print(lamb)
print(a_r * tau_m / tau_r *m*tau_m/2 )

ax1.set_xlim(-tau_end, tau_end)
ax1.set_xlim(tau_r, tau_end)
ax1.set_ylabel(r"$v_k(\tau)$")
ax1.set_xscale('log')

'''
plt.axvline(x=-tau_r, color='red', linestyle='dashed',
            linewidth=1)
 '''       
plt.axvline(x=tau_r, color='red', linestyle='dashed',
            linewidth=1, label=r'$\pm \tau_r$')
plt.axvline(x=tau_m, color='blue', linestyle='dashed',
            linewidth=1, label=r'$\tau_m$')

plt.title(r"Solns. to the Diff eq of $v_k(\tau)$")

'''
fig, (ax2) = plt.subplots(1)
time_start = -1.e9
time = np.linspace(time_start, time_start+50, N)

# time = np.linspace(-.5, -.00001, N, dtype=np.complex_)
ax2.plot(
    time, 1j*1/np.sqrt(2*k)*np.exp(-1j*k*(time + np.pi)), label=r"Bunch-Davies Result: $\frac{1}{\sqrt{2k}} e^{-ik\tau}$", color='red'
)

ax2.plot(
    time, 1j*np.sqrt(-np.pi*time) / 2 * scipy.special.hankel1(nu, -k*time), label=r"Analytical Solution: $\frac{\sqrt{-\pi \tau}}{2} H_\nu^{(1)} (-k\tau)$", color='blue'
)
reference_init = np.empty(len(time[0:100]))
reference_init.fill(time_start)
print(time[0:100] - reference_init)

print((scipy.special.hankel1(nu, -k*time))[0:100])
# print(time**(1/2 - nu)*k**(-nu))
# ax2.plot(
#    time, time**(1/2 - nu)*k**(-nu), label=r"Superhorizon lim (tau -> 0): $\tau^{\frac{1}{2} - \nu} k^{-\nu}$", color='green'
# )

ax2.set_xlabel(r"$\tau$")
# print((time**(1/2 - nu)*k**(-nu))[N-10:N-1])
# print(time[N-10:N-1])
plt.title(r"Comparison betw. Analytical soln. and expected lim, complex")
'''

plt.legend()
# plt.savefig("fixed-mode-func-small-k.pdf")
plt.show()
