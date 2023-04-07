import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

# Define the parameters
k_GW = 0.02  # Mpc^-1
a_EQ = 1 / (1 + 3400)
H_0 = 1 / (4.28 * 0.7 * 1e3)  # Gpc^-1 / h_0 * (1e3 Mpc / 1 Gpc)^-1
omega_M = 0.27
omega_R = 8.24e-5
H_EQ = H_0 * (omega_R * a_EQ ** (-4) + omega_M * a_EQ ** (-3)) ** (1 / 2)
H_EQ_WB = H_0 * (2 * omega_M * a_EQ ** (-3)) ** 0.5
k_EQ = a_EQ * H_EQ
Q = np.sqrt(2) * k_GW / k_EQ
print(Q)
fv0 = 0.40523
u0 = 0.01
u_max = 1000
N = 1000
x = np.linspace(u0, u_max, N)
chi = 0
chi_prime = np.zeros(N)
chi_p_val = 0
chi_prime[0] = chi_p_val
u = u0
du = (u_max - u0) / N

# Define the equation. The equation looks like this:
# chi''(u) + 2/u chi'(u) + chi(u) = -24f_v(0)/u^2 integral from 0 to u of
#                                    ( -sin(u-U)/(u-U)^3 - 3cos(u-U)/(u-U)^4 + 3sin(u-U)/(u-U)^5 )chi'(U) dU


def f1(chi, chi_prime, u, idx):
    dchidu = chi_prime[idx]
    return dchidu


def f2(chi, chi_prime, u, idx):
    dchi_primedu = (
        -chi_prime[idx] * (2 / u + 1 / (2 * (1 + u)))
        - Q**2 * chi / (1 + u)
        - 24 * fv0 / u**2 / (1 + u) * scipy.integrate.simps(chi_prime * func(u), x)
    )
    return dchi_primedu


def func(u):
    arr = np.zeros(N)
    max_idx_arr = np.where(x >= u)
    if len(max_idx_arr[0]) == 0:
        max_idx = np.int64(N - 1)
    else:
        max_idx = max_idx_arr[0][0]
    # arr = np.zeros(max_idx + 1)
    # arr = -np.sin(u-U) / (u- U)**3 - 3*np.cos(u-U) / (u- U)**4 + 3*np.sin(u-U) / (u- U)**5
    for idx, z in np.ndenumerate(x):
        s = 2 * Q * (np.sqrt(1 + u) - np.sqrt(1 + z))
        if idx > max_idx:
            arr[idx] = 0
        elif abs(s) < 0.0001:
            arr[idx] = 1 / 15
        # print( -np.sin(u-z) / (u- z)**3 - 3*np.cos(u-z) / (u- z)**4 + 3*np.sin(u-z) / (u- z)**5)
        else:
            arr[idx] = (
                -np.sin(s) / (s) ** 3
                - 3 * np.cos(s) / (s) ** 4
                + 3 * np.sin(s) / (s) ** 5
            )
    # print(arr)
    return arr


# Runge Kutta 4 for solving the diff eq
chi_array = np.zeros(N)
for idx in range(N):
    chi_array[idx] = chi
    chi_prime[idx] = chi_p_val
    k11 = du * f1(chi, chi_prime, u, idx)
    k21 = du * f2(chi, chi_prime, u, idx)
    chi_prime[idx] += 0.5 * k21
    k12 = du * f1(chi + 0.5 * k11, chi_prime, u + 0.5 * du, idx)
    k22 = du * f2(chi + 0.5 * k11, chi_prime, u + 0.5 * du, idx)
    chi_prime[idx] = chi_p_val
    chi_prime[idx] += 0.5 * k22
    k13 = du * f1(chi + 0.5 * k12, chi_prime, u + 0.5 * du, idx)
    k23 = du * f2(chi + 0.5 * k12, chi_prime, u + 0.5 * du, idx)
    chi_prime[idx] = chi_p_val
    chi_prime[idx] += 0.5 * k23
    k14 = du * f1(chi + k13, chi_prime, u + du, idx)
    k24 = du * f2(chi + k13, chi_prime, u + du, idx)
    chi_prime[idx] = chi_p_val
    chi += (k11 + 2 * k12 + 2 * k13 + k14) / 6
    chi_p_val += (k21 + 2 * k22 + 2 * k23 + k24) / 6
    u += du


def u_to_y(u):
    return (u / (2 * Q) + 1) ** 2 - 1


# The input for the homogeneous version of the equation, just chi''(u) + 2/u chi'(u) + chi(u) = 0
def M_derivs_homo(M, u):
    return [M[1], -2 / u * M[1] - M[0]]


# My attempt to try to use the method of breaking a 2nd order diffeq into 2 first order diffeq.
def M_derivs_inhomo(M, u):
    integral = scipy.integrate.cumtrapz(M[1] * func(u), x, initial=0)
    integral = scipy.integrate.simps(M[1] * func(u))
    return [M[1], -4 / u * M[1] - M[0] - 24 * fv0 / u**2 * integral]


# What Weinberg said the solution approaches for u >> 1, equation (24) in paper
def Weinberg(A, delta, u):
    return A * np.sin(u + delta) / u


# Get the homogeneous solution using scipy.integrate.odeint
Chi, chi_prime = odeint(M_derivs_homo, [u_to_y(1), u_to_y(0)], x).T
print(u_to_y(1), u_to_y(0))

# This method didn't work because I need the entire history of chi' up to the u
# I'm at, but it only gives me the single value of chi' for the current u, so I
# can't integrate it in the integral.
# chiIH, chi_primeIH = odeint(M_derivs_inhomo, [1, 0], x).T

# Plot the solutions
fig, (ax1) = plt.subplots(1)
ax1.set_xlabel("y")
ax1.plot(
    x, Chi, label="Homogeneous soln for small wavelength: chi = sin(y)/y", color="blue"
)
ax1.set_ylabel("chi(y)")
# ax1.plot(x, chiIH, label = "inhomo", color = "red")
ax1.plot(x, chi_array, label="General soln using RK4", color="black")

r_H = 314  # Mpc
a_GW_Reenter = (r_H * H_0 * np.sqrt(omega_M)) ** (1 / 2)
y_GW_Reenter = a_GW_Reenter / a_EQ
print(y_GW_Reenter, a_GW_Reenter)
ax1.vlines(
    [y_GW_Reenter],
    0,
    1,
    linestyles="dashed",
    label="y when Primordial GW Reenter Horizon",
    colors="red",
)
# ax1.plot(x, Weinberg(0.8026, 0.001, x), label="Weinberg's u>>1 soln", color="red")
plt.title("Solns. to the Diff eq of Chi(y) for k = k_primordial_GW")
plt.legend()
plt.savefig("Primoridal Reenter.pdf")
plt.show()
