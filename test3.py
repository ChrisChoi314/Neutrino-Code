import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

# Define the parameters
a = 1.0
b = 2.0
fv0 = .40523


# Define the function to be integrated
def f(x, y, w):
    # Calculate the integrand
    integrand = np.sin(x - w) * (y - np.interp(w, x, y)) / ((x - w) ** 4)
    # integrand = 0 # in the case that the RHS is treated as 0 as u >> 1
    return np.trapz(integrand, w)


# Define the function to solve the differential equation
def solve_ode(x, y):
    # Calculate the second derivative using the finite difference method
    dx = x[1] - x[0]
    ddy = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
    print("ddy dx", ddy, dx)

    # Calculate the first derivative using the central difference method
    dx = x[2] - x[0]
    dydx = (y[2:] - y[:-2]) / (2 * dx)
    print("dydx dx", dydx, dx)
    """
    # Calculate the integral using the trapezoidal rule
    integral = np.zeros(len(x))
    for i in range(1, len(x)):
        integral[i] = b/x[i]**2 * f(x[:i+1], y[:i+1], x[:i+1])
        integral[i] = 0
    """
    # Calculate the solution of the differential equation
    d2y = a / x[1:-1] * dydx + y[1:-1]  # + integral[1:-1]

    # Apply boundary conditions
    d2y = np.concatenate(([0], d2y, [0]))

    return d2y
x0 = 0.01
x_max = 20.0
x_max = 100
#x0 = 0.0001
#x_max = .01
N = 1000
x = np.linspace(x0, x_max, N)
def func(u,U):
    arr = np.zeros(N)
    max_idx_arr = np.where(x >= u )
    if len(max_idx_arr[0]) == 0: 
        max_idx = np.int64(N -1)
    else :
        max_idx = max_idx_arr[0][0]
    #arr = np.zeros(max_idx + 1)
    #arr = -np.sin(u-U) / (u- U)**3 - 3*np.cos(u-U) / (u- U)**4 + 3*np.sin(u-U) / (u- U)**5
    for idx, z in np.ndenumerate(arr):
        if idx > max_idx:
            arr[idx] = 0
        elif abs(u - z) < .001:
            arr[idx] = 1/15
        else: 
            arr[idx] = -np.sin(u-z) / (u- z)**3 - 3*np.cos(u-z) / (u- z)**4 + 3*np.sin(u-z) / (u- z)**5
    #print(arr)
    return arr

def M_derivs_homo(M,u):
    return [M[1], -4/ u *M[1] - M[0] ]
def M_derivs_inhomo(M,u): 
    return [M[1], -4/ u *M[1] - M[0] -24*fv0/u**2 * scipy.integrate.simps(M[1]*func(u,x))]
# Define the initial conditions

chi, chi_prime = odeint(M_derivs_homo, [1,0], x).T

integrand = func()

chiIH, chi_primeIH = odeint(M_derivs_inhomo, [1,0], x).T

y = np.zeros(N)
y[0] = 1.0
y[1] = .9

# Solve the differential equation
for i in range(1, N - 2):
    y[i + 1] = 2 * y[i] - y[i - 1]# + solve_ode(x[i : i + 3], y[i : i + 3])[1]

# Plot the solution
fig, (ax1) = plt.subplots(1)
ax1.set_xlabel("u")
#ax1.plot(x,y, color="black")
ax1.plot(x, chi,label="Homo", color = "blue")
ax1.set_ylabel(r"chi")
ax1.plot(x, chiIH, label = "inhomo", color = "red")
plt.legend()
plt.show()
