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

#x0 = 0.0001
#x_max = .01
N = 100 
x = np.linspace(x0, x_max, N)


chi= 1
chi_prime= np.zeros(N)
chi_prime[0] = 0
chi_p_val = 0
u=x0
du=(x_max - x0)/N

def f1(chi,chi_prime,u,idx):
  dchidu = chi_prime[idx]
  return dchidu

def f2(chi,chi_prime,u, idx):
    #print(scipy.integrate.simps(chi_prime*func(u)) , print(func(u)))
    dchi_primedu = -chi_prime[idx]*2/ u - chi - 24*fv0 /u**2*scipy.integrate.simps(chi_prime*func(u),x)
    return dchi_primedu

def func(u):
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
        elif abs(u - z) < .00001:
            arr[idx] = 1/15
        else: 
            arr[idx] = -np.sin(u-z) / (u- z)**3 - 3*np.cos(u-z) / (u- z)**4 + 3*np.sin(u-z) / (u- z)**5
    #print(arr)
    return arr
chi_array = np.zeros(N)
for idx in range(N):
    chi_array[idx] = chi
    chi_prime[idx] = chi_p_val
    k11 = du*f1(chi,chi_prime,u,idx)
    k21 = du*f2(chi,chi_prime,u,idx)
    chi_prime[idx] += 0.5*k21
    k12 = du*f1(chi+0.5*k11,chi_prime,u+0.5*du,idx)
    k22 = du*f2(chi+0.5*k11,chi_prime,u+0.5*du,idx)
    chi_prime[idx] = chi_p_val
    chi_prime[idx] += 0.5*k22
    k13 = du*f1(chi+0.5*k12,chi_prime,u+0.5*du,idx)
    k23 = du*f2(chi+0.5*k12,chi_prime,u+0.5*du,idx)
    chi_prime[idx] = chi_p_val
    chi_prime[idx] += 0.5*k23
    k14 = du*f1(chi+k13,chi_prime,u+du,idx)
    k24 = du*f2(chi+k13,chi_prime,u+du,idx)
    chi_prime[idx] = chi_p_val
    chi += (k11+2*k12+2*k13+k14)/6
    chi_p_val += (k21+2*k22+2*k23+k24)/6
    #print(chi_p_val)
    u += du

def truncate_x(u,x):
    arr = np.zeros(N)
    max_idx_arr = np.where(x >= u )
    if len(max_idx_arr[0]) == 0: 
        max_idx = np.int64(N -1)
    else:
        max_idx = max_idx_arr[0][0]
    for idx, z in np.ndenumerate(x):
        if idx > max_idx:
            arr[idx] = 0
        else:
            arr[idx] = x[idx]
    return arr
def integrals(u,x):
    arr = np.zeros(N)
    for idx, z in np.ndenumerate(arr):
        #arr[idx] = scipy.integrate.simps(M[1]*func(u,x), truncate_x(u,x))
        x =1
def M_derivs_homo(M,u):
    return [M[1], -2/ u *M[1] - M[0] ]
def M_derivs_inhomo(M,u): 
    integral = scipy.integrate.cumtrapz(M[1]*func(u),x, initial = 0)
    integral = scipy.integrate.simps(M[1]*func(u))
    return [M[1], -4/ u *M[1] - M[0] -24*fv0/u**2 * integral]
# Define the initial conditions

Chi, chi_prime = odeint(M_derivs_homo, [1,0], x).T

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
ax1.plot(x, Chi,label="Homo", color = "blue")
ax1.set_ylabel(r"chi")
#ax1.plot(x, chiIH, label = "inhomo", color = "red")
ax1.plot(x, chi_array, label = "RK4", color = "black")
plt.legend()
plt.show()
