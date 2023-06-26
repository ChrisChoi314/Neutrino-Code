import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *

N = 2000
eta = np.logspace(-7,1,N)
omega_0 = k_0/a_0
k_prime = a_0*omega_0 # seems a bit circular, but that is how Mukohyama defined it in his paper
eq_idx = 0
H_square = np.vectorize(Hubble)(eta)**2
ang_freq_square = np.vectorize(ang_freq)(eta)**2
for i in range(0, len(eta)):
    if H_square[i] <= ang_freq_square[i]:
        eta_k = eta[i]
        eq_idx = i
        break
k = a_0*np.sqrt(omega_0**2-M_GW**2)

S = k_prime* a_k / (k* a_k_prime_GR)*np.sqrt(omega_k*a_k/(omega_0*a_0))