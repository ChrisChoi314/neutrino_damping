import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

# Define the parameters
fv0 = 0.40523
u0 = 0.01
u_max = 20.0
N = 1000
x = np.linspace(u0, u_max, N)
chi = 1
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
        -chi_prime[idx] * 2 / u
        - chi
        - 24 * fv0 / u**2 * scipy.integrate.simps(chi_prime * func(u), x)
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
        if idx > max_idx:
            arr[idx] = 0
        elif abs(u - z) < 0.0001:
            arr[idx] = 1 / 15
        # print( -np.sin(u-z) / (u- z)**3 - 3*np.cos(u-z) / (u- z)**4 + 3*np.sin(u-z) / (u- z)**5)
        else:
            arr[idx] = (
                -np.sin(u - z) / (u - z) ** 3
                - 3 * np.cos(u - z) / (u - z) ** 4
                + 3 * np.sin(u - z) / (u - z) ** 5
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
Chi, chi_prime = odeint(M_derivs_homo, [1, 0], x).T

# This method didn't work because I need the entire history of chi' up to the u
# I'm at, but it only gives me the single value of chi' for the current u, so I
# can't integrate it in the integral.
# chiIH, chi_primeIH = odeint(M_derivs_inhomo, [1, 0], x).T

# Plot the solutions
fig, (ax1) = plt.subplots(1)
ax1.set_xlabel("u")
ax1.plot(x, Chi, label="Homogeneous soln: chi = sin(u)/u", color="blue")
ax1.set_ylabel("chi")
# ax1.plot(x, chiIH, label = "inhomo", color = "red")
ax1.plot(x, chi_array, label="General soln using RK4", color="black")
ax1.plot(x, Weinberg(0.8026, 0.001, x), label="Weinberg's u>>1 soln", color="red")
plt.title("Solns. to the Diff eq of Chi in Short Wavelength Regime")
plt.legend()

ratio = chi_array/Chi
print(ratio)

#plt.savefig("Solutions_of_Diffeq.pdf")
plt.show()
