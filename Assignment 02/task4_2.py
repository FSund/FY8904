import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, exp

# set up plot look
mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
# use r"$x^3_i$" to produce math using cm font
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

L = 1


def _detuned_well(x, L, V0, Vr):
    if x < 0 or x > L:
        return 1e300  # side walls
    else:
        if x < L/3:
            return 0  # left third
        elif x < 2*L/3:
            return V0  # middle third
        else:
            return Vr  # right third


def potential(x, L, V0, Vr):
    # return _potential_well(x, L)
    # return _potential_barrier(x, L, V0)
    return _detuned_well(x, L, V0, Vr)


def eval(N=201, V0=0, Vr=0):
    dx = L/(N - 1)
    dx2 = dx*dx

    x = np.linspace(0, L, N)

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N], dtype=np.complex_)

    # fill in A
    for i in range(N):
        V = potential(x[i + 1], L, V0, Vr)

        A[i, i] = 2/dx2 + V
        if i > 0:  # skip first row
            A[i, i-1] = -1/dx2
        if i < N-1:  # skip last row
            A[i, i+1] = -1/dx2

    eigvals, eigvecs = np.linalg.eigh(A)  # returns sorted eigvals and vecs

    # add boundary conditions (zeros)
    eigvecs = np.append(np.zeros([1, N]), eigvecs, axis=0)
    eigvecs = np.append(eigvecs, np.zeros([1, N]), axis=0)

    # normalize
    eigvecs /= sqrt(dx)

    # fix sign
    # for i in range(eigvecs.shape[1]):
    #     if eigvecs[1, i] < 0:
    #         eigvecs[:, i] *= -1

    return x, eigvals, eigvecs


## task 4.2
# calculate integral in eq. (4.3)
N = 101
V0 = 100
nV = 101
Vrs = np.linspace(-100, 100, nV)
lambda_r = np.zeros([2, nV])
tau = np.zeros([nV], dtype=np.complex_)
for idx, Vr in enumerate(Vrs):
    x, lambda_n, psi_n = eval(N, V0, Vr)
    lambda_r[:, idx] = lambda_n[0:2]
    dx = x[1] - x[0]
    dx2 = dx*dx

    psi0 = psi_n[:, 0]
    psi1 = psi_n[:, 1]

    integrand = np.zeros([N], dtype=np.complex_)
    for i in range(len(integrand)):
        if i == 0:
            # psi is zero outside well
            double_derivative = (psi1[i + 1] - 2*psi1[i] + 0)/dx2
        elif i == len(integrand) - 1:
            # psi is zero outside well
            double_derivative = (0 - 2*psi1[i] + psi1[i - 1])/dx2
        else:
            double_derivative = (psi1[i + 1] - 2*psi1[i] + psi1[i - 1])/dx2

        integrand[i] = psi0[i]*(psi1[i]*potential(x[i], L, V0, Vr) - double_derivative)
    tau[idx] = np.trapz(integrand, x)

    if Vr == 0:
        fig, ax = plt.subplots()
        # ax.plot(x, integrand)
        ax.plot(x, np.real(psi0), label="Re psi0")
        ax.plot(x, np.real(psi1), label="Re psi1")
        ax.plot(x, np.imag(psi0), label="Im psi0")
        ax.plot(x, np.imag(psi1), label="Im psi1")
        ax.plot(x, psi0**2, label="psi0**2")
        ax.plot(x, psi1**2, label="psi1**2")
        ax.legend()
        ax.set_title("Vr = %f" % Vr)
        plt.show()
        breakpoint()


fig, ax = plt.subplots()
ax.plot(Vrs, lambda_r[0, :], label=r"$\lambda_0$")
ax.plot(Vrs, lambda_r[1, :], label=r"$\lambda_1$")
ax.legend()
ax.set_xlabel(r"$\nu_r$")
ax.set_ylabel(r"$\lambda_n(\nu_r)$")
ax.set_xlim([Vrs[0], Vrs[-1]])

fig, ax = plt.subplots()
ax.plot(Vrs, tau)
# ax.legend()
ax.set_xlabel(r"$\nu_r$")
ax.set_ylabel(r"$\tau(\nu_r)$")
ax.set_xlim([Vrs[0], Vrs[-1]])


plt.show()
