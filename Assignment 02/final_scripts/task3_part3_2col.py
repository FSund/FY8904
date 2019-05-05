import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, exp
from scipy.optimize import minimize

mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif Roman 2'

V0 = 1e3  # potential barrier in the middle
L = 1


def potential(x, L):
    if x < 0 or x > 1:
        return 1e300  # side walls
    else:
        if x > L/3 and x < 2*L/3:
            return V0  # middle barrier
        else:
            return 0


def eval(N=201):
    dx = L/(N - 1)
    dx2 = dx*dx

    x = np.linspace(0, L, N)

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N])

    # fill in A
    for i in range(N):
        xi = x[i + 1]
        V = potential(xi, L)

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

    return x, eigvals, eigvecs


def threeFour(lamb):
    """ Equation (3.4) """
    if lamb > V0:
        return (lamb - V0)**2
    k = sqrt(lamb)
    kappa = sqrt(V0 - lamb)
    p1 = exp(kappa/3)*(kappa*sin(k/3) + k*cos(k/3))**2
    p2 = exp(-kappa/3)*(kappa*sin(k/3) - k*cos(k/3))**2
    return p1 - p2


def f(l):
    return np.array([threeFour(lamb) for lamb in l])


n_its = 100  # production

## estimate how the number of eigenstates < V0 scales with the barrier height
n_minimums = []
V0s = np.linspace(0, 1e3, n_its)
for V0 in V0s:
    x, lambda_n, psi_n = eval(1001)

    lambda_n = lambda_n[lambda_n <= V0]

    mins = []
    for l in lambda_n:
        res = minimize(threeFour, l)
        mins.append(res.x[0])
    n_minimums.append(len(mins))

fig, axes = plt.subplots(2, 1, figsize=[3.45, 3])

ax = axes[0]
ax.plot(V0s, n_minimums)
ax.set_xlabel(r"$\nu_0$")
ax.set_ylabel(r"Eigenvalues below $\nu_0$")
ax.set_xlim([0, 1e3])

n_minimums = []
V0s = np.linspace(21.5, 23, n_its)
for V0 in V0s:
    x, lambda_n, psi_n = eval(1001)

    lambda_n = lambda_n[lambda_n <= V0]

    mins = []
    for l in lambda_n:
        res = minimize(threeFour, l)
        mins.append(res.x[0])
    n_minimums.append(len(mins))

ax = axes[1]
ax.plot(V0s, n_minimums)
ax.set_xlabel(r"$\nu_0$")
ax.set_ylabel(r"Eigenvalues below $\nu_0$")
ax.set_xlim([21.65, 22.75])

fig.tight_layout(pad=0)
fig.savefig("report_2col/figs/number_of_roots.pdf", bbox_inches="tight", pad=0)


plt.show()