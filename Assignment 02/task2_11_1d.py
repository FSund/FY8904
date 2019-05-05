import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos


mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})
plt.close("all")


def eval(N=201):
    L = 1
    dx = L/(N-1)
    dx2 = dx*dx
    x = np.linspace(0, L, N)

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N])

    for i in range(N):
        A[i, i] = 2/dx2
        if i > 0:  # skip first row
            A[i, i-1] = -1/dx2
        if i < N-1:  # skip last row
            A[i, i+1] = -1/dx2

    # eigh is for symmetric matrices, and returns sorted eigenvalues/vectors
    eigvals, eigvecs = np.linalg.eigh(A)

    # add boundary conditions (zeros)
    eigvecs = np.append(np.zeros([1, N]), eigvecs, axis=0)
    eigvecs = np.append(eigvecs, np.zeros([1, N]), axis=0)

    # sort according to size of eigenvalues
    # eigvecs = eigvecs[:, np.argsort(eigvals)]
    # eigvals = np.sort(eigvals)

    # normalize
    eigvecs /= sqrt(dx)

    # fix sign
    for i in range(eigvecs.shape[1]):
        if eigvecs[1, i] < 0:
            eigvecs[:, i] *= -1

    return x, eigvals, eigvecs


## solve the time-dependent Schrödinger equation
N = 501
x, eigvals, psi_n = eval(N)
nSteps = 100
dt = 0.01/1000
n = len(eigvals)
nHalf = round(n/2)

# set up initial condition
alpha = np.zeros([n])

# delta function
psi0 = np.zeros([N])
psi0[nHalf] = 1  # delta function
for i in range(n):  # loop over eigenvalues
    # alpha[i] = np.trapz(psi_n[:, i]*psi0, x)

    # analytical result of int psi_n(x)*delta(x-0.5)dx == psi_n(x=0.5)
    alpha[i] = psi_n[nHalf, i]

psi = np.zeros([N, nSteps], dtype=np.complex_)
for i in range(nSteps):  # loop over time steps
    t = dt*i
    psi[:, i] = np.sum(alpha*np.exp(-1j*eigvals*t)*psi_n, axis=1)

psi2 = np.abs(psi**2)
fig, axes = plt.subplots(2, 2, sharex=True, figsize=[7, 3.5])
axes = [item for sublist in axes for item in sublist]
axes[0].plot(x, psi2[:, 0])
axes[1].plot(x, psi2[:, 1])
axes[2].plot(x, psi2[:, 20])
axes[3].plot(x, psi2[:, 40])
for ax in axes[0:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
for ax in axes:
    ax.set_xlim([0, 1])
axes[-1].set_xlabel(r"Position $x/L$")
#axes[-1].set_ylabel(r"$|\Psi(x,t)|^2$")
fig.tight_layout(h_pad=.01)

plt.show()
