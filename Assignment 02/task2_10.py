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
N = 201
x, eigvals, psi_n = eval(N)
nSteps = 2000
dt = 1/nSteps
n = len(eigvals)

# set up initial condition (skip solving 2.14 since we know the result)
alpha = np.zeros([n])
alpha[0] = 1  # Psi(x, 0) = psi_1
# alpha[0:2] = 0.5  # Psi(x, 0) = psi_1 + psi_2

# print(alpha)
psi = np.zeros([N, nSteps], dtype=np.complex_)
for i in range(nSteps):  # loop over time steps
    t = dt*i

    # completely manual
    # for ix in range(N):  # loop over positions
    #     for ni in range(n):  # loop over eigenvalues
    #         psi[ix, i] += alpha[ni]*np.exp(-1j*eigvals[ni]*t)*psi_n[ix, ni]

    # one loop
    # for ix in range(N):  # loop over positions
    #     psi[ix, i] = np.sum(alpha*np.exp(-1j*eigvals*t)*psi_n[ix, :])

    # one line
    psi[:, i] = np.sum(alpha*np.exp(-1j*eigvals*t)*psi_n, axis=1)

plt.figure(figsize=[3.5, 2.5])
plt.imshow(np.real(psi), extent=[0, nSteps*dt, x[0], x[-1]], aspect="auto")
plt.title(r"Re $\Psi(x,t)$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$")
plt.ylabel(r"Position $x/L$")
plt.colorbar()
plt.tight_layout()
plt.savefig("report/figs/box_psi0_real.pdf", bbox_inches="tight")

plt.figure(figsize=[3.5, 2.5])
plt.imshow(np.imag(psi), extent=[0, nSteps*dt, x[0], x[-1]], aspect="auto", zorder=0)
plt.title(r"Im $\Psi(x,t)$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$")
plt.ylabel(r"Position $x/L$")
plt.colorbar()
plt.tight_layout()
plt.gca().set_rasterization_zorder(0.5)
plt.savefig("report/figs/box_psi0_imag.pdf", bbox_inches="tight")

plt.figure(figsize=[3.5, 2.5])
plt.imshow(np.real(psi.conj()*psi), extent=[0, nSteps*dt, x[0], x[-1]], aspect="auto")
plt.title(r"$|\Psi(x,t)|^2$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$")
plt.ylabel(r"Position $x/L$")
plt.colorbar()
plt.tight_layout()
plt.savefig("report/figs/box_psi0_prob.pdf", bbox_inches="tight")

plt.show()
