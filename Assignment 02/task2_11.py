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
N = 101
x, eigvals, psi_n = eval(N)
nSteps = 1000
dt = 0.01/nSteps
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

plt.figure(figsize=[3.5, 2])
plt.imshow(np.real(psi), extent=[0, nSteps*dt/1e-2, x[0], x[-1]], aspect="auto")
plt.title(r"Re $\Psi(x,t)$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$ [$10^{-2}$]")
plt.ylabel(r"Position $x/L$")
plt.colorbar()
plt.tight_layout(pad=0)
plt.savefig("report/figs/box_deltaf_real.pdf", bbox_inches="tight", pad=0)

plt.figure(figsize=[3.5, 2])
plt.imshow(np.imag(psi), extent=[0, nSteps*dt/1e-2, x[0], x[-1]], aspect="auto")
plt.title(r"Im $\Psi(x,t)$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$ [$10^{-2}$]")
plt.ylabel(r"Position $x/L$")
plt.colorbar()
plt.tight_layout(pad=0)
plt.savefig("report/figs/box_deltaf_imag.pdf", bbox_inches="tight", pad=0)

plt.figure(figsize=[3.5, 2])
plt.imshow(np.real(psi.conj()*psi), extent=[0, nSteps*dt/1e-2, x[0], x[-1]], aspect="auto")
plt.title(r"$|\Psi(x,t)|^2$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$ [$10^{-2}$]")
plt.ylabel(r"Position $x/L$")
plt.colorbar()
plt.tight_layout(pad=0)
plt.savefig("report/figs/box_deltaf_prob.pdf", bbox_inches="tight", pad=0)

fig, ax = plt.subplots()
ax.plot(x, psi[:, 0])
ax.plot(x, psi[:, 1])
ax.plot(x, psi[:, 10])
ax.plot(x, psi[:, 20])
ax.plot(x, psi[:, 50])
ax.plot()

plt.show()
