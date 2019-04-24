import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos


def eval(N=201):
    L = 1
    dx = L/(N-2)
    x = np.linspace(0, L, N-1)
    x = x[1:]  # remove endpoint

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N])

    dx2 = dx*dx
    for i in range(N):
        A[i, i] = 2/dx2
        if i > 0:  # skip first row
            A[i, i-1] = -1/dx2
        if i < N-1:  # skip last row
            A[i, i+1] = -1/dx2

    # solve eigenvalue problem
    eigvals, eigvecs = np.linalg.eigh(A)

    # normalize
    for i in range(eigvecs.shape[1]):
        eigvecs[:, i] /= sqrt(sum(eigvecs[:, i]*eigvecs[:, i])*dx)

    # fix sign
    for i in range(eigvecs.shape[1]):
        eigvecs[:, i] = np.abs(eigvecs[:, i])

    return x, eigvals, eigvecs


# plot eigenfunctions
N = 7
n = 0

x, eigvals, eigvecs = eval(N)
psi_n = eigvecs[:, n]

# exact solution
x2 = np.linspace(0, 1, 10000)
exact = sqrt(2)*np.sin(pi*x2*(n + 1))

fig, ax = plt.subplots()
ax.plot(x2, exact, label=r"$\psi_%d$ exact" % n)
ax.plot(x, psi_n, "-o", label=r"$\psi_%d$ num" % n)
ax.legend()
ax.set_ylabel(r"$\Psi(x')$")
ax.set_xlabel(r"$x'$")

plt.show()
