import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos


def potential(x):
    if x <= 0 or x >= 1:
        return 1e300
    else:
        return 0


def eval(N):
    x = np.linspace(-2, 2, N)  # use a large domain
    dx = x[1] - x[0]
    dx2 = dx*dx
    A = np.zeros([N, N])

    for i in range(N):
        A[i, i] = 2/dx2 + potential(x[i])
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
N = 100
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
ax.set_xlim([-0.25, 1.25])


## task 2.6
# calculate error in eigenfunction as function of discretization step

Ns = [13, 51, 101]
# m = [0, 10, 50]
m = [0, 3, 10]
# m = [0, 10, 50]
error = np.zeros([len(Ns), len(m)])
x2 = np.linspace(0, 1, 10000)
for idx, N in enumerate(Ns):
    x, eigvals, eigvecs = eval(N)

    fig, ax = plt.subplots()
    for j, n in enumerate(m):
        eigvec = eigvecs[:, n]
        eig_squared = eigvec*eigvec

        if True:
            exact = sqrt(2)*np.sin(pi*x*(n + 1))
            error[idx, j] = np.sum(np.abs(eigvec - exact))/len(eigvec)

            exact = sqrt(2)*np.sin(pi*x2*(n + 1))
            l, = ax.plot(x2, exact)
            ax.plot(x, eigvec, "-o", color=l.get_color())

        else:
            # error using difference between integrals
            exact_area = 1  # normalized solution should integrate to 1
            area = np.trapz(eig_squared, x)
            error[idx, j] = area - exact_area

            exact = sqrt(2)*np.sin(pi*x2*(n + 1))
            l, = ax.plot(x2, exact*exact)
            ax.plot(x, eig_squared, "-o", color=l.get_color())

# plot error
fig, ax = plt.subplots()
for j, n in enumerate(m):
    ax.plot(Ns, np.abs(error[:, j]), "-o", label="n = %d" % (n+1))
ax.legend()


plt.show()
