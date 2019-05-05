import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos


mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif Roman 2'

plt.close("all")


def normEigen(x, u):
    # normalizes eigenfunctions
    norm = np.trapz(u*u, x)
    # print("norm = %f, dx = %f" % (norm, x[1] - x[0]))
    # print(sqrt(norm))
    return u/sqrt(norm)


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

    eigvals, eigvecs = np.linalg.eig(A)

    # add boundary conditions (zeros)
    eigvecs = np.append(np.zeros([1, N]), eigvecs, axis=0)
    eigvecs = np.append(eigvecs, np.zeros([1, N]), axis=0)

    # sort according to size of eigenvalues
    eigvecs = eigvecs[:, np.argsort(eigvals)]
    eigvals = np.sort(eigvals)

    # normalize
    for i in range(eigvecs.shape[1]):
        eigvecs[:, i] /= sqrt(dx)
        # eigvecs[:, i] = normEigen(x, eigvecs[:, i])

    # fix sign
    for i in range(eigvecs.shape[1]):
        if eigvecs[1, i] < 0:
            eigvecs[:, i] *= -1

    return x, eigvals, eigvecs


x, eigvals, eigvecs = eval(N=201)
print("dx = %f" % (x[1] - x[0]))

## task 2.5
# plot eigenvalues
fig, ax = plt.subplots(figsize=[3.45, 1.7])
n = np.array(list(range(len(eigvals)))) + 1
ax.plot(n, eigvals/1e5, label="Modelled")
lam = pi*pi*n*n
ax.plot(n, lam/1e5, label="Exact")
ax.set_xlim([1, 100])
# ax.set_ylim([0, lam[99]/1e5])
ax.set_ylim([0, 1])
ax.legend(fontsize="small")
ax.set_ylabel(r"$E_n/(2mL^2/\hbar^2)$ [$10^5$]")
ax.set_xlabel(r"$n$")
fig.tight_layout(pad=0)
fig.savefig("report_2col/figs/box_eigenvalues.pdf", bbox_inches="tight", pad=0)


# plot eigenvectors
fig, ax = plt.subplots(figsize=[3.45, 1.7])
for n in range(4):
    y = eigvecs[:, n]
    label = r"$\lambda_n$ = %.2f" % eigvals[n]
    label = r"$n$ = %d" % (n + 1)
    l, = ax.plot(x[::2], y[::2], "o", markersize=1.5, label=label)
    psi = sqrt(2)*np.sin(pi*x*(n+1))
    ax.plot(x, psi, color=l.get_color(), linewidth=1)

ax.legend(loc=[0.05, 0.05], fontsize="small", handlelength=0.5)
ax.set_ylabel(r"$\psi_n(x')$")
ax.set_xlabel(r"$x/L$")
ax.set_xlim([0, 1])
fig.tight_layout(pad=0)
fig.savefig("report_2col/figs/box_eigenvectors.pdf", bbox_inches="tight", pad=0)


## Task 2.6
# see task2_6.py


## expansion in eigenfunctions
x, eigvals, eigvecs = eval(101)
N = len(eigvals)
alpha = np.zeros([N, N])
for n in range(N):
    for m in range(N):
        alpha[m, n] = np.trapz(eigvecs[:, n]*eigvecs[:, m], x)

plt.figure()
plt.imshow(alpha, aspect="equal")
plt.colorbar()
plt.xlabel("m")
plt.ylabel("n")

# plt.figure()
# plt.plot(np.diagonal(alpha))
# plt.plot(np.max(alpha), '--')

# these are all equal to 1???


plt.show()
