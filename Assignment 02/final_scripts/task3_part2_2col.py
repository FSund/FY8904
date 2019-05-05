import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, exp
from scipy.optimize import minimize, root_scalar, root

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


x, lambda_n, psi_n = eval(1001)

fig2, ax2 = plt.subplots(figsize=[3.45, 1.7])
lambda_n = lambda_n[lambda_n <= V0]
print(lambda_n)

x = np.linspace(0, V0, 1000000)
y = f(x)
ax2.plot(x, y/1e7)
colors = ["C1", "C2", "C3", "C4", "C5", "C6"]
ax2.set_xlabel(r"$\nu_0$")
ax2.set_ylabel(r"$f(\nu_0)$")
ax2.set_xlim([-10, 850])
ax2.set_ylim([0, 3.615])

roots = []
for idx, l in enumerate(lambda_n):
    if idx % 2 == 0:
        l += 1

    res = root(threeFour, x0=l)
    roots.append(res.x[0])

for i in range(len(roots)):
    ax2.axvline(roots[i], label="n = %d" % (i + 1), color=colors[i])

ax2.legend(fontsize="small", handlelength=1)
ax2.set_ylabel(r"$f(\nu_0)$ [$10^7$]")
fig2.tight_layout(pad=0)
fig2.savefig("report_2col/figs/roots_with_roots.pdf", bbox_inches="tight", pad=0)

line = ""
for i in range(0, len(roots), 2):
    line += r"$\lambda_%d$ & %.7f &" % (i+1, roots[i])
print(line)
line = ""
for i in range(1, len(roots), 2):
    line += r"$\lambda_%d$ & %.7f &" % (i+1, roots[i])
print(line)

plt.show()
