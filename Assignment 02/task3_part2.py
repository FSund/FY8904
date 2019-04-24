import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, exp
from scipy.optimize import minimize, root_scalar, root

# set up plot look
mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
# use r"$x^3_i$" to produce math using cm font
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

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

fig1, ax1 = plt.subplots(figsize=[3.5, 2.5])
fig2, ax2 = plt.subplots(figsize=[3.5, 2])
lambda_n = lambda_n[lambda_n <= V0]
print(lambda_n)

x = np.linspace(0, V0, 1000000)
y = f(x)
ax1.plot(x, y)
ax2.plot(x, y/1e7)
# ax.plot(lambda_n, np.zeros(len(lambda_n)), "o")
colors = ["C1", "C2", "C3", "C4", "C5", "C6"]
for i in range(len(lambda_n)):
    ax1.axvline(lambda_n[i], label="n = %d" % (i + 1), color=colors[i])
# ax.axhline(0, color="k")
for ax in [ax1, ax2]:
    ax.set_xlabel(r"$\nu_0$")
    ax.set_ylabel(r"$f(\nu_0)$")
    ax.set_xlim([-10, 850])
ax1.legend(fontsize="small", handlelength=1)
fig1.tight_layout()
fig1.savefig("report/figs/roots_with_eigenvalues.pdf", bbox_inches="tight")

roots = []
for idx, l in enumerate(lambda_n):
    if idx % 2 == 0:
        l += 1
    # res = minimize(threeFour, l)
    # roots.append(res.x[0])

    # res = root_scalar(threeFour, x0=l)
    # roots.append(res.root)

    res = root(threeFour, x0=l)
    roots.append(res.x[0])
ax1.plot(roots, np.zeros(len(roots)), "x")

for i in range(len(roots)):
    ax2.axvline(roots[i], label="n = %d" % (i + 1), color=colors[i])

ax2.legend(fontsize="small", handlelength=1)
ax.set_ylabel(r"$f(\nu_0)$ [$10^7$]")
fig2.tight_layout(pad=0)
fig2.savefig("report/figs/roots_with_roots.pdf", bbox_inches="tight", pad=0)

# for idx, root in enumerate(roots):
#     print(r"$\lambda_%d$ = %.7f" % (idx+1, root))

line = ""
for i in range(0, len(roots), 2):
    line += r"$\lambda_%d$ & %.7f &" % (i+1, roots[i])
print(line)
line = ""
for i in range(1, len(roots), 2):
    line += r"$\lambda_%d$ & %.7f &" % (i+1, roots[i])
print(line)

plt.show()
