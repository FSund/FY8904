import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos


mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})
# rc('text', usetex=True)  # render using LaTeX
# rc('mathtext.fontset' : dejavusans
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


# plot eigenfunctions
N = 11
x2 = np.linspace(0, 1, 10000)
n = 2
x, eigvals, eigvecs = eval(N)
psi_n = eigvecs[:, n]
fig, ax = plt.subplots()
exact = sqrt(2)*np.sin(pi*x2*(n + 1))
l, = ax.plot(x2, exact, label=r"$\psi_%d$ exact" % n)
ax.plot(x, psi_n, "-o", color=l.get_color(), label=r"$\psi_%d$ num" % n)
ax.legend()
ax.set_ylabel(r"$\psi_n(x')$")
ax.set_title(r"$\psi_n(x')$")
ax.set_xlabel(r"$x/L$")
ax.set_xlim([0, 1])

fig, ax = plt.subplots()
l, = ax.plot(x2, exact*exact, label=r"$\psi_%d$ exact" % n)
ax.plot(x, psi_n*psi_n, "-o", color=l.get_color(), label=r"$\psi_%d$ num" % n)
ax.set_ylabel(r"$|\psi_n(x')|^2$")
ax.set_title(r"$|\psi_n(x')|^2$")
ax.set_xlabel(r"$x/L$")
ax.set_xlim([0, 1])
ax.legend()


## task 2.6
# calculate error in eigenfunction as function of discretization step
# Ns = [101, 501, 1001]
# Ns = [101, 501, 1001]
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
