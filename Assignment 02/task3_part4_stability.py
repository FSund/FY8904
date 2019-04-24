import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, exp

# set up plot look
mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
if False:
    mpl.rcParams['mathtext.fontset'] = 'cm'
    rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})
else:
    # use pdflatex
    mpl.rcParams['text.usetex'] = 'true'
    rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

V0 = 1e3  # potential barrier in the middle
L = 1


def _potential_well(x, L):
    if x < 0 or x > 1:
        return 1e300  # side walls
    else:
        return 0


def _potential_barrier(x, L):
    if x < 0 or x > 1:
        return 1e300  # side walls
    else:
        if x > L/3 and x < 2*L/3:
            return V0  # middle barrier
        else:
            return 0


def potential(x, L):
    # return _potential_well(x, L)
    return _potential_barrier(x, L)


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


def advance(psi, x, dt):
    """
    Solves equation (3.6)
    """
    N = len(psi)
    dx = x[1] - x[0]  # assume uniform
    dx2 = dx*dx

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N], dtype=np.complex_)

    # fill in A
    for i in range(N):
        V = potential(x[i + 1], L)

        # diagonal
        A[i, i] = 1 - (1j)*dt*V - 2*(1j)*dt/dx2

        # lower diagonal
        if i > 0:  # skip first row
            A[i, i-1] = (1j)*dt/dx2

        # upper diagonal
        if i < N-1:  # skip last row
            A[i, i+1] = (1j)*dt/dx2

    psi_out = A@psi[1:-1]  # @ is matrix-vector multiplication

    # boundary conditions
    psi_out = np.append(0, psi_out)
    psi_out = np.append(psi_out, 0)

    # breakpoint()

    return psi_out


N = 51

# set up initial condition
x, lambda_n, psi_n = eval(N)
psi0 = psi_n[:, 0]  # initial condition == psi_0'

nSteps = 500
dx = x[1] - x[0]
fig, ax = plt.subplots(figsize=[3.5, 2])
# for CFL in [0.01, 0.1, 0.5, 0.8, 1.0, 1.2, 1.5]:
for CFL in [0.6, 0.8, 1.0, 1.2, 1.4]:
    dt = dx*dx*CFL
    print("dt = %f" % dt)
    print("CFL= %f" % (dt/dx/dx))

    psi = np.zeros([N, nSteps], dtype=np.complex_)
    psi[:, 0] = psi0
    for i in range(1, nSteps):  # loop over time steps
        if i > 0 and i % 100 == 0:
            print(i)

        psi[:, i] = advance(psi[:, i - 1], x, dt)

    # psi2 = np.abs(psi)**2
    psi2 = np.real(psi.conj()*psi)

    integral = np.zeros(nSteps)
    for i in range(0, nSteps):  # loop over time steps
        integral[i] = np.trapz(psi2[:, i], x)

    t = np.linspace(0, nSteps*dt, nSteps)
    ax.plot(t, integral, label=r"CFL = %.2f" % CFL)

ax.set_ylim([0, 10])
ax.legend(fontsize="x-small")
plt.ylabel(r"$\int |\Psi(x,t)|^2 dx'$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$")
# ax.grid()
ax.set_xlim([0, 0.0115])
ax.set_ylim([1, 1.15])

fig.tight_layout()
fig.savefig("report/figs/euler_stability.pdf", bbox_inches="tight")

plt.show()
