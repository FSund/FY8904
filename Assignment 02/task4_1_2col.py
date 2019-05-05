import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, exp

# set up plot look
mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
# use r"$x^3_i$" to produce math using cm font
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

L = 1


def _potential_well(x, L):
    if x < 0 or x > L:
        return 1e300  # side walls
    else:
        return 0


def _potential_barrier(x, L, V0):
    if x < 0 or x > L:
        return 1e300  # side walls
    else:
        if x > L/3 and x < 2*L/3:
            return V0  # middle barrier
        else:
            return 0


def _right_barrier(x, L, V0, Vr):
    if x < 0 or x > L:
        return 1e300  # side walls
    else:
        if x < L/3:
            return 0  # left third
        elif x < 2*L/3:
            return V0  # middle third
        else:
            return Vr  # right third


def potential(x, L, V0, Vr):
    # return _potential_well(x, L)
    # return _potential_barrier(x, L, V0)
    return _right_barrier(x, L, V0, Vr)


def eval(N=201, V0=0, Vr=0):
    dx = L/(N - 1)
    dx2 = dx*dx

    x = np.linspace(0, L, N)

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N], dtype=np.complex_)

    # fill in A
    for i in range(N):
        V = potential(x[i + 1], L, V0, Vr)

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


class CrankNicholsonSolver:
    def __init__(self, x, dt):
        N = len(x)
        dx = x[1] - x[0]  # assume uniform grid
        dx2 = dx*dx

        # don't set up matrix for psi_0 and psi_N, since they are known
        N = N - 2
        A = np.zeros([N, N], dtype=np.complex_)
        B = np.zeros([N, N], dtype=np.complex_)

        # fill in A and B
        for i in range(N):
            V = potential(x[i + 1], L)  # i+1 since we have discarded first point!

            # diagonal
            A[i, i] = 1.0 + (1.0j)/2.0*dt*(V + 2.0/dx2)
            B[i, i] = 1.0 - (1.0j)/2.0*dt*(V + 2.0/dx2)

            # lower diagonal
            if i > 0:  # skip first row
                A[i, i-1] = -(1.0j)/2.0*dt/dx2
                B[i, i-1] = (1.0j)/2.0*dt/dx2

            # upper diagonal
            if i < N-1:  # skip last row
                A[i, i+1] = -(1.0j)/2.0*dt/dx2
                B[i, i+1] = (1.0j)/2.0*dt/dx2

        self.B = B
        self.A = A
        self.b = np.zeros([N], dtype=np.complex_)

    def advance(self, psi):
        self.b = self.B @ psi[1:-1]  # @ is matrix-vector multiplication, which returns a vec

        psi_out = np.zeros([len(psi)], dtype=np.complex_)
        psi_out[1:-1] = np.linalg.solve(self.A, self.b)

        # boundary conditions
        psi_out[0] = 0
        psi_out[-1] = 0

        return psi_out


fig, axes = plt.subplots(2, 1, figsize=[3.45, 4])

## task 4.1
# calculate two lowest eigenvalues as functions of Vr
N = 1001
V0 = 100
nV = 11  # testing
nV = 101  # prod
Vrs = np.linspace(-100, 100, nV)
lambda_r = np.zeros([2, nV])
for idx, Vr in enumerate(Vrs):
    x, lambda_n, psi_n = eval(N, V0, Vr)
    lambda_r[:, idx] = lambda_n[0:2]

ax = axes[0]
ax.plot(Vrs, lambda_r[0, :], label=r"$\lambda_1$")
ax.plot(Vrs, lambda_r[1, :], label=r"$\lambda_2$")
ax.legend(fontsize="small", handlelength=1)
ax.set_xlabel(r"$\nu_r$")
ax.set_ylabel(r"$\lambda_n(\nu_r)$")
ax.set_xlim([Vrs[0], Vrs[-1]])

# # check that for very large Vr, the ground state is localized mostly on the
# # left side of the barrier, while for negative Vr, the ground state is on the
# # other side
# fig, ax = plt.subplots()
# for Vr in [-10, 10]:
#     x, lambda_n, psi_n = eval(N, V0, Vr)
#     psi = psi_n[:, 0]
#     psi2 = np.real(psi.conj()*psi)
#     ax.plot(x, psi2, label=r"$\psi_0$, $\nu_r = %d$" % Vr)
# ax.legend()
# ax.set_ylabel(r"$|\Psi|^2$")
# ax.set_xlabel(r"$x/L$")
# ax.set_xlim([0, 1])

# plot the ground state psi_0 and the excited state psi_1
ax = axes[1]
from cycler import cycler
ax.set_prop_cycle(cycler(color=[u'#1f77b4', u'#ff7f0e', u'#d62728']))  # remove the green color which clashes a bit with the blue
for Vr in [-10, 0, 10]:
    x, lambda_n, psi_n = eval(N, V0, Vr)
    psi_0 = psi_n[:, 0]
    psi_1 = psi_n[:, 1]
    psi0squared = np.real(psi_0.conj()*psi_0)
    psi1squared = np.real(psi_1.conj()*psi_1)
    l, = ax.plot(x, psi0squared,
                 # label=r"$\psi_1$, $\nu_r = %d$" % Vr
                 label=r"$\nu_r = %d$" % Vr
                 )
    ax.plot(x, psi1squared,
            # label=r"$\psi_2$, $\nu_r = %d$" % Vr,
            dashes=[4, 4], color=l.get_color())
ax.legend(fontsize="small", handlelength=1)
ax.set_ylabel(r"$|\Psi|^2$")
ax.set_xlabel(r"$x/L$")
ax.set_xlim([0, 1])

fig.tight_layout(pad=0)
fig.savefig("report_2col/figs/task4.pdf", bbox_inches="tight", pad=0)

# energy difference epsilon_0 between the ground state and the excited state at
# zero detuning, that is, for Vr = 0
x, lambda_n, psi_n = eval(N, V0=0, Vr=0)
eps_0 = lambda_n[1] - lambda_n[0]
print(r"$\varepsilon_0 = %f$" % eps_0)

plt.show()
