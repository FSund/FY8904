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

V0 = 1000  # potential barrier in the middle
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
    A = np.zeros([N, N], dtype=np.complex_)

    # fill in A
    for i in range(N):
        V = potential(x[i + 1], L)

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
    Solves equation (3.8)
    """
    N = len(psi)
    dx = x[1] - x[0]  # assume uniform grid
    dx2 = dx*dx

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N], dtype=np.complex_)
    B = np.zeros([N, N], dtype=np.complex_)

    # fill in A
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

    b = B @ psi[1:-1]  # @ is matrix-vector multiplication, which returns a vec

    psi_out = np.linalg.solve(A, b)

    # boundary conditions
    psi_out = np.append(0, psi_out)
    psi_out = np.append(psi_out, 0)

    return psi_out


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


## use Crank-Nicholsom scheme from eq. (3.8) to do time evolution

# set up initial condition
N = 51
x, lambda_n, psi_n = eval(N)
dx = x[1] - x[0]
dt = dx*dx
T = pi/(lambda_n[1] - lambda_n[0])  # total time -- time needed for tunnelling
# nSteps = 10000
nSteps = int(round(T/dt))
print("T = %f" % T)
# dt = T/nSteps
print("dt = %f" % dt)
print("nSteps = %d" % nSteps)
psi0 = 1/sqrt(2)*(psi_n[:, 0] + psi_n[:, 1])  # initial condition

# time evolution
solver = CrankNicholsonSolver(x, dt)
psi = np.zeros([N, nSteps], dtype=np.complex_)
psi[:, 0] = psi0
for i in range(1, nSteps):  # loop over time steps
    if i > 0 and i % 1000 == 0:
        print(i)

    # psi[:, i] = advance(psi[:, i - 1], x, dt)
    psi[:, i] = solver.advance(psi[:, i - 1])

# psi2 = np.abs(psi)**2
psi2 = np.real(psi.conj()*psi)  # equivalent to the above, as far as I can see
extent = [0, nSteps*dt, x[0], x[-1]]

plt.figure()
plt.imshow(psi2, extent=extent, aspect="auto")
plt.title(r"$|\Psi(x,t)|^2$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

plt.figure()
plt.imshow(np.real(psi), extent=extent, aspect="auto")
plt.title(r"$Re(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

plt.figure()
# use same limits as plot of real
vmax = np.max(np.real(psi))
vmin = np.min(np.real(psi))
plt.imshow(np.imag(psi), extent=extent, aspect="auto", vmax=vmax, vmin=vmin)
plt.title(r"$Im(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

r = np.round(np.linspace(0, nSteps-1, 10)).astype(int)

fig, ax = plt.subplots()
for i in r:
    ax.plot(psi2[:, i], label="t = %f" % (dt*i))
ax.legend()
plt.title(r"$|\Psi(x,t)|^2$")
plt.xlabel(r"Position $x/L$")
plt.ylabel(r"$|\Psi(x,t)|^2$")

fig, ax = plt.subplots()
for i in r:
    ax.plot(np.real(psi[:, i]), label="t = %f" % (dt*i))
ax.legend()
plt.title(r"$Re(\Psi(x,t))$")
plt.xlabel(r"Position $x/L$")
plt.ylabel(r"$Re(\Psi(x,t))$")

fig, ax = plt.subplots()
for i in r:
    ax.plot(np.imag(psi[:, i]), label="t = %f" % (dt*i))
ax.legend()
plt.title(r"$Im(\Psi(x,t))$")
plt.xlabel(r"Position $x/L$")
plt.ylabel(r"$Im(\Psi(x,t))$")

plt.show()
