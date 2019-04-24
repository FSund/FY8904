import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from math import sqrt, pi, sin, cos, exp

# set up plot look
mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
# use r"$x^3_i$" to produce math using cm font
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

V0 = 500  # potential barrier in the middle
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
    dx = x[1] - x[0]  # assume uniform
    dx2 = dx*dx

    # don't set up matrix for psi_0 and psi_N, since they are known
    N = N - 2
    A = np.zeros([N, N], dtype=np.complex_)
    b = np.zeros([N, N], dtype=np.complex_)

    # fill in A
    for i in range(N):
        V = potential(x[i + 1], L)  # i+1 since we have discarded first point!

        # diagonal
        A[i, i] = 1.0 + (1.0j)/2.0*dt*(V + 2.0/dx2)
        b[i, i] = 1.0 - (1.0j)/2.0*dt*(V + 2.0/dx2)

        # lower diagonal
        if i > 0:  # skip first row
            A[i, i-1] = -(1.0j)/2.0*dt/dx2
            b[i, i-1] = (1.0j)/2.0*dt/dx2

        # upper diagonal
        if i < N-1:  # skip last row
            A[i, i+1] = -(1.0j)/2.0*dt/dx2
            b[i, i+1] = (1.0j)/2.0*dt/dx2

    # b = b @ psi[1:-1]  # @ is matrix-vector multiplication, which returns a vec
    b = np.matmul(b, psi[1:-1])

    psi_out = np.linalg.solve(A, b)

    # boundary conditions
    psi_out = np.append(0, psi_out)
    psi_out = np.append(psi_out, 0)

    # breakpoint()

    return psi_out


## use eq. (3.6) to do time evolution

# set up initial condition
N = 51
x, lambda_n, psi_n = eval(N)
dx = x[1] - x[0]
n = len(lambda_n)

T = pi/(lambda_n[1] - lambda_n[0])
dt = dx*dx  # CFL = 1
nSteps = int(round(T/dt))
psi0 = 1/sqrt(2)*(psi_n[:, 0] + psi_n[:, 1])

print("nSteps = %d" % nSteps)
print("T = %f" % (dt*nSteps))
print("dt = %f" % dt)
print("dx = %f" % dx)

# time evolution
load = True
# load = False

psi = np.zeros([N, nSteps], dtype=np.complex_)
if load:
    psi = np.load("CN_results.npy")
else:
    psi[:, 0] = psi0
    for i in range(1, nSteps):  # loop over time steps
        if i > 0 and i % 100 == 0:
            print(i)

        psi[:, i] = advance(psi[:, i - 1], x, dt)
    np.save("CN_results.npy", psi)

# psi2 = np.abs(psi)**2
psi2 = np.real(psi.conj()*psi)
extent = [0, nSteps*dt, x[0], x[-1]]

fig, ax = plt.subplots(figsize=[3.5, 2.5])
plt.imshow(psi2, extent=extent, aspect="auto")
plt.title(r"$|\Psi(x,t)|^2$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$")
plt.ylabel(r"Position $x/L$")
ax.xaxis.set_major_locator(MultipleLocator(10))
plt.colorbar()
fig.tight_layout()
plt.savefig("report/figs/double_cn_tunnel.pdf", bbox_inches="tight")

plt.figure()
plt.imshow(np.real(psi), extent=extent, aspect="auto")
plt.title(r"$Re(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

plt.figure()
vmax = np.max(np.real(psi))  # use max of real, since we have trouble with the imaginary values
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
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

fig, ax = plt.subplots()
for i in r:
    ax.plot(np.real(psi[:, i]), label="t = %f" % (dt*i))
ax.legend()
plt.title(r"$Re(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

fig, ax = plt.subplots()
for i in r:
    ax.plot(np.imag(psi[:, i]), label="t = %f" % (dt*i))
ax.legend()
plt.title(r"$Im(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

plt.show()
