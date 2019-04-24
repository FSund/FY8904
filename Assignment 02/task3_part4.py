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


## use eq. (3.6) to do time evolution
N = 51
x, lambda_n, psi_n = eval(N)
n = len(lambda_n)

# find time evolution
nSteps = 200
dx = x[1] - x[0]
dt = dx*dx*1.1
# dt = 0.0000005
print("dt = %f" % dt)
print("CFL= %f" % (dt/dx/dx))

if True:
    psi0 = psi_n[:, 0]  # initial condition == psi_0
else:
    psi0 = 1/sqrt(2)*(psi_n[:, 0] + psi_n[:, 1])  # initial condition

psi = np.zeros([N, nSteps], dtype=np.complex_)
psi[:, 0] = psi0
for i in range(1, nSteps):  # loop over time steps
    if i > 0 and i % 100 == 0:
        print(i)

    psi[:, i] = advance(psi[:, i - 1], x, dt)

psi2 = np.abs(psi)**2

integral = np.zeros(nSteps)
for i in range(1, nSteps):  # loop over time steps
    integral[i] = np.trapz(x, psi[:, i])

fig, ax = plt.subplots()
x = np.linspace(0, nSteps*dt, nSteps)
ax.plot(x, integral)

# find cut off for plots (where the values get too large to make any meaning)
i1 = int(np.argwhere(np.max(psi2, axis=0) > 100)[0][0])

plt.figure()
plt.imshow(psi2, extent=[0, nSteps*dt, x[0], x[-1]], vmax=np.max(psi2[:, 0])*1.1, aspect="auto")
plt.title(r"$|\Psi(x,t)|^2$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

plt.figure()
vmin = np.min(np.real(psi[:, 0]))*0.9
vmax = np.max(np.real(psi[:, 0]))*1.1
plt.imshow(np.real(psi), vmin=vmin, vmax=vmax, extent=[0, nSteps*dt, x[0], x[-1]], aspect="auto")
plt.title(r"$Re(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

plt.figure()
vmin = np.min(np.imag(psi[:, 0]))*0.9
vmax = np.max(np.imag(psi[:, 0]))*1.1
vmin = -0.0001
vmax = 0.0001
plt.imshow(np.imag(psi), vmin=vmin, vmax=vmax, extent=[0, nSteps*dt, x[0], x[-1]], aspect="auto")
plt.title(r"$Im(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

fig, ax = plt.subplots()
ax.plot(psi2[:, 0], label="t = 0")
ax.plot(psi2[:, 1], label="t = %f" % (dt*1))
ax.plot(psi2[:, 2], label="t = %f" % (dt*2))
ax.plot(psi2[:, 3], label="t = %f" % (dt*3))
ax.plot(psi2[:, -1], label="t = %f" % (dt*nSteps))
ax.legend()
ax.set_ylim([0, np.max(psi2[:, 0])*1.1])

fig, ax = plt.subplots()
ax.plot(np.real(psi[:, 0]), label="Re t = 0")
ax.plot(np.imag(psi[:, 0]), label="Im t = 0")
ax.plot(np.abs(psi[:, 0]), label="Abs t = 0")
ax.plot(np.real(psi[:, 1]), label="Re t = %f" %(dt*1))
ax.plot(np.imag(psi[:, 1]), label="Im t = %f" %(dt*1))
ax.plot(np.abs(psi[:, 1]), label="Abs t = %f" %(dt*1))
ax.legend()
ax.set_ylim([0, np.max(psi2[:, 0])*1.1])

fig, ax = plt.subplots()
for i in range(0, i1, round(i1/10)):
    ax.plot(psi2[:, i], label="t = %f" % (dt*i))
ax.legend()
ax.set_ylim([0, np.max(psi2[:, 0])*1.1])
plt.title(r"$|\Psi(x,t)|^2$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

fig, ax = plt.subplots()
for i in range(0, i1, round(i1/10)):
    ax.plot(np.real(psi[:, i]), label="t = %f" % (dt*i))
ax.legend()
# ax.set_ylim([0, np.max(psi2[:, 0])*1.1])
plt.title(r"$Re(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

fig, ax = plt.subplots()
for i in range(0, i1, round(i1/10)):
    ax.plot(np.imag(psi[:, i]), label="t = %f" % (dt*i))
ax.legend()
# ax.set_ylim([0, np.max(psi2[:, 0])*1.1])
plt.title(r"$Im(\Psi(x,t))$")
plt.xlabel(r"Time $t'$")
plt.ylabel(r"Position $x/L$")

plt.show()
