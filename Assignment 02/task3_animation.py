import numpy as np
from math import sqrt, pi, sin, cos
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mpl.rcParams['mathtext.fontset'] = 'cm'  # use r"$x^3_i$" to produce math using cm font
mpl.rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

L = 1
# V0 = 0  # try to recreate previous results
V0 = 1e3  # potential barrier in the middle


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

    eigvals, eigvecs = np.linalg.eigh(A)

    # add boundary conditions (zeros)
    eigvecs = np.append(np.zeros([1, N]), eigvecs, axis=0)
    eigvecs = np.append(eigvecs, np.zeros([1, N]), axis=0)

    # normalize
    eigvecs /= sqrt(dx)

    # fix sign
    # for i in range(eigvecs.shape[1]):
    #     if eigvecs[1, i] < 0:
    #         eigvecs[:, i] *= -1

    return x, eigvals, eigvecs


## solve the time-dependent Schrödinger equation
N = 1001
x, lambda_n, psi_n = eval(N)
nSteps = 1000  # time steps
# T = 1
T = pi/(lambda_n[1] - lambda_n[0])*20
print("T = %f" % T)
dt = T/nSteps
n = len(lambda_n)

# set up initial condition
alpha = np.zeros([n])
if False:
    # do integration in eq. (2.14)
    psi0 = 1/sqrt(2)*(psi_n[:, 0] + psi_n[:, 1])
    for i in range(n):  # loop over eigenvalues
        alpha[i] = np.trapz(psi_n[:, i]*psi0, x)
else:
    # use exact alpha
    alpha[0:2] = 1/sqrt(2)

# find time evolution
psi = np.zeros([N, nSteps], dtype=np.complex_)
for i in range(nSteps):  # loop over time steps
    if i % 100 == 0:
        print(i)
    t = dt*i
    # the only variable in the equation below is time t, so could optimize
    # this part some more by calculating the other factors outside the loop
    psi[:, i] = np.sum(alpha*np.exp(-1j*lambda_n*t)*psi_n, axis=1)

psi2 = np.abs(psi)**2

## make animation
fig, ax = plt.subplots()
ln, = ax.plot(x, psi2[:, 0])
ax.set_xlim([x[0], x[-1]])
ax.set_title(r"$|\Psi(x,t)|^2$")
ax.set_xlabel(r"Position $x/L$")
ax.set_ylabel(r"$|\Psi(x,t)|^2$")
ax.axvspan(L/3, 2*L/3, facecolor='k', alpha=0.3)
text = ax.text(0.5, 0.95, "t = %f" % 0, transform=ax.transAxes, va="top", ha="center")


def init():
    return ln, text


def update(frame):
    ln.set_ydata(psi2[:, frame])
    text.set_text("t = %f" % (frame*dt))
    return ln, text

anim = FuncAnimation(fig, update, frames=range(nSteps), init_func=init, blit=True)

anim.save("perturbed_well.mp4", fps=25)
# plt.show()
