import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from math import sqrt, pi, sin, cos


# set up plot look
mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'  # use r"$x^3_i$" to produce math using cm font
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})
plt.close("all")


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
    for i in range(eigvecs.shape[1]):
        if eigvecs[1, i] < 0:
            eigvecs[:, i] *= -1

    return x, eigvals, eigvecs


## plot eigenvalues and vectors
x, eigvals, eigvecs = eval(1001)

fig, ax = plt.subplots(figsize=[3.5, 2])
for n in range(3):
    ax.plot(x, eigvecs[:, n], label="n = %d" % (n + 1))
ax.legend(fontsize="small", handlelength=1, loc="upper center", borderaxespad=0.2)
ax.set_ylabel(r"$\psi_n(x')$")
ax.set_xlabel(r"$x/L$")
ax.set_xlim([0, 1])
ax.axvspan(1/3, 2/3, color=(0, 0, 0, 0.2))
fig.tight_layout()
fig.savefig("report/figs/double_eigenfunctions.pdf")

for n in range(6):
    print("lambda_%d = %.16f" % (n, eigvals[n]))

# fig, ax = plt.subplots()
# for n in range(6):
#     y = np.abs(eigvecs[:, n])**2
#     ax.plot(x, y, label="n = %d" % (n+1))
# ax.legend()
# ax.set_ylabel(r"$|\psi_n(x')|^2$")
# ax.set_xlabel(r"$x/L$")

fig, ax = plt.subplots(figsize=[3.5, 2])
n = np.array(list(range(len(eigvals)))) + 1
ax.plot(n, eigvals)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$E_n/(2mL^2/\hbar^2)$")
ax.set_xlim([1, 8])
ax.set_ylim([0, 1200])
ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.axhline(V0, color="k")
fig.tight_layout()
fig.savefig("report/figs/double_eigenvalues.pdf")

## solve the time-dependent Schrödinger equation
N = 1001
x, lambda_n, psi_n = eval(N)
nSteps = 200  # time steps
# T = 1
T = pi/(lambda_n[1] - lambda_n[0])*2
print("T = %f" % T)
dt = T/nSteps
n = len(lambda_n)

# set up initial condition
alpha = np.zeros([n])
# psi0 = sqrt(2)*np.sin(pi*x)  # initial condition
# alpha[0] = 1  # Psi(x, 0) = psi_1
# alpha[0:2] = 0.5  # Psi(x, 0) = psi_1 + psi_2
# alpha[5] = 1

# if False:
if True:
    # do integration in eq. (2.14)
    psi0 = 1/sqrt(2)*(psi_n[:, 0] + psi_n[:, 1])
    for i in range(n):  # loop over eigenvalues
        alpha[i] = np.trapz(psi_n[:, i]*psi0, x)
else:
    # use exact alpha
    alpha[0:2] = 1/sqrt(2)

# breakpoint()

# find time evolution
psi = np.zeros([N, nSteps], dtype=np.complex_)
for i in range(nSteps):  # loop over time steps
    if i % 100 == 0:
        print(i)
    t = dt*i
    # the only variable in the equation below is time t, so could optimize
    # this part some more by calculating the other factors outside the loop
    psi[:, i] = np.sum(alpha*np.exp(-1j*eigvals*t)*psi_n, axis=1)

plt.figure()
plt.imshow(np.real(psi), extent=[0, T, x[0], x[-1]], aspect="auto")
plt.gca().axvline(pi/(lambda_n[1] - lambda_n[0]), color="k")
plt.title(r"Re $\Psi(x,t)$")
# plt.xlabel(r"Time $t'$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$ [$10^{-2}$]")
plt.ylabel(r"Position $x/L$")
plt.colorbar()

plt.figure()
plt.imshow(np.imag(psi), extent=[0, T, x[0], x[-1]], aspect="auto")
plt.gca().axvline(pi/(lambda_n[1] - lambda_n[0]), color="k")
plt.title(r"Im $\Psi(x,t)$")
# plt.xlabel(r"Time $t'$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$ [$10^{-2}$]")
plt.ylabel(r"Position $x/L$")
plt.colorbar()

psi2 = np.real(psi.conj()*psi)
f = 1e3
plt.figure(figsize=[3.5, 2.5])
plt.imshow(psi2, extent=[0, T/f, x[0], x[-1]], aspect="auto")
plt.gca().axvline(pi/(lambda_n[1] - lambda_n[0])/f, color="r")
plt.title(r"$|\Psi(x,t)|^2$")
# plt.xlabel(r"Time $t'$")
plt.xlabel(r"Time $t/(2mL^2/\hbar)$ [$10^{3}$]")
plt.ylabel(r"Position $x/L$")
plt.colorbar()
plt.tight_layout()
plt.savefig("report/figs/double_twopsi_prob.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=[3.5, 2.5])
tunneltime = pi/(lambda_n[1] - lambda_n[0])
for i in range(0, int(round(nSteps/2)) + 1, 25):
    ax.plot(x, psi2[:, i], label=r"$t' = %.2fT$" % ((i*dt)/tunneltime))
ax.set_title(r"$|\Psi(x,t)|^2$")
ax.set_xlabel(r"Position $x/L$")
ax.set_ylabel(r"$|\Psi(x,t)|^2$")
ax.set_xlim([0, 1])
ax.legend(fontsize="small")
fig.tight_layout()
plt.savefig("report/figs/double_twopsi_prob_1d.pdf", bbox_inches="tight")
print(list(range(0, int(round(nSteps/2)) + 1, 10))[-1])

# animate time evolution
# fig, ax = plt.subplots()
# ax.set_title(r"$|\Psi(x,t)|^2$")
# ax.set_xlabel(r"Position $x/L$")
# ax.set_ylabel(r"$|\Psi(x,t)|^2$")
# l, = ax.plot(x, psi2[:, 0])
# ax.axvspan(L/3, 2*L/3, facecolor='k', alpha=0.3)
# plt.show()
# for i in range(1, nSteps):
#     l.set_ydata(psi2[:, i])
#     fig.canvas.draw()
#     plt.pause(0.01)


plt.show()
