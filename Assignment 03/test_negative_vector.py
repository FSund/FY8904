from math import pi, atan, sqrt
import numpy as np
from simulator import Simulator, IncidentWave, LatticeSite
# from simulator_vectorized import Simulator, IncidentWave, LatticeSite
import matplotlib.pyplot as plt

# surface
a = 3.5  # lattice dimension (dimensionless)
xi0 = 0.3  # surface amplitude (dimensionless)

# wave space
H = 9

sim = Simulator(a, xi0)

# theta = 1.99
# phi = -5
theta = pi
phi = 0

As = []
bs = []
rs = []
for add in [0, pi]:
    theta0 = theta + add
    phi0 = phi + add
    k = IncidentWave(theta0, phi0)

    r = sim.simulate(theta0, phi0, H)
    As.append(np.copy(sim.A))
    bs.append(np.copy(sim.b))
    rs.append(r)

    # k = IncidentWave(theta0, phi0)
    Hs = range(-H, H+1)  # +1 to include endpoint
    N = len(sim.h)
    extent = [Hs[0], Hs[-1]+1, Hs[-1]+1, Hs[0]]

    if True:
        # make image
        n = int(round(sqrt(N)))
        image = np.zeros([n, n])
        for idx in range(N):
            i = idx // n
            j = idx % n
            image[i, j] += np.abs(sim.r[idx])

        plt.figure()
        plt.imshow(image, extent=extent)
        plt.colorbar()

    if True:
        # illustrate points included in conservation calculation
        image = np.zeros([n, n])
        for idx in range(N):
            i = idx//n
            j = idx % n
            h1 = Hs[i]
            h2 = Hs[j]

            G = LatticeSite(h1, h2)
            K = k + G
            if K.magnitude < 2*pi:
                image[i, j] = 1

        plt.figure()
        plt.imshow(image, extent=extent)
        plt.colorbar()

    if False:
        # show matrix A
        plt.figure()
        plt.imshow(np.real(sim.A))

    print()


if False:
    diff = As[0] - As[1]

    plt.figure()
    plt.title("Relative difference A (imag)")
    plt.imshow(np.imag(diff/As[0]), aspect="auto", extent=[0, N+1, 0, N+1])
    plt.colorbar()

    plt.figure()
    plt.title("Relative difference A (real)")
    plt.imshow(np.real(diff/As[0]), aspect="auto", extent=[0, N+1, 0, N+1])
    plt.colorbar()

    plt.figure()
    plt.title("Relative difference A (abs)")
    plt.imshow(np.abs(diff/As[0]), aspect="auto", extent=[0, N+1, 0, N+1])
    plt.colorbar()

    plt.figure()
    plt.title("Relative difference A (abs)")
    plt.plot(np.abs(diff/As[0]).flatten())

    plt.figure()
    plt.title("Relative difference b")
    diff = bs[0] - bs[1]
    plt.plot(np.abs(diff/bs[0]))


plt.figure()
plt.title("abs(A)")
plt.plot(np.abs(As[0].flatten()), label="A0")
plt.plot(np.abs(As[1].flatten()), label="A1")
plt.legend()

plt.figure()
plt.title("abs(r)")
plt.plot(np.abs(rs[0]), label="r0")
plt.plot(np.abs(rs[1]), label="r1")
plt.legend()


diff = As[0] - As[1]
print("A mean abs diff = {}".format(np.mean(np.abs(diff/As[0]))))

diff = rs[0] - rs[1]
print("r mean abs diff = {}".format(np.mean(np.abs(diff/rs[0]))))


plt.show()
