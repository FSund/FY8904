from math import pi, atan, sqrt
import numpy as np
from simulator import Simulator, IncidentWave, LatticeSite
import matplotlib.pyplot as plt

# surface
a = 3.5  # lattice dimension
xi0 = 0.3  # surface amplitude
# xi0 = 0

# wave space
H = 5  # 10 should be fine for testing

sim = Simulator(a, xi0)

theta = 1.99
phi = -5
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

    if False:
        # show points included in conservation calculation
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

# plt.imshow(np.log(np.abs(As[0] - As[1])), aspect="auto")
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
# print("mean b diff = {}".format(np.mean(np.abs(bs[0] - bs[1]))))

plt.figure()
plt.title("abs(A)")
plt.plot(np.abs(As[0].flatten()))
plt.plot(np.abs(As[1].flatten()))

plt.figure()
plt.title("abs(r)")
plt.plot(np.abs(rs[0]))
plt.plot(np.abs(rs[1]))

# plt.figure()
# A0 = As[0][0, :]
# A1 = As[1][0, :]
# plt.plot(np.abs(A0), label="A0")
# plt.plot(np.abs(A1), '--', label="A1")
# plt.legend()
# plt.figure()
# plt.plot(np.abs(A0 - A1), label="diff")
# print("A[:, 0] diff = {}".format(np.mean(np.abs(A0 - A1))))

# plt.figure()
# A0 = As[0][:, 0]
# A1 = As[1][:, 0]
# plt.plot(np.abs(A0), label="A0")
# plt.plot(np.abs(A1), '--', label="A1")
# plt.legend()
# plt.figure()
# plt.plot(np.abs(A0 - A1), label="diff")
# # print("A[:, 0] diff = {}".format(np.mean(np.abs(A0 - A1))))
# diff = A0 - A1
# print("A[:, 0] diff = {}".format(np.mean(np.abs(diff/A0))))
# print(np.argmax(np.abs(diff/A0)))
# print(np.max(np.abs(diff/A0)))

plt.show()
