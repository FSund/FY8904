from math import pi, atan, sqrt
import numpy as np
from simulator import Simulator, IncidentWave, LatticeSite
import matplotlib.pyplot as plt

# surface
a = 3.5  # lattice dimension
# xi0 = 0.3  # surface amplitude
xi0 = 0.1

# incident wave
theta0 = 0  # polar angle of incidence (vary this?)
phi0 = 0  # azimuthal angle of incidence (0 or 45 in the article)

# wave space
H = 12  # 10 should be fine for testing

sim = Simulator(a, xi0)
sim.simulate(theta0, phi0, H)

k = IncidentWave(theta0, phi0)

N = len(sim.h)
phi0s = np.zeros([N])
for i in range(N):
    h1 = sim.h[i][0]
    h2 = sim.h[i][1]
    Gprime = LatticeSite(h1, h2)
    Kprime = k + Gprime

    if Kprime.y == 0:
        phi0s[i] = pi/2
    else:
        phi0s[i] = atan(np.real(Kprime.y/Kprime.x))

fig, ax = plt.subplots()
index = np.argsort(phi0s)
ax.plot(phi0s[index], sim.r[index])

plt.figure()
plt.plot(np.sort(phi0s))

# make image
n = int(round(sqrt(N)))
image = np.zeros([n, n])
for idx in range(N):
    i = idx // n
    j = idx % n
    try:
        image[i, j] += np.abs(sim.r[idx])
    except Exception:
        breakpoint()

plt.figure()
plt.imshow(image)

plt.show()