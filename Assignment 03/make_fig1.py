from math import pi, atan, sqrt
import numpy as np
from simulator import Simulator, IncidentWave, LatticeSite
from simulator_vectorized import Simulator as SimulatorVec
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

# surface
a = 3.5  # lattice dimension (dimensionless)
xi0 = 0.3  # surface amplitude (dimensionless)
H = 9
sim = Simulator(a, xi0)
phi0 = 0

n_its = 500
theta = np.linspace(0, pi/2*0.99, n_its)

if False:
    Rs = np.zeros([n_its])
    for idx, theta0 in enumerate(theta):
        print(idx)
        r, U, R = sim.simulate(theta0, phi0, H)

        Rs[idx] = R
else:
    # parallell version
    # Rs = [sim.simulate(theta0, phi0, H)[2] for theta0 in theta]

    def simulate(a, xi0, theta0, phi0, H, idx):
        sim = Simulator(a, xi0)
        # sim = SimulatorVec(a, xi0)
        r, U, R = sim.simulate(theta0, phi0, H)
        print(idx)
        return R

    t0 = time.perf_counter()
    Rs = Parallel(n_jobs=6)(delayed(simulate)(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta))
    print("Time elapsed = {} seconds".format(time.perf_counter() - t0))

fig, ax = plt.subplots()
ax.plot(theta/(2*pi)*360, Rs)
ax.set_yscale("log")
ax.set_ylim([3e-4, 1.1])

fig.savefig("output.png", dpi=300)

plt.show()

np.save("Rs.npy", Rs)
np.save("theta.npy", theta)
