from math import pi
import numpy as np
from simulator import Simulator
from joblib import Parallel, delayed
import time
from os import path
import os


Simulator.dirichlet = True

# surface
a = 3.5  # lattice dimension (dimensionless)
H = 9
phi0 = 0

n_its = 10
theta = np.linspace(0, pi/2, n_its, endpoint=False)

for xi0 in [0.3, 0.5, 0.7]:
    def simulate(a, xi0, theta0, phi0, H, idx):
        sim = Simulator(a, xi0)
        r, U, R = sim.simulate(theta0, phi0, H)
        print(idx)
        return [U, R]

    def compute_parallel(n_jobs=6):
        t0 = time.perf_counter()
        # parallel computation
        results = Parallel(n_jobs=n_jobs)(delayed(simulate)(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta))
        results = np.array(results, dtype=np.complex_)
        Us = results[:, 0]
        Rs = results[:, 1]
        print("Time elapsed = {} seconds".format(time.perf_counter() - t0))
        return Us, Rs

    Us, Rs = compute_parallel()

    folder = path.join("fig3_N{}".format(n_its), "xi0_{}".format(xi0))
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(path.join(folder, "Rs.npy"), Rs)
    np.save(path.join(folder, "Us.npy"), Us)
    np.save(path.join(folder, "theta.npy"), theta)
