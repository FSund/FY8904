from math import pi
import numpy as np
from simulator import Simulator
from joblib import Parallel, delayed
import time
from os import path
import os

# surface
a = 3.5  # lattice dimension (dimensionless)
H = 9
phi0 = 0

n_its = 1000
theta = np.linspace(0, pi/2, n_its, endpoint=False)

for xi0 in [0.3, 0.5, 0.7]:
# for xi0 in [0.5]:  # for fig 4
    def simulate(a, xi0, theta0, phi0, H, idx):
        sim = Simulator(a, xi0, dirichlet=False)
        r = sim.simulate(theta0, phi0, H)
        U = sim.conservation()
        R = sim.reflectivity()

        return [U, R, r]

    def compute_parallel(n_jobs=4):
        t0 = time.perf_counter()
        # parallel computation
        results = Parallel(n_jobs=n_jobs)(delayed(simulate)(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta))
        Us = np.zeros(len(results))
        Rs = np.zeros(len(results))
        N = (2*H+1)**2
        r = np.zeros([len(results), N], dtype=np.complex_)
        for idx, res in enumerate(results):
            Us[idx] = res[0]
            Rs[idx] = res[1]
            r[idx, :] = res[2]
        print("Time elapsed = {} seconds".format(time.perf_counter() - t0))

        return Us, Rs, r

    Us, Rs, r = compute_parallel()

    folder = path.join("fig1_data", "N{}_H{}_a{:.1f}".format(n_its, H, a))
    folder = path.join(folder, "xi0_{:.1f}".format(xi0))
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(path.join(folder, "Rs.npy"), Rs)
    np.save(path.join(folder, "Us.npy"), Us)
    np.save(path.join(folder, "theta.npy"), theta)
    np.save(path.join(folder, "r.npy"), r)
