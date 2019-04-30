from math import pi
import numpy as np
from simulator import Simulator
from joblib import Parallel, delayed
import time
from os import path
import os

# surface
H = 9
phi0 = 0
theta0 = 0

n_its = 100

for a in [0.5, 3.5]:
    for dirichlet in [True, False]:
        def simulate(a, xi0, theta0, phi0, H, idx):
            sim = Simulator(a, xi0, dirichlet)
            sim.simulate(theta0, phi0, H)
            U = sim.conservation()
            R = sim.reflectivity()
            print(idx)
            return [U, R]

        def compute_parallel(n_jobs=4):
            t0 = time.perf_counter()
            # parallel computation
            results = Parallel(n_jobs=n_jobs)(delayed(simulate)(a, xi0, theta0, phi0, H, idx) for idx, xi0 in enumerate(xi))
            results = np.array(results, dtype=np.complex_)
            Us = results[:, 0]
            Rs = results[:, 1]
            print("Time elapsed = {} seconds".format(time.perf_counter() - t0))
            return Us, Rs

        if a == 0.5:
            xi = np.linspace(0, 0.5, n_its)
        else:
            xi = np.linspace(0, 2, n_its)
        Us, Rs = compute_parallel()

        if dirichlet:
            text = "Dirichlet"
        else:
            text = "Neumann"

        folder = path.join("task2_conservation", "N{}_H{}_a{:.1f}_{}".format(n_its, H, a, text))
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(path.join(folder, "Rs.npy"), Rs)
        np.save(path.join(folder, "Us.npy"), Us)
        np.save(path.join(folder, "xi.npy"), xi)
