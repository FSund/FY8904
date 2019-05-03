from math import pi
import numpy as np
from simulator import Simulator
from surface import TruncatedCone, TruncatedCosine, DoubleCosine
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
from joblib import Parallel, delayed
import time
from os import path
import os

# plt.close("all")

# surface
H = 9
h2 = 0
phi0 = 0

n_its = 3600
theta = np.linspace(-pi/2, pi/2, n_its+2)
theta = theta[1:-1]  # remove both endpoints

for a in [0.5, 3.5]:
    for dirichlet in [True, False]:
        if a == 0.5:
            xi0 = 0.7
        else:
            xi0 = 0.05
        for profile in range(3):
            if profile == 0:
                surface = DoubleCosine(a, xi0)
            elif profile == 1:
                rho_b = (a/2)*0.8
                rho_t = rho_b*0.8
                surface = TruncatedCone(a, xi0, rho_t, rho_b)
            elif profile == 2:
                rho0 = (a/2)*0.8
                surface = TruncatedCosine(a, xi0, rho0)

            def simulate(a, xi0, theta0, phi0, H, idx):
                sim = Simulator(a, xi0, dirichlet, surface)
                r, U, e_m, m = sim.task4(theta0, H)
                return [U, e_m, m]

            def compute(n_jobs=6):
                t0 = time.perf_counter()
                results = Parallel(n_jobs=n_jobs)(delayed(simulate)(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta))
                # results = [simulate(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta)]
                print("Time elapsed = {} seconds".format(time.perf_counter() - t0))
                return results

            results = compute()

            # sort through results
            ms = np.array(range(-H, H+1))
            e_m = np.zeros([len(ms), n_its])
            for m in range(-H, H+1):
                for i in range(n_its):
                    e_m[m, i] = results[i][1][m]

            U = np.zeros(n_its)
            for i in range(n_its):
                U[i] = results[i][0]

            if dirichlet:
                bc = "Dirichlet"
            else:
                bc = "Neumann"
            if profile == 0:
                surface_type = "DoubleCosine"
            elif profile == 1:
                surface_type = "TruncatedCone"
            elif profile == 2:
                surface_type = "TruncatedCosine"

            # save results to file
            folder = path.join("task4_data", "N{}_H{}".format(n_its, H))
            folder = path.join(folder, "{}_{}_a{:.1f}_xi0_{:.2f}".format(bc, surface_type, a, xi0))
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save(path.join(folder, "U.npy"), U)
            np.save(path.join(folder, "e_m.npy"), e_m)
            np.save(path.join(folder, "theta.npy"), theta)
