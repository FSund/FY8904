
# just run make_fig1.py instead for xi0=0.5

# from math import pi
# import numpy as np
# from simulator import Simulator
# from joblib import Parallel, delayed
# import time
# from os import path
# import os

# # surface
# a = 3.5  # lattice dimension (dimensionless)
# H = 9
# phi0 = 0
# xi0 = 0.5

# n_its = 100
# theta = np.linspace(0, pi/2, n_its, endpoint=False)

# h = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
# for i in range(len(h)):
#     h1 = h[i][0]
#     h2 = h[i][0]

#     def simulate(a, xi0, theta0, phi0, H, idx):
#         sim = Simulator(a, xi0, dirichlet=False)
#         sim.simulate(theta0, phi0, H)
#         # U = sim.conservation()
#         # R = sim.reflectivity()
#         e = sim.diffraction_efficiency(h1, h2)

#         return e

#     def compute_parallel(n_jobs=6):
#         t0 = time.perf_counter()
#         # parallel computation
#         results = Parallel(n_jobs=n_jobs)(delayed(simulate)(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta))
#         results = np.array(results, dtype=np.complex_)
#         print("Time elapsed = {} seconds".format(time.perf_counter() - t0))
#         return results

#     results = compute_parallel()

#     folder = path.join("fig4_data", "N{}_H{}_a{:.1f}_xi0{:.1f}".format(n_its, H, a, xi0))
#     folder = path.join(folder, "idx{:2d}".format(i))
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     np.save(path.join(folder, "results.npy"), results)
#     np.save(path.join(folder, "theta.npy"), theta)
