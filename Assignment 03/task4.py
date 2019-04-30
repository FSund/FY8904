from math import pi
import numpy as np
from simulator import Simulator
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from os import path
import os

plt.close("all")

# surface
a = 3.5  # 0.5 or 3.5 for task 4
H = 9
h2 = 0
phi0 = 0
xi0 = 0.1  # find optimal value for this in task 4
dirichlet = True

n_its = 1000
theta = np.linspace(-pi/2, pi/2, n_its+2)
theta = theta[1:-1]  # remove both endpoints


def simulate(a, xi0, theta0, phi0, H, idx):
    sim = Simulator(a, xi0, dirichlet)
    r, U, e_m, m = sim.task4(theta0, H)
    return [U, e_m, m]


def compute(n_jobs=6):
    t0 = time.perf_counter()
    # results = Parallel(n_jobs=n_jobs)(delayed(simulate)(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta))
    results = [simulate(a, xi0, theta0, phi0, H, idx) for idx, theta0 in enumerate(theta)]
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

fig, ax = plt.subplots()
for idx, m in enumerate(range(-H, H+1)):
    if not np.all(np.isnan(e_m[idx, :])):
        ax.plot(theta/pi*180, e_m[idx, :], label=r"$m={}$".format(m))
ax.legend(fontsize="small", handlelength=1)
ax.set_ylabel(r"$e_m(\theta_0)$")
ax.set_xlabel(r"$\theta_0$")

fig, ax = plt.subplots()
ax.plot(theta/pi*180, U)
ax.set_ylabel(r"$U(\theta_0)$")
ax.set_xlabel(r"$\theta_0$")
ax.set_title("Energy conservation")

# folder = path.join("task4_data", "N{}_H{}_a{:.1f}".format(n_its, H, a))
# folder = path.join(folder, "xi0_{:.1f}".format(xi0))
# if not os.path.exists(folder):
#     os.makedirs(folder)
# np.save(path.join(folder, "Rs.npy"), Rs)
# np.save(path.join(folder, "Us.npy"), Us)
# np.save(path.join(folder, "theta.npy"), theta)

plt.show()
