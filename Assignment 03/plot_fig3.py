from math import pi
import numpy as np
import matplotlib.pyplot as plt
from os import path

n_its = 10
for xi0 in [0.3, 0.5, 0.7]:
    folder = path.join("fig3_N{}".format(n_its), "xi0_{}".format(xi0))
    Rs = np.load(path.join(folder, "Rs.npy"))
    Us = np.load(path.join(folder, "Us.npy"))
    theta = np.load(path.join(folder, "theta.npy"))

    fig, ax = plt.subplots()
    ax.plot(theta/(2*pi)*360, Rs)
    ax.set_yscale("log")
    ax.set_ylim([3e-4, 1.1])


plt.show()
