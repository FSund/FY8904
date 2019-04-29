from math import pi
import numpy as np
import matplotlib.pyplot as plt
from os import path

H = 12
n_its = 100
fig, ax = plt.subplots()
for dirichlet in [True, False]:
    for a in [0.5, 3.5]:
        if dirichlet:
            text = "Dirichlet"
        else:
            text = "Neumann"
        label = "N{}_H{}_a{:.1f}_{}".format(n_its, H, a, text)
        folder = path.join("task2_conservation", label)
        # Rs = np.load(path.join(folder, "Rs.npy"))
        Us = np.load(path.join(folder, "Us.npy"))
        xi = np.load(path.join(folder, "xi.npy"))

        ax.plot(xi, Us, label=label)

# ax.set_yscale("log")
# ax.set_ylim([3e-4, 1.1])
ax.legend()
ax.set_xlabel(r"$\xi_0$")


plt.show()
