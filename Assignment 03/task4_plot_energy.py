from math import pi
import numpy as np
from simulator import Simulator
from surface import TruncatedCone, TruncatedCosine, DoubleCosine
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from joblib import Parallel, delayed
from os import path
import os

mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif Roman 2'

# surface
H = 9
h2 = 0
phi0 = 0
# xi0 = 0.1  # find optimal value for this

# n_its = 3600
n_its = 1000
theta = np.linspace(-pi/2, pi/2, n_its+2)
theta = theta[1:-1]  # remove both endpoints

bcs = ["Dirichlet", "Neumann"]
surface_types = ["DoubleCosine", "TruncatedCone", "TruncatedCosine"]
surface_texts = ["Double cosine", "Truncated cone", "Truncated cosine"]

for a in [0.5, 3.5]:
    fig, ax = plt.subplots(figsize=[10, 5])  # energy conservation
    ax.set_title(r"$a = {}$".format(a))
    for dirichlet in [True, False]:
        for profile in range(3):
            if a == 0.5:
                xi0 = 0.7
            else:
                xi0 = 0.05
            bc = bcs[dirichlet]
            surface_type = surface_types[profile]

            # load results
            folder = path.join("task4_data", "N{}_H{}".format(n_its, H))
            folder = path.join(folder, "{}_{}_a{:.1f}_xi0_{:.2f}".format(bc, surface_type, a, xi0))
            U = np.load(path.join(folder, "U.npy"))
            # e_m = np.load(path.join(folder, "e_m.npy"))
            theta = np.load(path.join(folder, "theta.npy"))

            label = r"{} {}, $a = {}$, $\xi_0 = {}$".format(surface_texts[profile], bc, a, xi0)
            l, = ax.plot(theta/pi*180, U, label=label)
            if xi0 == 0.3:
                l.set_linestyle("dashed")

            print(label)
            print("Max abs energy conservation = {}".format(np.max(np.abs(U))))

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize="small")
    fig.subplots_adjust(right=0.7)

plt.show()
