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
mpl.rcParams['grid.color'] = (0.85, 0.85, 0.85)
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif Roman 2'

plt.close("all")

# surface
H = 9
h2 = 0
phi0 = 0
xi0 = 0.1  # find optimal value for this in task 4

n_its = 3600
theta = np.linspace(-pi/2, pi/2, n_its+2)
theta = theta[1:-1]  # remove both endpoints

bcs = ["Neumann", "Dirichlet"]
surface_types = ["DoubleCosine", "TruncatedCone", "TruncatedCosine"]
surface_texts = ["Double cosine", "Truncated cone", "Truncated cosine"]

fig, axes = plt.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.05}, figsize=[4, 3])
for dirichlet in [0, 1]:
    for a in [0.5, 3.5]:
        for profile in range(3):
            bc = bcs[dirichlet]
            surface_type = surface_types[profile]
            if a == 0.5:
                xi0 = 0.7
            else:
                xi0 = 0.05

            # load results
            folder = path.join("task4_data", "N{}_H{}".format(n_its, H))
            folder = path.join(folder, "{}_{}_a{:.1f}_xi0_{:.2f}".format(bc, surface_type, a, xi0))
            U = np.load(path.join(folder, "U.npy"))
            e_m = np.load(path.join(folder, "e_m.npy"))
            theta = np.load(path.join(folder, "theta.npy"))

            # label = r"{} {}, $a = {}$, $\xi_0={}$".format(surface_texts[profile], bc, a, xi0)
            label = bc
            for idx, m in enumerate(range(-H, H+1)):
                if not np.all(np.isnan(e_m[idx, :])):
                    if m == 0 and a == 3.5:
                        axes[profile].plot(theta/pi*180, e_m[idx, :], label=label)

for ax in axes:
    ax.grid()
    # ax.legend(loc="lower right", fontsize="small")

axes[0].legend(loc="lower center", fontsize="small", bbox_to_anchor=(0.5, 1), ncol=2)

for ax in axes[:-1]:
    ax.tick_params(axis="x", direction="in")

axes[-1].set_xlim([-90, 90])
axes[-1].set_xlabel(r"Polar angle of incidence $\theta_0$ [degrees]")
axes[-1].xaxis.set_major_locator(MultipleLocator(15))
axes[1].set_ylabel(r"Diffraction efficiency $e_{m=0}(\theta_0)$")

axes[0].set_ylim([0.925, 1])
axes[1].set_ylim([0.93, 1])
axes[2].set_ylim([0.965, 1])

for i in range(3):
    axes[i].text(0.5, 0.95, surface_texts[i],
                 transform=axes[i].transAxes,
                 va="top", ha="center"
                 )

fig.savefig("report/figs/task4_m0.pdf", pad=0, bbox_inches="tight")

plt.show()
