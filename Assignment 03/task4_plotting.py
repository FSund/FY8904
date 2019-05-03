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

n_its = 3600
theta = np.linspace(-pi/2, pi/2, n_its+2)
theta = theta[1:-1]  # remove both endpoints

bcs = ["Dirichlet", "Neumann"]
surface_types = ["DoubleCosine", "TruncatedCone", "TruncatedCosine"]
surface_texts = ["Double cosine", "Truncated cone", "Truncated cosine"]

U_fig, U_ax = plt.subplots()  # energy conservation
axes = []
figs = []
for dirichlet in [True, False]:
    for a in [0.5, 3.5]:
        for profile in range(3):
            bc = bcs[dirichlet]
            surface_type = surface_types[profile]
            if a == 3.5:
                xi0 = 0.3
            else:
                xi0 = 0.1

            # load results
            folder = path.join("task4_data", "N{}_H{}".format(n_its, H))
            folder = path.join(folder, "{}_{}_a{:.1f}_xi0_{:.1f}".format(bc, surface_type, a, xi0))
            U = np.load(path.join(folder, "U.npy"))
            e_m = np.load(path.join(folder, "e_m.npy"))
            theta = np.load(path.join(folder, "theta.npy"))

            fig, ax = plt.subplots()
            axes.append(ax)
            figs.append(fig)
            for idx, m in enumerate(range(-H, H+1)):
                if not np.all(np.isnan(e_m[idx, :])):
                    ax.plot(theta/pi*180, e_m[idx, :], label=r"$m={}$".format(m))

            label = r"{} {}, $a = {}$".format(surface_texts[profile], bc, a)
            ax.set_title(label)

            if a == 3.5:
                label2 = r"{}, {}".format(bc, surface_texts[profile])
                U_ax.plot(theta/pi*180, U, label=label2)
                ax.set_yscale("log")

                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
                          fontsize="small", handlelength=1)
            else:
                ax.legend(fontsize="small", handlelength=1)

            print(label)
            print("Max abs energy conservation = {}".format(np.max(np.abs(U))))

for ax in axes:
    ax.set_xlim([-90, 90])
    ax.set_xlabel(r"Polar angle of incidence $\theta_0$ [degrees]")
    ax.set_ylabel(r"Diffraction efficiency $e_m(\theta_0)$")
    ax.xaxis.set_major_locator(MultipleLocator(15))

for fig in figs:
    fig.subplots_adjust(right=0.85)

U_ax.set_ylabel(r"$U(\theta_0)$")
U_ax.set_xlabel(r"$\theta_0$")
# U_ax.set_title("Energy conservation")
U_ax.set_title(r"Energy conservation $\xi_0 = {}, a = 3.5$".format(xi0))
U_ax.xaxis.set_major_locator(MultipleLocator(15))
U_ax.legend()
# U_fig.suptitle(r"{}, $a = {}$, {}".format(bc, a, surface_type))

plt.show()
