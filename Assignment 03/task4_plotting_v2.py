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

U_fig, U_ax = plt.subplots()  # energy conservation
axes = []
figs = []
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

            label = r"{} {}, $a = {}$, $\xi_0={}$".format(surface_texts[profile], bc, a, xi0)
            # figsize = [3.5, 3]  # textwidth is 7
            # figsize = [6.4, 4.8]  # default [6.4, 4.8]
            figsize = [5, 4]
            if a == 3.5:
                fig, ax = plt.subplots(2, 1, sharex=True,
                                       figsize=figsize,
                                       gridspec_kw={
                                        "height_ratios": [1, 3],
                                        "hspace": 0.02,
                                       })
                top_ax = ax[0]
                ax = ax[1]
                top_ax.set_title(label, fontsize=10)
                top_ax.tick_params(axis="x", direction="in")
                top_ax.grid()
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_title(label, fontsize=10)
            axes.append(ax)
            figs.append(fig)
            lhandles = []
            labels = []
            for idx, m in enumerate(range(-H, H+1)):
                if not np.all(np.isnan(e_m[idx, :])):
                    if m < 0:
                        linestyle = "--"
                    else:
                        linestyle = "-"
                    legend_label = r"$m={}$".format(m)
                    if a == 3.5 and m == 0:
                        l, = top_ax.plot(theta/pi*180, e_m[idx, :], label=legend_label, linestyle=linestyle)
                    else:
                        l, = ax.plot(theta/pi*180, e_m[idx, :], label=legend_label, linestyle=linestyle)
                        lhandles.append(l)
                        labels.append(legend_label)

            if a == 3.5:
                label2 = r"{}, {}".format(bc, surface_texts[profile])
                U_ax.plot(theta/pi*180, U, label=label2)
                ax.set_yscale("log")

                ax.legend(lhandles, labels, loc="center left", bbox_to_anchor=(1, 0.5),
                          fontsize="small", handlelength=1)
                top_ax.legend(loc="lower right")
                if not dirichlet:
                    if profile == 0:
                        top_ax.set_ylim([0.925, 1])
                    elif profile == 1:
                        top_ax.set_ylim([0.93, 1])
                    elif profile == 2:
                        top_ax.set_ylim([0.965, 1])
                    # top_ax.yaxis.set_minor_locator(MultipleLocator(0.025))
                    if profile == 2:
                        top_ax.yaxis.set_major_locator(MultipleLocator(0.01))
                    else:
                        top_ax.yaxis.set_major_locator(MultipleLocator(0.02))
                else:
                    # top_ax.set_ylim([0.75, 1])
                    # top_ax.yaxis.set_minor_locator(MultipleLocator(0.001))
                    top_ax.yaxis.set_major_locator(MultipleLocator(0.01))
            else:
                ax.legend(fontsize="small", handlelength=1)

            print(label)
            print("Max abs energy conservation = {}".format(np.max(np.abs(U))))

# FOR DEVELOPMENT
# for fig in figs[:-1]:
#     plt.close(fig)

for ax in axes:
    ax.set_xlim([-90, 90])
    ax.set_xlabel(r"Polar angle of incidence $\theta_0$ [degrees]")
    ax.set_ylabel(r"Diffraction efficiency $e_m(\theta_0)$")
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.grid()

for idx, fig in enumerate(figs):
    fig.subplots_adjust(right=0.8)
    fig.savefig("report/figs/task4_fig{:02d}.pdf".format(idx), pad=0, bbox_inches="tight")

U_ax.set_ylabel(r"$U(\theta_0)$")
U_ax.set_xlabel(r"$\theta_0$")
# U_ax.set_title("Energy conservation")
U_ax.set_title(r"Energy conservation $\xi_0 = {}, a = 3.5$".format(xi0))
U_ax.xaxis.set_major_locator(MultipleLocator(15))
U_ax.legend()
# U_fig.suptitle(r"{}, $a = {}$, {}".format(bc, a, surface_type))

plt.show()
