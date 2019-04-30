from math import pi
import numpy as np
import matplotlib.pyplot as plt
from os import path
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

H = 9
n_its = 100

# fig, ax = plt.subplots()
# for dirichlet in [True, False]:
#     for a in [0.5, 3.5]:
#         if dirichlet:
#             text = "Dirichlet"
#         else:
#             text = "Neumann"
#         label = "N{}_H{}_a{:.1f}_{}".format(n_its, H, a, text)
#         folder = path.join("task2_conservation", label)
#         # Rs = np.load(path.join(folder, "Rs.npy"))
#         Us = np.load(path.join(folder, "Us.npy"))
#         xi = np.load(path.join(folder, "xi.npy"))

#         ax.plot(xi, Us, label=label)

# ax.legend()
# ax.set_xlabel(r"$\xi_0$")

fig, axes = plt.subplots(1, 2, sharey=True,# gridspec_kw={"wspace": 0.05},
                         figsize=(7, 2.5))
for a, ax in zip([0.5, 3.5], axes):
    for dirichlet in [True, False]:
        if dirichlet:
            text = "Dirichlet"
        else:
            text = "Neumann"
        label = "N{}_H{}_a{:.1f}_{}".format(n_its, H, a, text)
        folder = path.join("task2_conservation", label)
        Us = np.load(path.join(folder, "Us.npy"))
        xi = np.load(path.join(folder, "xi.npy"))

        legend = text
        ax.plot(xi, Us, label=legend)

        text = ax.text(0.05, 0.05, r"$a = {}$".format(a),
                       transform=ax.transAxes,
                       va="bottom",
                       ha="left",
                       bbox=dict(boxstyle="round",
                                 fc=(1.0, 1.0, 1.0, 0.8),  # default legend
                                 ec=(0.8, 0.8, 0.8, 0.8))  # fancybox colors
                       )
    legend = ax.legend()
    ax.set_xlabel(r"$\xi_0$")
    ax.set_xlim([xi[0], xi[-1]])
    ax.grid()

axes[0].set_ylim([0.9, 1.1])
axes[0].set_ylabel(r"$U(\xi_0)$")
axes[1].tick_params(axis="y", left=False)

# axes[0].set_title(r"a = 0.5")
# axes[1].set_title(r"a = 3.5")

fig.tight_layout(w_pad=0.05, pad=0)
fig.savefig("report/figs/task2.pdf", bbox_inches="tight", pad=0)

plt.show()
