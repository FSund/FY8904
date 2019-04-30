from math import pi
import numpy as np
import matplotlib.pyplot as plt
from os import path
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

fig, axes = plt.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0.05})

# add anomaly angles
theta0 = np.load("anomaly_angles_degrees.npy")
for theta in theta0:
    for ax in axes:
        ax.axvline(theta, color=(0.7, 0.7, 0.7), linestyle="--", linewidth=0.5)

n_its = 100
H = 9
a = 3.5
main_folder = "fig3_data"
for ax, xi0 in zip(axes, [0.3, 0.5, 0.7]):
    folder = path.join(main_folder, "N{}_H{}_a{:.1f}".format(n_its, H, a))
    folder = path.join(folder, "xi0_{:.1f}".format(xi0))
    Rs = np.load(path.join(folder, "Rs.npy"))
    Us = np.load(path.join(folder, "Us.npy"))
    theta = np.load(path.join(folder, "theta.npy"))

    ax.plot(theta/(2*pi)*360, Rs)
    ax.set_yscale("log")
    ax.set_ylim([3e-4, 1.1])

for ax in axes[0:2]:
    ax.tick_params(axis="x", bottom=False)

axes[0].set_ylim([4e-3, 1.1])
axes[1].set_ylabel(r"Reflectivity")

ax = axes[2]
ax.set_xlabel(r"$\theta_0$ [degrees]")
ax.set_xlim([0, 90])

# plot energy conservation
fig, ax = plt.subplots()
for xi0 in [0.3, 0.5, 0.7]:
    folder = path.join(main_folder, "N{}_H{}_a{:.1f}".format(n_its, H, a))
    folder = path.join(folder, "xi0_{:.1f}".format(xi0))
    theta = np.load(path.join(folder, "theta.npy"))
    Us = np.load(path.join(folder, "Us.npy"))

    ax.plot(theta/(2*pi)*360, Us, label=r"$\xi_0 = {}$".format(xi0))
    print("max U(xi0={}) = {}".format(xi0, np.max(Us)))
ax.legend()
ax.set_xlabel(r"$\theta_0$ [degrees]")
ax.set_xlim([0, 90])
ax.set_title("Energy conservation")

# fig.tight_layout()
plt.tight_layout()
plt.show()
