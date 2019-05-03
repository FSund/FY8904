from simulator import alpha0, LatticeSite, IncidentWave, abs2
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from os import path
import matplotlib as mpl
from matplotlib import rc

print(mpl.rcParams.get('figure.figsize'))
print(mpl.rcParams.get('legend.borderpad'))
mpl.rcParams['lines.linewidth'] = 1.0  # reduce default line width
mpl.rcParams['mathtext.fontset'] = 'cm'
rc('font', **{'family': 'serif', 'serif': ['CMU Serif Roman 2']})

n_its = 3600
H = 9
a = 3.5
LatticeSite.a = a
main_folder = "fig1_data"
phi0 = 0
xi0 = 0.5

folder = path.join(main_folder, "N{}_H{}_a{:.1f}".format(n_its, H, a))
folder = path.join(folder, "xi0_{:.1f}".format(xi0))
r = np.load(path.join(folder, "r.npy"))
theta = np.load(path.join(folder, "theta.npy"))

Hs = range(-H, H + 1)  # +1 to include endpoint
n = len(Hs)
N = n**2

# find the different values of e
# h = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
h = [[0, 0], [1, 0], [-1, 0], [0, 1], [1, 1], [-1, 1]]  # removed duplicates
fig, axes = plt.subplots(len(h), 1, sharex=True,
                         gridspec_kw={"hspace": 0.035},
                         figsize=[7, 9]
                         )
for ax, idx in zip(axes, range(len(h))):
    h1 = h[idx][0]
    h2 = h[idx][1]
    e = np.zeros([len(theta)], dtype=np.complex_)
    for i, theta0 in enumerate(theta):
        k = IncidentWave(theta0, phi0)

        # search through results for h1 and h2 matching the wanted ones
        for j in range(N):
            k1, k2 = divmod(j, n)
            if Hs[k1] == h1 and Hs[k2] == h2:
                G = LatticeSite(h1, h2)
                K = k + G
                r2 = abs2(r[i, j])
                e[i] = alpha0(K)/alpha0(k)*r2
    if idx > 2:
        label = r"$h = \{{{}, \pm{}\}}$".format(h1, h2)
    else:
        label = r"$h = \{{{}, {}\}}$".format(h1, h2)
    ax.plot(theta/pi*180, e, color="C0" if idx == 0 else "C1", label=label)

# add anomaly angles
theta0 = np.load("anomaly_angles_degrees.npy")
for theta in theta0:
    for ax in axes:
        ax.axvline(theta, color=(0.7, 0.7, 0.7), linestyle="--", linewidth=0.5)

axes[3].set_ylabel(r"Diffraction efficiency $e(\mathbf{k}_\| + \mathbf{G}_\|(h) \vert \mathbf{k}\|)$")

axes[-1].xaxis.set_minor_locator(MultipleLocator(2))

for ax in axes:
    ax.legend(loc="center left", fontsize="small",
              handlelength=1,
              # borderpad=0.2
              )
    ax.tick_params(axis="x", which="both", direction="in")

for ax in axes:
    ax.set_ylim([-0.0125, 0.22])
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

axes[0].set_ylim([-0.05, 1])
axes[0].yaxis.set_major_locator(MultipleLocator(0.2))

axes[3].set_ylim([-0.005, 0.11])
axes[3].yaxis.set_major_locator(MultipleLocator(0.02))

axes[-1].set_ylim([-0.0125, 0.248])

ax = axes[-1]
ax.set_xlabel(r"Polar angle of incidence $\theta_0$ [degrees]")
ax.set_xlim([0, 90])
# fig.tight_layout(h_pad=0.025, pad=0.1)
fig.subplots_adjust(top=0.98, bottom=0.05, right=0.99, left=0.1)
fig.savefig("report/figs/fig4.pdf", pad=0, bbox_inches="tight")

plt.show()

# e = np.zeros([len(h), len(theta)], dtype=np.complex_)
# for theta_idx, theta0 in enumerate(theta):
#     k = IncidentWave(theta0, phi0)
#     for idx in range(len(h)):
#         h1 = h[idx][0]
#         h2 = h[idx][0]
#         for i in range(N):  # search through all results
#             k1, k2 = divmod(i, n)
#             if Hs[k1] == h1 and Hs[k2] == h2:
#                 G = LatticeSite(h1, h2)
#                 K = k + G
#                 r2 = abs2(r[theta_idx, i])
#                 e[idx, theta_idx] = alpha0(K)/alpha0(k)*r2
#                 break
