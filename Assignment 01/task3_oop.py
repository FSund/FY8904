if False:
    ## task 3.1
    # open file for reading
    with open("squareLattice.csv", "r") as f:
        # read number of sites and number of bonds from first line
        line = f.readline()
        line = line.strip()  # remove \n and \r characters
        line = line.replace(" ", "")  # remove whitespace
        line = line.split(",")  # split string at ","
        N = int(line[0])
        M = int(line[1])

        # read sites/bonds
        bonds = []
        for line in f:  # read the rest of the lines in the file
            line = line.strip()  # remove \n and \r characters
            line = line.replace(" ", "")  # remove whitespace
            # split string at "," and convert result from string to int
            bond = [int(value) for value in line.split(",")]
            bonds.append(bond)

    print("number of bonds: {}".format(M))
    print("number of sites: {}".format(N))
    print("bonds:")
    print(bonds)

    ## task 3.2
    import random
    random.seed(2)
    for i in range(10):
        random.random()

    ## task 3.3
    from latticemaker import shuffleList
    bonds = shuffleList(bonds)
    print("bonds after shuffling")
    print(bonds)

## task 3.4
# sites contain the status of each node
# If node i is the root node of a cluster, then sites[i] contains the number
# of nodes in the cluster. If not, then sites[i] contains the index of the root
# node of the cluster it belongs to.
# Since we keep both sizes and indexes in the same array, we code sizes using
# negative numbers, and indexes using positive
# So the array will initially look like [-1, -1, ...], since all nodes are root
# nodes of clusters with size 1
from sites import Sites


## task 3.8
from latticemaker import makeSquareLattice, shuffleList
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from math import ceil

plt.close("all")
fig, ax = plt.subplots(figsize=[5, 5], tight_layout={"pad": 0.1})
ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False,
    length=0,
)

L = 1000
N = L*L
sites = Sites(L, L)
bonds = makeSquareLattice(L, L)
bonds = shuffleList(bonds)

nSteps = 10
step = round(len(bonds)/nSteps)
# assert nSteps*step == len(bonds), "nSteps should be a divisor of len(bonds)"
if nSteps*step != len(bonds):
    print("warning: nSteps not a divisor of len(bonds)")
nn = round(len(bonds)/step)
print("step: " + str(step))
print("nn: " + str(nn))

fig2, axes = plt.subplots(2, ceil(nSteps/2), figsize=[10, 3.5], gridspec_kw={"wspace": 0.4, "hspace": 0.1}, subplot_kw={"aspect": "equal"})
axes = [y for x in axes for y in x]  # flatten
for ax in axes:
    ax.xaxis.set_major_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(
        axis='both',
        which='major',
        bottom=False,
        left=False,
        # top=False,
        labelbottom=False,
        labelleft=False,
        # length=0,
    )

P = np.zeros(len(bonds))
size = np.zeros(len(bonds))
s = np.zeros(len(bonds))
pvec = np.zeros(len(bonds))
j = 0
for i in range(nn):
    for bond in bonds[i*step:(i+1)*step]:
        sites.activate([bond])
        P[j] = sites.giantComponent
        s[j] = sites.averageSquaredSize
        size[j] = sites.sizeOfLargestCluster
        pvec[j] = j/len(bonds)
        j += 1

    print("start: {}, stop: {}, index: {}".format(i*step, (i+1)*step, i))

    image = sites.makeImage()
    ax.imshow(image, aspect="equal", origin="upper", vmin=0, vmax=1, cmap="Purples")
    axes[i].imshow(image, aspect="equal", origin="upper", vmin=0, vmax=1, cmap="Purples")

    # p = np.sum(np.sum(image))/(L*L)
    # print(p)
    p = (i+1)*step/len(bonds)
    print(p)
    # axes[i].text(0.02, 0.02, "p=%.2f" % p, transform=axes[i].transAxes, color="grey")
    axes[i].text(0.5, 1.04, "p=%.2f" % p, transform=axes[i].transAxes, ha="center")

    plt.draw()
    fig.savefig("figs/%03d.png" % i, bbox_inces="tight")
    # fig.savefig("figs/%03d.png" % i, bbox_inces=ax.bbox)

plt.close(fig)
plt.show()
fig2.savefig("test.png", bbox_inches="tight", dpi=300)
# plt.close(fig2)

fig, ax = plt.subplots()
ax.plot(pvec, P, label="N=%d" % N)
ax.set_xlim([0.4, 0.6])
ax.set_ylim([0, 1])
ax.legend()

fig, ax = plt.subplots()
ax.plot(pvec, s)
ax.set_xlim([0.4, 0.6])

plt.show()