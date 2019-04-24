# import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, MultipleLocator
import os

figsize = [6, 4]
small_figsize = [5, 4]
small_fontsize = None  # use default


def plotting():
    if False:
        mainfolder = "./cpp_results_old/cpp_results_square/"
        Ls = [100, 200, 300, 500, 700, 1000]
        its = 500
    else:
        mainfolder = "./cpp_results_square/"
        Ls = [100, 200, 300, 500, 700, 1000]
        its = 1000

    axes = []
    figs = []
    for i in range(3):
        fig, ax = plt.subplots(figsize=figsize)
        axes.append(ax)
        figs.append(fig)

    for L in Ls:
        subfolder = "L{}_{}iterations/".format(L, its)
        folder = mainfolder + subfolder
        p = pd.read_csv(folder + "probability.csv", delimiter=",", header=None)
        P = pd.read_csv(folder + "giant.csv", delimiter=",", header=None)
        # P2 = pd.read_csv(folder + "giant2.csv", delimiter=",", header=None)
        s = pd.read_csv(folder + "averageSize.csv", delimiter=",", header=None)
        # s2 = pd.read_csv(folder + "averageSize2.csv", delimiter=",", header=None)
        chi = pd.read_csv(folder + "susceptibility.csv", delimiter=",", header=None)
        # chi2 = pd.read_csv(folder + "susceptibility2.csv", delimiter=",", header=None)

        label = "N = %d" % (L*L)
        axes[0].plot(p, P, label=label, linewidth=1)
        # axes[0].plot(p, P2, label=label, linewidth=1)
        axes[1].plot(p, s, label=label, linewidth=1)
        # axes[1].plot(p, s2, label=label, linewidth=1)
        axes[2].plot(p, chi, label=label, linewidth=1)
        # axes[2].plot(p, chi2, label=label, linewidth=1)

    for ax in axes:
        ax.set_xlim([0.4, 0.6])
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(0.05))

    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel("Giant component")
    axes[1].set_ylim([0, 20000])
    axes[1].set_ylabel("Average cluster size")
    axes[2].set_ylim([0, 120000])
    axes[2].set_ylabel("Susceptibility")

    for ax in axes:
        ax.grid()
        ax.set_xlabel("Probability")

    plt.draw()
    for fig in figs:
        fig.tight_layout()

    figs[0].savefig("figs/giant.png", dpi=300, bbox_inces="tight")
    figs[1].savefig("figs/averageSize.png", dpi=300, bbox_inces="tight")
    figs[2].savefig("figs/susceptibility.png", dpi=300, bbox_inces="tight")
    # plt.show()


def find_coeffs_square():
    # task 5.3
    mainfolder = "./cpp_results_square/"
    Ls = np.array([100, 200, 300, 500, 700, 1000])
    its = 1000

    # load data
    p = np.array(pd.read_csv(mainfolder + "L{}_{}iterations/".format(Ls[0], its) + "probability.csv", delimiter=",", header=None))
    P = np.zeros([len(p), len(Ls)])
    s = np.zeros([len(p), len(Ls)])
    for idx, L in enumerate(Ls):
        subfolder = "L{}_{}iterations/".format(L, its)
        folder = mainfolder + subfolder
        P[:, idx] = np.array(pd.read_csv(folder + "giant.csv", delimiter=",", header=None)).transpose()
        s[:, idx] = np.array(pd.read_csv(folder + "averageSize.csv", delimiter=",", header=None)).transpose()
        # chi = pd.read_csv(folder + "susceptibility.csv", delimiter=",", header=None)

    # remove last value, since we get 0/nan/inf there
    P = P[:-1, :]
    s = s[:-1, :]
    p = p[:-1]

    xi = np.sqrt(Ls*Ls)

    # fit straight line to P vs xi for each q
    R2 = np.zeros([P.shape[0]])
    for idx in range(P.shape[0]):
        x = np.log(xi)
        y = np.log(P[idx, :])
        # ax.plot(x, y)
        # slope, intercept = np.polyfit(x, y, deg=1)
        rSquared = np.corrcoef(x, y)[0, 1]**2
        # slope, intercept = np.polyfit(x, y, 1)
        # rSquared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))

        R2[idx] = rSquared

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(p, R2)

    # indicate max value
    pc_idx = 4900 + np.argmax(R2[4900:])
    pc = p[pc_idx]  # critical/percolation threshold
    # print("pc = %f" % pc)
    ax.plot(pc, R2[pc_idx], "x")

    ax.set_xlabel("Probability $q$")
    ax.set_ylabel("Correlation coefficient $R^2$")
    ax.set_xlim([0.3, 0.7])
    ax.set_ylim([-0.05, 1.05])
    ax.grid()
    fig.tight_layout()
    fig.savefig("figs/rsquared.png", dpi=300, bbox_inches="tight")

    # plot loglog P vs xi
    fig, ax = plt.subplots(figsize=figsize)
    # for idx in list(range(P.shape[0]))[4000:6000:200]:
    if True:
        idx = pc_idx
        q = p[idx]
        x = np.log(xi)
        y = np.log(P[idx, :])
        l, = ax.plot(x, y, "o", label="q = %.4f" % q)
        coeffs = np.polyfit(x, y, deg=1)
        x2 = [0, 10]
        y2 = np.polyval(coeffs, x2)
        ax.plot(x2, y2, color="k", lw=1, label="Linear fit p0 = %.3f, p1 = %.4f" % (coeffs[0], coeffs[1]))
        beta_over_nu = coeffs[0]
        # print("First linear fit slope (beta/nu) = %f" % beta_over_nu)
    ax.legend()
    ax.set_xlim([x[0]*0.95, x[-1]*1.05])
    ax.set_ylim([-0.825, -0.475])
    # ax.set_xlabel("Log of size of the system")
    ax.set_xlabel("$\\log(\\xi)$")
    # ax.set_ylabel("Log of giant component")
    ax.set_ylabel("$\\log(P_\\infty)$")
    fig.tight_layout()
    fig.savefig("figs/giant_vs_xi.png", dpi=300, bbox_inches="tight")

    # average cluster size
    fig, ax = plt.subplots(figsize=figsize)
    smax = np.max(s, axis=0)
    x = np.log(xi)
    y = np.log(smax)
    for xi, yi, L in zip(x, y, Ls):
        ax.plot(xi, yi, "o", label="L = %d" % L)

    coeffs = np.polyfit(x, y, deg=1)
    x2 = [0, 10]
    y2 = np.polyval(coeffs, x2)
    ax.plot(x2, y2, color="k", linewidth=1, label="Linear fit p0 = %.2f, p1 = %.2f" % (coeffs[0], coeffs[1]))
    gamma_over_nu = coeffs[0]
    # print("Second linear fit slope (gamma/nu) = %f" % gamma_over_nu)
    # for idx in list(range(P.shape[0]))[4000:6000:200]:
    #     q = p[idx]
    #     x = np.log(xi)
    #     y = np.log(P[idx, :])
    #     l, = ax.plot(x, y, "o", label="q = %.2f" % q)
    #     coeffs = np.polyfit(x, y, deg=1)
    #     x2 = [0, 10]
    #     y2 = np.polyval(coeffs, x2)
    #     ax.plot(x2, y2, color=l.get_color())
    ax.legend()
    ax.set_xlim([x[0]*0.95, x[-1]*1.05])
    ax.set_ylim([5.5, 10.5])
    # ax.set_xlabel("Log of size of the system")
    # ax.set_ylabel("Log of maximum average cluster size")
    ax.set_xlabel("$\\log(\\xi)$")
    ax.set_ylabel("$\\max(\\langle s\\rangle)$")
    fig.tight_layout()
    fig.savefig("figs/clustersize_vs_xi.png", dpi=300, bbox_inches="tight")

    # plot |qmax - p_c| against xi
    fig, ax = plt.subplots(figsize=figsize)
    x = np.log(np.sqrt(Ls*Ls))
    y = np.log(np.abs(p[np.argmax(s, axis=0)] - pc))
    ax.plot(x, y, "o")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeffs = np.polyfit(x, y, deg=1)
    x2 = [0, 10]
    y2 = np.polyval(coeffs, x2)
    ax.plot(x2, y2, color="k", linewidth=1, label="Linear fit p0 = %.3f, p1 = %.3f" % (coeffs[0], coeffs[1]))
    one_over_nu = coeffs[0]
    # print("Third linear fit slope (1/nu) = %f" % one_over_nu)
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$\\log(\\xi)$")
    ax.set_ylabel("$\\log(|q_\\mathrm{max} - p_\\mathrm{c}|)$")
    fig.tight_layout()
    fig.savefig("figs/qmax_vs_xi.png", dpi=300, bbox_inches="tight")

    nu = 1/one_over_nu
    beta = beta_over_nu*nu
    gamma = gamma_over_nu*nu

    print("one over nu = %f" % one_over_nu)
    print("beta over nu = %f" % beta_over_nu)
    print("gamma over nu = %f" % gamma_over_nu)

    print("pc = %f" % pc)
    print("beta = %f" % beta)
    print("gamma = %f" % gamma)
    print("nu = %f" % nu)

    # plt.show()


def makeImages():
    L = 2000
    folder = "./cpp_results/images_L%d" % L
    files = os.listdir(folder)

    fig, axes = plt.subplots(2, 3, figsize=[8, 5], gridspec_kw={"wspace": 0.2, "hspace": 0.2}, subplot_kw={"aspect": "equal"})
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

    for file, ax in zip(files, axes):
        # load image from c++ code from file
        # use pd.read_csv for faster loading of large files (has c implementation)
        image = np.array(pd.read_csv(folder + "/" + file, header=None))

        ax.imshow(image, aspect="equal", origin="upper", vmin=0, vmax=1, cmap="Purples")

        # extract percentage from filename
        p = file.split("p")[1]
        p = float(p.rstrip(".csv"))
        ax.text(0.5, 1.04, "p=%.3f" % p, transform=ax.transAxes, ha="center")

    plt.draw()
    fig.savefig("figs/largest_cluster_L%d.png" % L, bbox_inces="tight", pad_inches=0, dpi=300)
    # plt.show()


def find_coeffs_triangular():
    # task 5.3
    mainfolder = "./cpp_results_triangular/"
    Ls = np.array([100, 200, 300, 500, 700, 1000])
    its = 1000

    # load data
    p = np.array(pd.read_csv(mainfolder + "L{}_{}iterations/".format(Ls[0], its) + "probability.csv", delimiter=",", header=None))
    P = np.zeros([len(p), len(Ls)])
    s = np.zeros([len(p), len(Ls)])
    for idx, L in enumerate(Ls):
        subfolder = "L{}_{}iterations/".format(L, its)
        folder = mainfolder + subfolder
        P[:, idx] = np.array(pd.read_csv(folder + "giant.csv", delimiter=",", header=None)).transpose()
        s[:, idx] = np.array(pd.read_csv(folder + "averageSize.csv", delimiter=",", header=None)).transpose()
        # chi = pd.read_csv(folder + "susceptibility.csv", delimiter=",", header=None)

    # remove last value, since we get 0/nan/inf there
    P = P[:-1, :]
    s = s[:-1, :]
    p = p[:-1]

    xi = np.sqrt(Ls*Ls)

    # fit straight line to P vs xi for each q
    R2 = np.zeros([P.shape[0]])
    for idx in range(P.shape[0]):
        x = np.log(xi)
        y = np.log(P[idx, :])
        # ax.plot(x, y)
        # slope, intercept = np.polyfit(x, y, deg=1)
        rSquared = np.corrcoef(x, y)[0, 1]**2
        # slope, intercept = np.polyfit(x, y, 1)
        # rSquared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))

        R2[idx] = rSquared

    fig, ax = plt.subplots(figsize=small_figsize)
    ax.plot(p, R2)

    # indicate max value
    pc_idx = 3400 + np.argmax(R2[3400:])
    pc = p[pc_idx]  # critical/percolation threshold
    # print("pc = %f" % pc)
    ax.plot(pc, R2[pc_idx], "x")

    ax.set_xlabel("Probability $q$")
    ax.set_ylabel("Correlation coefficient $R^2$")
    ax.set_xlim([0.3, 0.7])
    ax.set_ylim([-0.05, 1.05])
    ax.grid()
    fig.tight_layout()
    fig.savefig("figs/triangular_rsquared.png", dpi=300, bbox_inches="tight")

    # plot loglog P vs xi
    fig, ax = plt.subplots(figsize=small_figsize)
    # for idx in list(range(P.shape[0]))[4000:6000:200]:
    if True:
        idx = pc_idx
        q = p[idx]
        x = np.log(xi)
        y = np.log(P[idx, :])
        l, = ax.plot(x, y, "o", label="q = %.4f" % q)
        coeffs = np.polyfit(x, y, deg=1)
        x2 = [0, 10]
        y2 = np.polyval(coeffs, x2)
        ax.plot(x2, y2, color="k", lw=1, label="Linear fit\np0 = %.3f\np1 = %.4f" % (coeffs[0], coeffs[1]))
        beta_over_nu = coeffs[0]
        # print("First linear fit slope (beta/nu) = %f" % beta_over_nu)
    ax.legend(fontsize=small_fontsize)
    ax.set_xlim([x[0]*0.95, x[-1]*1.05])
    ax.set_ylim([-0.825, -0.475])
    # ax.set_xlabel("Log of size of the system")
    ax.set_xlabel("$\\log(\\xi)$")
    # ax.set_ylabel("Log of giant component")
    ax.set_ylabel("$\\log(P_\\infty)$")
    fig.tight_layout()
    fig.savefig("figs/triangular_giant_vs_xi.png", dpi=300, bbox_inches="tight")

    # average cluster size
    fig, ax = plt.subplots(figsize=small_figsize)
    smax = np.max(s, axis=0)
    x = np.log(xi)
    y = np.log(smax)
    for xi, yi, L in zip(x, y, Ls):
        ax.plot(xi, yi, "o", label="L = %d" % L)

    coeffs = np.polyfit(x, y, deg=1)
    x2 = [0, 10]
    y2 = np.polyval(coeffs, x2)
    ax.plot(x2, y2, color="k", linewidth=1, label="Linear fit\np0 = %.2f\np1 = %.2f" % (coeffs[0], coeffs[1]))
    gamma_over_nu = coeffs[0]
    # print("Second linear fit slope (gamma/nu) = %f" % gamma_over_nu)
    # for idx in list(range(P.shape[0]))[4000:6000:200]:
    #     q = p[idx]
    #     x = np.log(xi)
    #     y = np.log(P[idx, :])
    #     l, = ax.plot(x, y, "o", label="q = %.2f" % q)
    #     coeffs = np.polyfit(x, y, deg=1)
    #     x2 = [0, 10]
    #     y2 = np.polyval(coeffs, x2)
    #     ax.plot(x2, y2, color=l.get_color())
    ax.legend(fontsize=small_fontsize)
    ax.set_xlim([x[0]*0.95, x[-1]*1.05])
    ax.set_ylim([5.5, 10.5])
    # ax.set_xlabel("Log of size of the system")
    # ax.set_ylabel("Log of maximum average cluster size")
    ax.set_xlabel("$\\log(\\xi)$")
    ax.set_ylabel("$\\max(\\langle s\\rangle)$")
    fig.tight_layout()
    fig.savefig("figs/triangular_clustersize_vs_xi.png", dpi=300, bbox_inches="tight")

    # plot |qmax - p_c| against xi
    fig, ax = plt.subplots(figsize=small_figsize)
    x = np.log(np.sqrt(Ls*Ls))
    y = np.log(np.abs(p[np.argmax(s, axis=0)] - pc))
    ax.plot(x, y, "o")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeffs = np.polyfit(x, y, deg=1)
    x2 = [0, 10]
    y2 = np.polyval(coeffs, x2)
    ax.plot(x2, y2, color="k", linewidth=1, label="Linear fit\np0 = %.3f\np1 = %.3f" % (coeffs[0], coeffs[1]))
    one_over_nu = coeffs[0]
    # print("Third linear fit slope (1/nu) = %f" % one_over_nu)
    ax.legend(fontsize=small_fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$\\log(\\xi)$")
    ax.set_ylabel("$\\log(|q_\\mathrm{max} - p_\\mathrm{c}|)$")
    fig.tight_layout()
    fig.savefig("figs/triangular_qmax_vs_xi.png", dpi=300, bbox_inches="tight")

    nu = -1/one_over_nu
    beta = -beta_over_nu*nu
    gamma = gamma_over_nu*nu

    print("pc = %f" % pc)
    print("beta = %f" % beta)
    print("gamma = %f" % gamma)
    print("nu = %f" % nu)

    # print("beta (with exact nu) = %f" % (-beta_over_nu*4/3.0))
    # print("gamma (with exact nu) = %f" % (gamma_over_nu*4/3.0))


def plotting_triangular():
    mainfolder = "./cpp_results_triangular/"
    Ls = [100, 200, 300, 500, 700, 1000]
    its = 1000

    axes = []
    figs = []
    for i in range(3):
        fig, ax = plt.subplots(figsize=figsize)
        axes.append(ax)
        figs.append(fig)

    for L in Ls:
        subfolder = "L{}_{}iterations/".format(L, its)
        folder = mainfolder + subfolder
        p = pd.read_csv(folder + "probability.csv", delimiter=",", header=None)
        P = pd.read_csv(folder + "giant.csv", delimiter=",", header=None)
        # P2 = pd.read_csv(folder + "giant2.csv", delimiter=",", header=None)
        s = pd.read_csv(folder + "averageSize.csv", delimiter=",", header=None)
        # s2 = pd.read_csv(folder + "averageSize2.csv", delimiter=",", header=None)
        chi = pd.read_csv(folder + "susceptibility.csv", delimiter=",", header=None)
        # chi2 = pd.read_csv(folder + "susceptibility2.csv", delimiter=",", header=None)

        label = "N = %d" % (L*L)
        axes[0].plot(p, P, label=label, linewidth=1)
        # axes[0].plot(p, P2, label=label, linewidth=1)
        axes[1].plot(p, s, label=label, linewidth=1)
        # axes[1].plot(p, s2, label=label, linewidth=1)
        axes[2].plot(p, chi, label=label, linewidth=1)
        # axes[2].plot(p, chi2, label=label, linewidth=1)

    for ax in axes:
        ax.set_xlim([0.25, 0.45])
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(0.05))

    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel("Giant component")
    axes[1].set_ylim([0, 20000])
    axes[1].set_ylabel("Average cluster size")
    axes[2].set_ylim([0, 120000])
    axes[2].set_ylabel("Susceptibility")

    for ax in axes:
        ax.grid()
        ax.set_xlabel("Probability")

    plt.draw()
    for fig in figs:
        fig.tight_layout()

    # figs[0].savefig("figs/giant.png", dpi=300, bbox_inces="tight")
    # figs[1].savefig("figs/averageSize.png", dpi=300, bbox_inces="tight")
    # figs[2].savefig("figs/susceptibility.png", dpi=300, bbox_inces="tight")


def plotting_honeycomb():
    mainfolder = "./cpp_results_honeycomb/"
    Ls = [100, 200, 300, 500, 700, 1000]
    its = 1000

    axes = []
    figs = []
    for i in range(3):
        fig, ax = plt.subplots(figsize=figsize)
        axes.append(ax)
        figs.append(fig)

    for L in Ls:
        subfolder = "L{}_{}iterations/".format(L, its)
        folder = mainfolder + subfolder
        p = pd.read_csv(folder + "probability.csv", delimiter=",", header=None)
        P = pd.read_csv(folder + "giant.csv", delimiter=",", header=None)
        # P2 = pd.read_csv(folder + "giant2.csv", delimiter=",", header=None)
        s = pd.read_csv(folder + "averageSize.csv", delimiter=",", header=None)
        # s2 = pd.read_csv(folder + "averageSize2.csv", delimiter=",", header=None)
        chi = pd.read_csv(folder + "susceptibility.csv", delimiter=",", header=None)
        # chi2 = pd.read_csv(folder + "susceptibility2.csv", delimiter=",", header=None)

        label = "N = %d" % (L*L)
        axes[0].plot(p, P, label=label, linewidth=1)
        # axes[0].plot(p, P2, label=label, linewidth=1)
        axes[1].plot(p, s, label=label, linewidth=1)
        # axes[1].plot(p, s2, label=label, linewidth=1)
        axes[2].plot(p, chi, label=label, linewidth=1)
        # axes[2].plot(p, chi2, label=label, linewidth=1)

    for ax in axes:
        ax.set_xlim([0.55, 0.75])
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(0.05))

    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel("Giant component")
    axes[1].set_ylim([0, 20000])
    axes[1].set_ylabel("Average cluster size")
    axes[2].set_ylim([0, 120000])
    axes[2].set_ylabel("Susceptibility")

    for ax in axes:
        ax.grid()
        ax.set_xlabel("Probability")

    plt.draw()
    for fig in figs:
        fig.tight_layout()


def find_coeffs_honeycomb():
    # task 5.3
    mainfolder = "./cpp_results_honeycomb/"
    Ls = np.array([100, 200, 300, 500, 700, 1000])
    its = 1000

    # load data
    p = np.array(pd.read_csv(mainfolder + "L{}_{}iterations/".format(Ls[0], its) + "probability.csv", delimiter=",", header=None))
    P = np.zeros([len(p), len(Ls)])
    s = np.zeros([len(p), len(Ls)])
    for idx, L in enumerate(Ls):
        subfolder = "L{}_{}iterations/".format(L, its)
        folder = mainfolder + subfolder
        P[:, idx] = np.array(pd.read_csv(folder + "giant.csv", delimiter=",", header=None)).transpose()
        s[:, idx] = np.array(pd.read_csv(folder + "averageSize.csv", delimiter=",", header=None)).transpose()
        # chi = pd.read_csv(folder + "susceptibility.csv", delimiter=",", header=None)

    # remove last value, since we get 0/nan/inf there
    P = P[:-1, :]
    s = s[:-1, :]
    p = p[:-1]

    xi = np.sqrt(Ls*Ls)

    # fit straight line to P vs xi for each q
    R2 = np.zeros([P.shape[0]])
    for idx in range(P.shape[0]):
        x = np.log(xi)
        y = np.log(P[idx, :])
        # ax.plot(x, y)
        # slope, intercept = np.polyfit(x, y, deg=1)
        rSquared = np.corrcoef(x, y)[0, 1]**2
        # slope, intercept = np.polyfit(x, y, 1)
        # rSquared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))

        R2[idx] = rSquared

    fig, ax = plt.subplots(figsize=small_figsize)
    ax.plot(p, R2)

    # indicate max value
    pc_idx = 6500 + np.argmax(R2[6500:])
    pc = p[pc_idx]  # critical/percolation threshold
    # print("pc = %f" % pc)
    ax.plot(pc, R2[pc_idx], "x")

    ax.set_xlabel("Probability $q$")
    ax.set_ylabel("Correlation coefficient $R^2$")
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([-0.05, 1.05])
    ax.grid()
    fig.tight_layout()
    fig.savefig("figs/honeycomb_rsquared.png", dpi=300, bbox_inches="tight")

    # plot loglog P vs xi
    fig, ax = plt.subplots(figsize=small_figsize)
    # for idx in list(range(P.shape[0]))[4000:6000:200]:
    if True:
        idx = pc_idx
        q = p[idx]
        x = np.log(xi)
        y = np.log(P[idx, :])
        l, = ax.plot(x, y, "o", label="q = %.4f" % q)
        coeffs = np.polyfit(x, y, deg=1)
        x2 = [0, 10]
        y2 = np.polyval(coeffs, x2)
        ax.plot(x2, y2, color="k", lw=1, label="Linear fit\np0 = %.3f\np1 = %.4f" % (coeffs[0], coeffs[1]))
        beta_over_nu = coeffs[0]
        # print("First linear fit slope (beta/nu) = %f" % beta_over_nu)
    ax.legend(fontsize=small_fontsize)
    ax.set_xlim([x[0]*0.95, x[-1]*1.05])
    ax.set_ylim([-0.825, -0.475])
    # ax.set_xlabel("Log of size of the system")
    ax.set_xlabel("$\\log(\\xi)$")
    # ax.set_ylabel("Log of giant component")
    ax.set_ylabel("$\\log(P_\\infty)$")
    fig.tight_layout()
    fig.savefig("figs/honeycomb_giant_vs_xi.png", dpi=300, bbox_inches="tight")

    # average cluster size
    fig, ax = plt.subplots(figsize=small_figsize)
    smax = np.max(s, axis=0)
    x = np.log(xi)
    y = np.log(smax)
    for xi, yi, L in zip(x, y, Ls):
        ax.plot(xi, yi, "o", label="L = %d" % L)

    coeffs = np.polyfit(x, y, deg=1)
    x2 = [0, 10]
    y2 = np.polyval(coeffs, x2)
    ax.plot(x2, y2, color="k", linewidth=1, label="Linear fit\np0 = %.2f\np1 = %.2f" % (coeffs[0], coeffs[1]))
    gamma_over_nu = coeffs[0]
    # print("Second linear fit slope (gamma/nu) = %f" % gamma_over_nu)
    # for idx in list(range(P.shape[0]))[4000:6000:200]:
    #     q = p[idx]
    #     x = np.log(xi)
    #     y = np.log(P[idx, :])
    #     l, = ax.plot(x, y, "o", label="q = %.2f" % q)
    #     coeffs = np.polyfit(x, y, deg=1)
    #     x2 = [0, 10]
    #     y2 = np.polyval(coeffs, x2)
    #     ax.plot(x2, y2, color=l.get_color())
    ax.legend(fontsize=small_fontsize)
    ax.set_xlim([x[0]*0.95, x[-1]*1.05])
    ax.set_ylim([5.5, 10.5])
    # ax.set_xlabel("Log of size of the system")
    # ax.set_ylabel("Log of maximum average cluster size")
    ax.set_xlabel("$\\log(\\xi)$")
    ax.set_ylabel("$\\max(\\langle s\\rangle)$")
    fig.tight_layout()
    fig.savefig("figs/honeycomb_clustersize_vs_xi.png", dpi=300, bbox_inches="tight")

    # plot |qmax - p_c| against xi
    fig, ax = plt.subplots(figsize=small_figsize)
    x = np.log(np.sqrt(Ls*Ls))
    y = np.log(np.abs(p[np.argmax(s, axis=0)] - pc))
    ax.plot(x, y, "o")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeffs = np.polyfit(x, y, deg=1)
    x2 = [0, 10]
    y2 = np.polyval(coeffs, x2)
    ax.plot(x2, y2, color="k", linewidth=1, label="Linear fit\np0 = %.3f\np1 = %.3f" % (coeffs[0], coeffs[1]))
    one_over_nu = coeffs[0]
    # print("Third linear fit slope (1/nu) = %f" % one_over_nu)
    ax.legend(fontsize=small_fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$\\log(\\xi)$")
    ax.set_ylabel("$\\log(|q_\\mathrm{max} - p_\\mathrm{c}|)$")
    fig.tight_layout()
    fig.savefig("figs/honeycomb_qmax_vs_xi.png", dpi=300, bbox_inches="tight")

    nu = -1/one_over_nu
    beta = -beta_over_nu*nu
    gamma = gamma_over_nu*nu

    print("pc = %f" % pc)
    print("beta = %f" % beta)
    print("gamma = %f" % gamma)
    print("nu = %f" % nu)


if __name__ == "__main__":
    # makeImages()

    # square grid
    print("Square grid")
    plotting()
    find_coeffs_square()

    # triangular grid
    print("\nTriangular grid")
    plotting_triangular()
    find_coeffs_triangular()

    print("\nHoneycomb")
    plotting_honeycomb()
    find_coeffs_honeycomb()

    # plt.show()


