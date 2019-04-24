if True:
    import matplotlib.pyplot as plt
    import numpy as np

    N = 100000
    im = np.random.rand(100, N)

    fig, ax = plt.subplots(figsize=[3.5, 2.5])
    ax.imshow(im, aspect="auto", extent=[0, 1, 0, 1], interpolation='nearest')
    fig.savefig("test_raster_default.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=[3.5, 2.5])
    ax.set_rasterization_zorder(0.5)
    ax.imshow(im, aspect="auto", rasterized=True, extent=[0, 1, 0, 1], zorder=0, interpolation='nearest')
    fig.savefig("test_rasterized_true.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=[3.5, 2.5])
    ax.set_rasterization_zorder(0.5)
    ax.imshow(im, aspect="auto", extent=[0, 1, 0, 1], zorder=0, interpolation='nearest')
    fig.savefig("test_rasterized_false.pdf", dpi=300)
else:
    import matplotlib.pyplot as plt
    from numpy.random import randn

    # doesn't work
    fig, ax = plt.subplots()
    ax.plot(randn(100), randn(100, 500), "k", alpha=0.03)
    fig.savefig("test_rasterized_fail.pdf", dpi=90)

    fig, ax = plt.subplots()
    ax.set_rasterization_zorder(0.5)
    ax.plot(randn(100), randn(100, 500), "k", alpha=0.03, zorder=0)
    fig.savefig("test_rasterized_works.pdf", dpi=90)
