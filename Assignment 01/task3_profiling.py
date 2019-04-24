from sites import Sites
from latticemaker import makeSquareLattice, shuffleList
import numpy as np

for L in [100, 200]:
    N = L*L
    nBonds = 2*N

    its = 100  # number of iterations
    P = np.zeros([nBonds, its])
    P2 = np.zeros([nBonds, its])  # P_inf**2
    size = np.zeros([nBonds, its])
    s = np.zeros([nBonds, its])
    p = np.zeros(nBonds)

    for j in range(its):
        sites = Sites(L, L)
        bonds = makeSquareLattice(L, L)
        bonds = shuffleList(bonds)

        for i in range(nBonds):
            sites.activate([bonds[i]])
            P[i, j] = sites.giantComponent
            P2[i, j] = pow(sites.giantComponent/N, 2)
            s[i, j] = sites.averageSquaredSize
            size[i, j] = sites.sizeOfLargestCluster
            p[i] = i/nBonds
            # p[i] = (N - np.sum(sites.sites == -1))/N

    P = np.mean(P, axis=1)
    P2 = np.mean(P2, axis=1)
    s = np.mean(s, axis=1)
    size = np.mean(size, axis=1)
