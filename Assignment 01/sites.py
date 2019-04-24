## task 3.4
# sites contain the status of each node
# If node i is the root node of a cluster, then sites[i] contains the number
# of nodes in the cluster. If not, then sites[i] contains the index of the root
# node of the cluster it belongs to.
# Since we keep both sizes and indexes in the same array, we code sizes using
# negative numbers, and indexes using positive
# So the array will initially look like [-1, -1, ...], since all nodes are root
# nodes of clusters with size 1


class Sites(object):
    def __init__(self, m, n):
        super(Sites, self).__init__()
        self.m = m
        self.n = n
        self.sites = [-1 for i in range(m*n)]
        self.N = m*n  # number of sites (same as len(self.sites))

        # largest cluster
        self.sizeOfLargestCluster = -1
        self.largestCluster = -1

        # statistics
        self.giantComponent = 0  # P_inf
        self.giantSquared = 0  # P_inf**2
        self.averageSizeSum = len(self.sites)  # average_s
        self.averageSquaredSize = 0  # <s>

    def _findRoot(self, i):
        # recursive function
        if self.sites[i] < 0:
            # if sites[i] is negative, this is a root node with size sites[i]
            # then return the index of the root node
            return i
        else:
            # this is not a root node, but a node belonging to a cluster with
            # root node sites[i]
            # call findroot again, update sites[i] to point to the root node,
            # and return the root node
            self.sites[i] = self._findRoot(self.sites[i])
            return self.sites[i]

    def activate(self, bonds):
        # check if bonds is nested list
        if not isinstance(bonds[0], list):
            # single bond version
            bonds = [bonds]

        n = len(bonds)

        assert n <= len(bonds), "n must be < len(bonds)"

        for bond in bonds:
            node1 = bond[0]
            node2 = bond[1]
            r1 = self._findRoot(node1)
            r2 = self._findRoot(node2)
            if r1 == r2:
                # if they belong to the same cluster already, we don't do anything
                pass
            else:
                # check cluster size
                # merge smallest cluster into largest
                # merge into r2 if they have the same size (in else clause)
                if abs(self.sites[r1]) > abs(self.sites[r2]):
                    self._mergeClusters(larger=r1, smaller=r2)
                else:
                    self._mergeClusters(larger=r2, smaller=r1)

    def _mergeClusters(self, larger, smaller):
        # subtract the square of the size of the two clusters that are going to
        # be merged from the average
        self.averageSizeSum -= (pow(self.sites[larger], 2) + pow(self.sites[smaller], 2))

        # update size and link root
        self.sites[larger] += self.sites[smaller]  # update size of larger cluster
        self.sites[smaller] = larger  # link root of smaller cluster to root of larger cluster

        # add the square of the size of the new size of the merged cluster to
        # the average
        self.averageSizeSum += pow(self.sites[larger], 2)

        self._checkLargestCluster(larger)

        if self.sizeOfLargestCluster == self.N:
            # avoid division by zero/divergence
            self.averageSquaredSize = 0
        else:
            self.averageSquaredSize = (self.averageSizeSum - pow(self.N*self.giantComponent, 2))/(self.N*(1 - self.giantComponent))

    def _checkLargestCluster(self, rootNode):
        if abs(self.sites[rootNode]) > self.sizeOfLargestCluster:
            self.largestCluster = rootNode
            self.sizeOfLargestCluster = abs(self.sites[rootNode])
            self.giantComponent = self.sizeOfLargestCluster/self.N
            self.giantSquared = pow(self.sizeOfLargestCluster/self.N, 2)

    def getBiggestCluster(self):
        nodes = []
        for node in range(len(self.sites)):
            root = self._findRoot(node)
            if root == self.largestCluster:
                nodes.append(node)

        return nodes

    def makeImage(self):
        import numpy as np
        nodes = self.getBiggestCluster()
        image = np.zeros([self.m, self.n], dtype=float)
        for node in nodes:
            i = node//self.n
            j = node%self.m
            image[i, j] = 1

        return image


def runTests():
    from latticemaker import makeSquareLattice, shuffleList
    import numpy as np

    print("Testing Sites()")

    ## some simple tests of a square 3x3 lattice
    sites = Sites(3, 3)

    sites.activate([0, 1])
    assert sites.sizeOfLargestCluster == 2, "Test failed"
    check = [1, -2, -1, -1, -1, -1, -1, -1, -1]
    assert (set(sites.sites) - set(check)) == set(), "Test failed"

    sites.activate([1, 2])
    assert sites.sizeOfLargestCluster == 3, "Test failed"
    check = [1, 2, -3, -1, -1, -1, -1, -1, -1]
    assert (set(sites.sites) - set(check)) == set(), "Test failed"

    sites.activate([3, 4])
    assert sites.sizeOfLargestCluster == 3, "Test failed"
    check = [1, 2, -3, 4, -2, -1, -1, -1, -1]
    assert (set(sites.sites) - set(check)) == set(), "Test failed"

    sites.activate([4, 5])
    assert sites.sizeOfLargestCluster == 3, "Test failed"
    check = [1, 2, -3, 4, 5, -3, -1, -1, -1]
    assert (set(sites.sites) - set(check)) == set(), "Test failed"

    sites.activate([0, 3])
    assert sites.sizeOfLargestCluster == 6, "Test failed"
    check = [1, 2, 5, 4, 5, -6, -1, -1, -1]
    assert (set(sites.sites) - set(check)) == set(), "Test failed"

    ## test merging two bigger clusters
    sites = Sites(3, 3)
    sites.activate([[0, 1], [1, 2]])  # size 3 cluster
    sites.activate([[3, 4], [4, 5], [4, 7]])  # size 4 cluster
    sites.activate([[0, 3]])
    assert sites.sizeOfLargestCluster == 7, "Test failed"

    sites2 = Sites(3, 3)
    sites2.activate([[0, 1], [1, 2]])  # size 3 cluster
    sites2.activate([[3, 4], [4, 5], [4, 7]])  # size 4 cluster
    sites2.activate([[1, 4]])
    assert sites2.sizeOfLargestCluster == 7, "Test failed"

    sites3 = Sites(3, 3)
    sites3.activate([[0, 1], [1, 2]])  # size 3 cluster
    sites3.activate([[3, 4], [4, 5], [4, 7]])  # size 4 cluster
    sites3.activate([[2, 5]])
    assert sites3.sizeOfLargestCluster == 7, "Test failed"

    # check that the order of merging doesn't matter
    for i in range(len(sites.sites)):
        assert sites.sites[i] == sites2.sites[i], "Test failed"
        assert sites2.sites[i] == sites3.sites[i], "Test failed"

    ## test merging across periodic boundaries
    sites = Sites(3, 3)
    sites.activate([[0, 1], [1, 2]])  # size 3 cluster
    sites.activate([[6, 7], [7, 8], [4, 7]])  # size 4 cluster
    sites.activate([[0, 6]])
    assert sites.sizeOfLargestCluster == 7, "Test failed"

    ## test larger lattice
    sites = Sites(10, 10)
    bonds = makeSquareLattice(10, 10)
    bonds = shuffleList(bonds)
    sites.activate(bonds[0:80])
    assert abs(np.min(sites.sites)) == sites.sizeOfLargestCluster, "Test failed"
    assert np.argmin(sites.sites) == sites.largestCluster, "Test failed"
    sites.activate(bonds[80:105])
    assert abs(np.min(sites.sites)) == sites.sizeOfLargestCluster, "Test failed"
    assert np.argmin(sites.sites) == sites.largestCluster, "Test failed"
    sites.activate(bonds[105:])
    assert abs(np.min(sites.sites)) == 100, "Test failed"
    assert abs(np.min(sites.sites)) == sites.sizeOfLargestCluster, "Test failed"
    assert np.argmin(sites.sites) == sites.largestCluster, "Test failed"

    ## test that the _findRoot works as expected
    sites = Sites(3, 3)
    sites.activate([[1, 0], [1, 2]])  # make a cluster
    sites.activate([[3, 4], [4, 5]])  # make another cluster
    sites.activate([4, 1])  # merge in a specific order, so upper left becomes the root node
    sites.activate([[6, 7], [7, 8]])  # make a third cluster
    sites.activate([1, 7])  # merge
    assert sites.sites[8] == 7, "Test failed"
    assert sites._findRoot(8) == 0, "Test failed"  # this should modify sites.sites[8]
    assert sites.sites[8] == 0, "Test failed"

    ## test larger cluster and many activations
    sites = Sites(100, 100)
    bonds = makeSquareLattice(100, 100)  # don't shuffle

    # sites.activate(bonds[::2][:9])  # activate 10 first nodes
    # assert sites.sizeOfLargestCluster == 10, "Test failed"

    sites.activate(bonds[::2][-10:-1])  # activate 10 last nodes

    # print(sites.sites[:100])
    print(sites.sites[-200:])

    print(bonds[-1])
    print(bonds[-2])
    print(bonds[:105])


    print(" -- All tests passed")


if __name__ == "__main__":
    runTests()
