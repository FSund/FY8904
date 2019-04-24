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
from latticemaker import shuffleList

## task 3.3
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
sites = [-1 for i in range(N)]
print("sites:")
print(sites)


# TODO: not sure how this function works if we end up with repetitions of the else clause
# consider making a class with self.sites and self.findroot(i)
def findroot(sites, i, debug=False):
    if sites[i] < 0:
        # if sites[i] is negative, this is a root node with size sites[i]
        # then return the index of the root node
        if debug: print("HEI 1")
        return i
    else:
        # this is not a root node, but a node belonging to a cluster with root
        # node sites[i]
        # call findroot again, update sites[i] to point to the root node, and
        # return the root node
        if debug: print("HEI 2")
        if debug: print("sites[{}] before".format(i)); print(sites[i])
        sites[i] = findroot(sites, sites[i])
        if debug: print("sites[{}] after".format(i)); print(sites[i])
        return sites[i]


def activate(sites, bonds, n=None, debug=False):
    if n is None:
        n = len(bonds)

    assert n <= len(bonds), "n must be < len(bonds)"

    for i in range(n):
        bond = bonds[i]
        node1 = bond[0]
        node2 = bond[1]
        if debug: print("node1: "); print(node1)
        if debug: print("node2: "); print(node2)
        r1 = findroot(sites, node1)
        r2 = findroot(sites, node2)
        if debug: print("r1: "); print(r1)
        if debug: print("r2: "); print(r2)
        # print("sites[r1]: "); print(sites[r1])
        # print("sites[r2]: "); print(sites[r2])
        if r1 == r2:
            # if they belong to the same cluster already, we don't do anything
            continue
        if abs(sites[r1]) >= abs(sites[r2]):  # check cluster size
            # if cluster 1 is bigger than cluster 2
            if debug: print("FIRST")
            sites[r1] += sites[r2]  # update size of cluster 1
            sites[node2] = r1  # link node2 to root of cluster 1
        else:
            # if cluster 2 is bigger than (or equal to) cluster 1
            if debug: print("SECOND")
            sites[r2] += sites[r1]  # update size of cluster 2 (both numbers should be negative)
            sites[node1] = r2  # link node1 to root of cluster 2
            # print("sites[r1]: "); print(sites[r1])
            # print("sites[r2]: "); print(sites[r2])
        if debug: print("sites:"); print(sites)

    return sites


def testActivation():
    from latticemaker import makeSquareLattice
    print("Test that reversing order of activation gives same result")
    bonds = [[0, 1], [1, 2]]
    sites = [-1 for i in range(9)]
    sites = activate(sites, bonds)

    check = [-1 for i in range(9)]
    check = activate(sites, [bonds[1], bonds[0]])  # reversed order
    fail = False
    for site, c in zip(sites, check):
        if site != c:
            print("ERROR")
            fail = True
    if not fail:
        print("TEST SUCCESS")

    print("Test making multiple clusters")
    bonds = [[0, 1], [3, 4]]
    sites = [-1 for i in range(9)]
    sites = activate(sites, bonds)
    check = [-2, 0, -1, -2, 3, -1, -1, -1, -1]
    fail = False
    for s, c in zip(sites, check):
        if s != c:
            print("ERROR")
            fail = True
    if not fail:
        print("TEST SUCCESS")
    else:
        print("Sites are")
        print(sites)
        print("Should be")
        print(check)

    print("Test appending to cluster")
    bonds = [[0, 1], [3, 4], [4, 5]]
    sites = [-1 for i in range(9)]
    sites = activate(sites, bonds)
    check = [-2, 0, -1, -3, 3, 3, -1, -1, -1]
    fail = False
    for s, c in zip(sites, check):
        if s != c:
            print("ERROR")
            fail = True
    if not fail:
        print("TEST SUCCESS")
    else:
        print("Sites are")
        print(sites)
        print("Should be")
        print(check)

    print("Test combining clusters (of same size)")
    bonds = [[0, 1], [3, 4], [0, 3]]
    sites = [-1 for i in range(9)]
    sites = activate(sites, bonds)
    check = [-2, 0, -1, -1, -1, -1, -1, -1, -1]  # after first activation
    check = [-2, 0, -1, -2, 3, -1, -1, -1, -1]  # after second activation
    check = [-4, 0, -1, 0, 3, -1, -1, -1, -1]  # after third activation (fails)
    fail = False
    for s, c in zip(sites, check):
        if s != c:
            print("ERROR")
            fail = True
    if not fail:
        print("TEST SUCCESS")
    else:
        print("Sites are")
        print(sites)
        print("Should be")
        print(check)

    print("Test combining clusters (of different size)")
    bonds = [[0, 1], [1, 2], [3, 4], [0, 3]]
    sites = [-1 for i in range(9)]
    sites = activate(sites, bonds)
    check = [-2, 0, -1, -1, -1, -1, -1, -1, -1]  # after first activation
    check = [-3, 0, 0, -1, -1, -1, -1, -1, -1]  # after second activation
    check = [-3, 0, 0, -2, 3, -1, -1, -1, -1]  # after third activation
    check = [-5, 0, 0, 0, 3, -1, -1, -1, -1]  # after fourth activation (fails)
    fail = False
    for s, c in zip(sites, check):
        if s != c:
            print("ERROR")
            fail = True
    if not fail:
        print("TEST SUCCESS")
    else:
        print("Sites are")
        print(sites)
        print("Should be")
        print(check)


testActivation()

bonds = [[1, 2], [2,0], [2,5], [4,7], [8,6], [8,2]]  # from slides
# bonds = [[0,1]]
# bonds = [[0,1], [1,2]]
# bonds = [[0,1], [2,1], [3,4], [0,3]]  # this fails at final step, combining two clusters
sites = activate(sites, bonds)
print("sites after activations:")
print(sites)