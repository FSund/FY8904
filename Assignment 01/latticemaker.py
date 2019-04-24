import sys
import random
random.seed(2)


def makeSquareLattice(m=3, n=3, zero_indexing=True):
    '''
    Find list of pairs of nodes connected by a bond in a square lattice of
    lateral size L and periodic boundary conditions, using double for-loops and
    0-based indexing.

    Parameters
    ----------
    m : int, optional
        vertical size of lattice
    n : int, optional
        horizontal size of lattice

    Returns
    -------
    list
        Nested list of node links [[node1, node2], [node1, node3], ...]
    '''
    add = 0
    if not zero_indexing:
        add = 1

    N = m*n  # number of nodes
    bonds = []
    for i in range(m):  # vertical index
        for j in range(n):  # horizontal index
            idx = j + i*n  # linear index
            right = (j + 1) % n + i*n
            down = (idx + n) % N
            bonds.append([idx + add, right + add])  # add 1 if 1-based indexing
            bonds.append([idx + add, down + add])  # add 1 if 1-based indexing

    return bonds


def makeTriangularLattice(m=4, n=4):
    if m % 2 != 0:
        raise RuntimeError("m must be an even number.")
    if n < 2:
        raise RuntimeError("n must >2.")
    # N = L*L
    bonds = []
    for i in range(m):  # vertical
        for j in range(n):  # horizontal
            right = (j + 1) % n + i*n

            down = ((i + 1) % m)*n
            if i % 2 == 0:  # even rows (0, 2, 4, ...)
                downRight = down + j % n
                downLeft = down + (j - 1) % n
            else:  # odd rows
                downRight = down + (j + 1) % n
                downLeft = down + j % n

            idx = j + i*n  # linear index
            bonds.append([idx, right])
            bonds.append([idx, downRight])
            bonds.append([idx, downLeft])

    return bonds


def makeHoneycombLattice(m=2, n=4):
    '''
    Strategy: generate lattice from pairs of nodes/unit cells
    '''
    if m % 2 != 0 or m < 2:
        raise RuntimeError("m must be an even number >=2.")
    if n % 4 != 0 or n < 4:
        raise RuntimeError("n must be divisible by 4 and >=4.")

    return []

    bonds = []
    for i in range(m):  # vertical
        for j in range(n/4):  # horizontal
            idx = i*n + j  # linear index

            # should have 6 bonds in total
            ## first node
            # upper left bond
            node = (((j - 1)%m)*n - 1)%n
            bonds.append([idx, node])
            # second node

            # third node

            # fourth node


def shuffleList(thelist):
    shuffled = thelist.copy()  # only works for lists of non-objects
    if True:
        random.shuffle(shuffled)
    else:
        N = len(shuffled)
        for i in range(N-1):
            # draw random int in range [i+1, M-1] (randint includes endpoints)
            # use M-1 instead of M since we are zero-indexed
            r = random.randint(i + 1, N - 1)
            shuffled[i], shuffled[r] = shuffled[r], shuffled[i]  # swap

    return shuffled


def testShuffle():
    print("Testing shuffle")

    # check that all bonds links the same sites
    bonds = makeSquareLattice(4, 4)
    bonds = shuffleList(bonds)
    bonds.sort()
    check = makeSquareLattice(4, 4)
    check.sort()
    for i in range(len(bonds)):
        if bonds[i][0] != check[i][0] or bonds[i][1] != check[i][1]:
            print(" - Test 1 FAILED")
            break
    print(" - Test 1 success")

    # check that shuffle worked
    bonds = makeSquareLattice(4, 4)
    bonds = shuffleList(bonds)
    check = makeSquareLattice(4, 4)
    shuffled = 0
    for i in range(len(bonds)):
        if bonds[i][0] != check[i][0] or bonds[i][1] != check[i][1]:
            shuffled += 1

    if shuffled:
        print(" - Test 2 success ({} of {} shuffled)".format(shuffled, len(bonds)))
    else:
        print(" - Test 2 FAILED")

    # randomness of shuffle
    import numpy as np
    from math import isclose

    test = np.linspace(0, 1, 100000)
    test = shuffleList(test)
    print(np.mean(test))
    print(np.mean(test[0:25000]))
    print(np.mean(test[25000:50000]))
    print(np.mean(test[50000:75000]))
    print(np.mean(test[75000:]))
    assert isclose(np.mean(test[0:25000]), 0.5, abs_tol=0.05), "Test failed"
    assert isclose(np.mean(test[25000:50000]), 0.5, abs_tol=0.05), "Test failed"
    assert isclose(np.mean(test[50000:75000]), 0.5, abs_tol=0.05), "Test failed"
    assert isclose(np.mean(test[75000:]), 0.5, abs_tol=0.05), "Test failed"
    print(" - Test 3 success")


def testSquareLattice():
    ## test square lattice
    print("Testing square lattice")

    # 3x3 square
    bonds = makeSquareLattice(3, 3, False)
    check = [[1, 2], [1, 4], [2, 3], [2, 5], [3, 1], [3, 6], [4, 5], [4, 7], [5, 6], [5, 8], [6, 4], [6, 9], [7, 8], [7, 1], [8, 9], [8, 2], [9, 7], [9, 3]]
    for i in range(len(bonds)):
        if bonds[i][0] != check[i][0] or bonds[i][1] != check[i][1]:
            print(" - Test 1 FAILED")
            break
    print(" - Test 1 success")

    # zero-indexed 3x3 square
    bonds = makeSquareLattice(3, 3, True)
    check = [[1, 2], [1, 4], [2, 3], [2, 5], [3, 1], [3, 6], [4, 5], [4, 7], [5, 6], [5, 8], [6, 4], [6, 9], [7, 8], [7, 1], [8, 9], [8, 2], [9, 7], [9, 3]]
    for i in range(len(bonds)):
        if bonds[i][0] != (check[i][0] - 1) or bonds[i][1] != (check[i][1] - 1):
            print(" - Test 2 FAILED")
            break
    print(" - Test 2 success")

    # 2x3 rectangle
    bonds = makeSquareLattice(2, 3, False)  # remove last row
    check = [[1, 2], [1, 4], [2, 3], [2, 5], [3, 1], [3, 6], [4, 5], [4, 1], [5, 6], [5, 2], [6, 4], [6, 3]]
    for i in range(len(bonds)):
        if bonds[i][0] != check[i][0] or bonds[i][1] != check[i][1]:
            print(" - Test 3 FAILED")
            break
    print(" - Test 3 success")

    # zero-index 2x3 rectangle
    bonds = makeSquareLattice(2, 3, True)
    check = [[1, 2], [1, 4], [2, 3], [2, 5], [3, 1], [3, 6], [4, 5], [4, 1], [5, 6], [5, 2], [6, 4], [6, 3]]
    for i in range(len(bonds)):
        if bonds[i][0] != (check[i][0] - 1) or bonds[i][1] != (check[i][1] - 1):
            print(" - Test 4 FAILED")
            break
    print(" - Test 4 success")

    # 10x10 lattice
    bonds = makeSquareLattice(10, 10)
    assert bonds[-1][0] == 99
    assert bonds[-1][1] == 9  # final bond is down
    assert bonds[-2][0] == 99
    assert bonds[-2][1] == 90  # right


def testTriangularLattice():
    ## test triangular lattice
    print("Testing triangular lattice")
    # 2x2 square
    bonds = makeTriangularLattice(2, 2)
    check = [[0, 1], [0, 2], [0, 3], [1, 0], [1, 3], [1, 2], [2, 3], [2, 1], [2, 0], [3, 2], [3, 0], [3, 1]]
    for i in range(len(bonds)):
        if bonds[i][0] != check[i][0] or bonds[i][1] != check[i][1]:
            print(" - Test 1 FAILED")
            break
    print(" - Test 1 success")

    # 4x3 rectangle
    bonds = makeTriangularLattice(4, 3)
    check = [
        [0, 1], [0, 3], [0, 5],
        [1, 2], [1, 4], [1, 3],
        [2, 0], [2, 5], [2, 4],
        [3, 4], [3, 7], [3, 6],
        [4, 5], [4, 8], [4, 7],
        [5, 3], [5, 6], [5, 8],
        [6, 7], [6, 9], [6, 11],
        [7, 8], [7, 10], [7, 9],
        [8, 6], [8, 11], [8, 10],
        [9, 10], [9, 1], [9, 0],
        [10, 11], [10, 2], [10, 1],
        [11, 9], [11, 0], [11, 2]
    ]
    for i in range(len(bonds)):
        if bonds[i][0] != check[i][0] or bonds[i][1] != check[i][1]:
            print(" - Test 2 FAILED")
            break
    print(" - Test 2 success")


def testHoneycombLattice():
    # test 3
    print("Testing honeycomb lattice")
    bonds = makeHoneycombLattice()


def runTests():
    testSquareLattice()
    testTriangularLattice()
    testHoneycombLattice()
    testShuffle()


if __name__ == "__main__":
    runTests()
