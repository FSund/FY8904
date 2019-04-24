from latticemaker import makeSquareLattice, makeTriangularLattice, makeHoneycombLattice

bonds = makeSquareLattice(3, 3)
nodes = set([node for node, _ in bonds])  # use set() to remove duplicates
nodes = list(nodes)  # convert to list again
nodes.sort()  # sort list

with open("squareLattice.csv", "w") as f:
    f.write("{}, {}\n".format(len(nodes), len(bonds)))
    for bond in bonds:
        f.write("{}, {}\n".format(bond[0], bond[1]))

# bonds = makeTriangularLattice(2, 2)
# [print("{} {}".format(a, b)) for a, b in bonds]
