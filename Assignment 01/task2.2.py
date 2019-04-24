from latticemaker import makeSquareLattice, makeTriangularLattice, makeHoneycombLattice

if __name__ == "__main__":
    bonds = makeTriangularLattice(2, 2)
    [print("{} {}".format(a, b)) for a, b in bonds]
