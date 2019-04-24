#pragma once

#include <iostream>
#include <cassert>
#include <random>
#include <armadillo>

using namespace std;
using namespace arma;

// initialize random number generator
// random_device seems to produce the same number sequence every time -- this is allowed by the standard (if no non-deterministic source of random numbers (e.g. a hardware device) is available), but not very useful for our application
static std::random_device rd;     // only used once to initialise (seed) engine
//static std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
static std::mt19937_64 rng(1337); // test 64-bit version

void shuffleBonds(umat& bonds)
{
    for (uword i = 0; i < (bonds.n_cols - 1); i++)
    {
        std::uniform_int_distribution<uword> uniform(i + 1, bonds.n_cols - 1);
        uword j = uniform(rng);

        bonds.swap_cols(i, j);
    }
}

umat makeRectangularLatticeBonds(const uword m, const uword n)
{
    // make bonds for rectangular lattice with dimensions (m, n)

    // allocate matrix with size (2, nBonds)
    // memory is not initialized
    umat bonds = umat(2, 2*m*n);

    for (uword i = 0; i < m; i++) // vertical index
    {
        for (uword j = 0; j < n; j++) // horizontal index
        {
            uword idx = j + i*n; // linear index
            uword right = (j + 1) % n + i*n; // periodic boundary conditions
            uword down = (idx + n) % (m*n);

            bonds(0, 2*idx) = idx;
            bonds(1, 2*idx) = right;
            bonds(0, 2*idx + 1) = idx;
            bonds(1, 2*idx + 1) = down;
        }
    }

    return bonds;
}

umat makeTriangularLatticeBonds(const uword m, const uword n)
{
    if (m < 1) throw runtime_error("n must be >= 1");
    if (n < 1) throw runtime_error("n must be >= 1");

    umat bonds(2, 3*m*n); // memory not initialized
    for (uword i = 0; i < m; i++) // vertical index
    {
        for (uword j = 0; j < n; j++) // horizontal index
        {
            uword right = (j + 1)%n + i*n;
            uword downRight = (i + 1)%m*n + j;

            // take modulo of j+n-1 instead of j-1, to avoid modulo of negative
            // number (which will happen at [i=0, j=0]), since this behaves
            // differently than Python
            uword downLeft = (i + 1)%m*n + ((sword(j + n) - 1))%n;

            uword idx = j + i*n;  // linear index
            bonds(0, 3*idx) = idx;
            bonds(1, 3*idx) = right;
            bonds(0, 3*idx + 1) = idx;
            bonds(1, 3*idx + 1) = downRight;
            bonds(0, 3*idx + 2) = idx;
            bonds(1, 3*idx + 2) = downLeft;
        }
    }

    return bonds;
}

umat makeHoneycombBonds(const uword m, const uword n)
{
    if (m < 1) throw runtime_error("n must be >= 1");
    if (n % 4 != 0) throw runtime_error("n must be divisible by 4");

    umat bonds(2, 3*m*n/2); // memory not initialized
    uword k = 0; // keep track of position in bonds matrix
    for (uword i = 0; i < m; i++) // vertical index
    {
        for (uword j = 0; j < n; j += 4) // horizontal index
        {
            // 6 bonds per "row" of 4 nodes
            uword idx = j + i*n;  // linear index
            uword node0 = idx;
            uword node1 = idx + 1;
            uword node2 = idx + 2;
            uword node3 = idx + 3;
            uword node;

            // node 0
            node = (i + m - 1)%m*n + (j + n - 1)%n;
            bonds(0, k) = node0;
            bonds(1, k) = node;
            k++;
            bonds(0, k) = node0;
            bonds(1, k) = node1;
            k++;

            // node 1
            node = (i + m - 1)%m*n + (j + 1 + 1)%n;
            bonds(0, k) = node1;
            bonds(1, k) = node;
            k++;
            bonds(0, k) = node1;
            bonds(1, k) = node2;
            k++;

            // node 2
            bonds(0, k) = node2;
            bonds(1, k) = node3;
            k++;

            // node 3
            node = i*n + (j + 3 + 1)%n;
            bonds(0, k) = node3;
            bonds(1, k) = node;
            k++;

            assert(k%6 == 0);
            if (k%6 != 0) throw runtime_error("k%6 != 0");
        }
    }

    return bonds;
}

void compareMatrices(const umat& check, const umat& bonds)
{
    for (uword i = 0; i < bonds.n_rows; i++)
    {
        for (uword j = 0; j < bonds.n_cols; j++)
        {
            assert(bonds(i, j) == check(i, j));
            if (bonds(i, j) != check(i, j)) throw runtime_error("Test failed");
        }
    }
}

void testHoneyComb()
{
    { // 1x4 matrix
        umat check;
        check << 0 << 0 << 1 << 1 << 2 << 3 << endr
              << 3 << 1 << 2 << 2 << 3 << 0 << endr;
        umat bonds = makeHoneycombBonds(1, 4);
        compareMatrices(check, bonds);
    }
    { // 2x4 matrix
        umat check;
        check << 0 << 0 << 1 << 1 << 2 << 3 << 4 << 4 << 5 << 5 << 6 << 7 << endr
              << 7 << 1 << 6 << 2 << 3 << 0 << 3 << 5 << 2 << 6 << 7 << 4 << endr;
        umat bonds = makeHoneycombBonds(2, 4);
        compareMatrices(check, bonds);
    }
    { // 1x8 matrix
        umat check;
        check << 0 << 0 << 1 << 1 << 2 << 3 << 4 << 4 << 5 << 5 << 6 << 7 << endr
              << 7 << 1 << 2 << 2 << 3 << 4 << 3 << 5 << 6 << 6 << 7 << 0 << endr;
        umat bonds = makeHoneycombBonds(1, 8);
        compareMatrices(check, bonds);
    }
    { // 2x8 matrix
        umat check;
        check << 0  << 0 << 1  << 1 << 2 << 3 << 4  << 4 << 5  << 5 << 6 << 7 << 8 << 8 << 9 <<  9 << 10 << 11 << 12 << 12 << 13 << 13 << 14 << 15 << endr
              << 15 << 1 << 10 << 2 << 3 << 4 << 11 << 5 << 14 << 6 << 7 << 0 << 7 << 9 << 2 << 10 << 11 << 12 <<  3 << 13 <<  6 << 14 << 15 << 8  << endr;
        umat bonds = makeHoneycombBonds(2, 8);
        compareMatrices(check, bonds);
    }
}
