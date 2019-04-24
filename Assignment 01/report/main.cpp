#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <cassert>
#include <chrono>

#include "sites.hpp"
#include "results.hpp"
#include "logbinom.hpp"

using namespace std;

// initialize random number generator
static std::random_device rd;     // only used once to initialise (seed) engine
static std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)

umat makeBonds(const uword m, const uword n)
{
    // makes bonds for square lattice

    umat bonds = umat(2, 2*m*n); // memory is not initialized
    uword N = m*n;

    for (uword i = 0; i < m; i++) // vertical index
    {
        for (uword j = 0; j < n; j++) // horizontal index
        {
            uword idx = j + i*n; // linear index
            uword right = (j + 1) % n + i*n;
            uword down = (idx + n) % N;

            bonds(0, 2*idx) = idx;
            bonds(1, 2*idx) = right;
            bonds(0, 2*idx + 1) = idx;
            bonds(1, 2*idx + 1) = down;
        }
    }

    return bonds;
}

void shuffleBonds(umat& bonds)
{
    for (uword i = 0; i < (bonds.n_cols - 1); i++)
    {
        std::uniform_int_distribution<uword> uniform(i + 1, bonds.n_cols - 1);
        uword j = uniform(rng);

        bonds.swap_cols(i, j);
    }
}

int v4()
{
    // convolution

//    const vector<uword> sizes = {100, 200, 500, 1000};
    const vector<uword> sizes = {100, 200, 500, 700};
    const uword its = 100; // lim

    const uword max = 2*sizes.at(sizes.size() - 1)*sizes.at(sizes.size() - 1);
    LogBinomCoeffGenerator gen(max);

    for (uword L : sizes)
    {
        uword N = L*L;
        uword nBonds = 2*N;

        cout << "L = " << L << endl;

        double memoryInKb = 3*sizeof(double)*nBonds*its/(1024*1024.0);
        if (memoryInKb > 5000)
        {
            cout << "Memory usage will exceed 5 Gb, stopping execution" << endl;
            return 1;
        }

        Results results(nBonds, gen);

        cout << "Approximate memory usage: " << results.getTotalMemoryMbString() << endl;

        for (uword it = 0; it < its; it++)
        {
            umat bonds = makeBonds(L, L);
            shuffleBonds(bonds);

            Sites sites(L, L);
            for (uword i = 0; i < bonds.n_cols; i++)
            {
                sites.activate(bonds.col(i));

                results.sample(sites, it, i+1);
            }
        }

        // save results to file
        std::stringstream ss;
        ss << "L" << L << "_" << its << "iterations";
        string folder = "../cpp_results/" + ss.str();
        results.save(folder);
    }

    cout << "Done" << endl;

    return 0;
}

int main()
{
    return v4();
}
