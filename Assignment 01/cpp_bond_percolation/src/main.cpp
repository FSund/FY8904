#include <iostream>
#include <string>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <cassert>
#include <chrono>

#include "sites.hpp"
#include "results.hpp"
#include "logbinom.hpp"
#include "lattices.hpp"

using namespace std;

int run()
{
    const uvec sizes = {100, 200, 300, 500, 700, 1000};
    const uword its = 1000;

    const uword maxNbonds = 2*sizes.max()*sizes.max();
    LogBinomCoeffGenerator gen(maxNbonds);

    for (uword L : sizes)
    {
        uword N = L*L;
        uword nBonds = 2*N;

        Results results(nBonds, N, gen);
        for (uword it = 0; it < its; it++)
        {
            umat bonds = makeRectangularLatticeBonds(L, L);
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
        string folder = "../cpp_results_square/" + ss.str();
        results.save(folder, 1e4);
    }

    return 0;
}

int runTriangular()
{
    const uvec sizes = {100, 200, 300, 500, 700, 1000};
//    const uvec sizes = {100, 200, 300};
    const uword its = 1000;

    const uword maxNbonds = 3*sizes.max()*sizes.max();
    LogBinomCoeffGenerator gen(maxNbonds);

    for (uword L : sizes)
    {
        uword N = L*L; // number if sites
        uword nBonds = 3*N;

        Results results(nBonds, N, gen);
        for (uword it = 0; it < its; it++)
        {
            umat bonds = makeTriangularLatticeBonds(L, L);
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
        string folder = "../cpp_results_triangular/" + ss.str();
        results.save(folder, 1e4);
    }

    return 0;
}

int runHoneycomb()
{
    const uvec sizes = {100, 200, 300, 500, 700, 1000};
//    const uvec sizes = {100, 200, 300};
    const uword its = 1000;

    const uword maxNbonds = 3*sizes.max()*sizes.max()/2;
    LogBinomCoeffGenerator gen(maxNbonds);

    for (uword L : sizes)
    {
        uword N = L*L; // number if sites
        uword nBonds = 3*N/2;

        Results results(nBonds, N, gen);
        for (uword it = 0; it < its; it++)
        {
            umat bonds = makeHoneycombBonds(L, L);
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
        string folder = "../cpp_results_honeycomb/" + ss.str();
        results.save(folder, 1e4);
    }

    return 0;
}

int run_old()
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

        Results results(nBonds, N, gen);

//        cout << "Approximate memory usage: " << results.getTotalMemoryMbString() << endl;

        for (uword it = 0; it < its; it++)
        {
            umat bonds = makeRectangularLatticeBonds(L, L);
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

void saveImage(const umat& image, const uword L, const double p)
{
    std::stringstream ss;
    ss << "../cpp_results/";
    ss << "images_L" << L << "";
    string folder = ss.str();

    // create folders
    bool readyToSave = true;
    if (!std::filesystem::exists(folder))
    {
        readyToSave = false;
        if (!std::filesystem::create_directories(folder)) // DOESN'T WORK WITH TRAILING SLASH
        {
            cout << "WARNING: could not create directory structure \"" << folder << "\"" << endl;
            readyToSave = false;
        }
        else
        {
            cout << "Created directory structure \"" << folder << "\"" << endl;
            readyToSave = true;
        }
    }
    if (readyToSave)
    {
        ss << "/image_p" << std::fixed << setprecision(3) << p;
        ss << ".csv";
        string filename = ss.str();
        cout << filename << endl;

        image.save(filename, arma::csv_ascii);
    }
}

int makeImages()
{
    const uword L = 2000;
    Sites sites(L, L);
    umat bonds = makeRectangularLatticeBonds(L, L);
    shuffleBonds(bonds);
    const uword N = 2*L*L;

    vec percentages = {0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65};
    uword n0 = 0;
    uword n1 = 0;
    for (uword i = 0; i < percentages.size() - 1; i++)
    {
        n0 = uword(N*percentages(i));
        n1 = uword(N*percentages(i + 1));

        double p = n1/double(N); // actual percentage
        sites.activateMat(bonds.cols(n0, n1));
        umat image = sites.makeImage();
        saveImage(image, L, p);
    }

    return 0;
}

int main()
{
    testHoneyComb();

    run();
    runTriangular();
    runHoneycomb();



//    return makeImages();

//    umat bonds = makeHoneycombBonds(2, 4);
//    cout << bonds << endl;

//    bonds = makeTriangularLatticeBonds(3, 3);
//    cout << bonds << endl;

    return 0;
}
