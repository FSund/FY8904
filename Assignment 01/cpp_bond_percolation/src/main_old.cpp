#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <cassert>
#include <chrono>

#include "sites.hpp"

using namespace std;

// initialize random number generator
static std::random_device rd;     // only used once to initialise (seed) engine
static std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)

//vector<uvec> makeBonds(const uword m, const uword n)
//{
//    // makes bonds for square lattice

//    vector<uvec> bonds = vector<uvec>(2*m*n, zeros<uvec>(2));
//    uword N = m*n;

//    for (uword i = 0; i < m; i++) // vertical index
//    {
//        for (uword j = 0; j < n; j++) // horizontal index
//        {
//            uword idx = j + i*n; // linear index
//            uword right = (j + 1) % n + i*n;
//            uword down = (idx + n) % N;

//            bonds.at(2*idx)(0) = idx;
//            bonds.at(2*idx)(1) = right;
//            bonds.at(2*idx + 1)(0) = idx;
//            bonds.at(2*idx + 1)(1) = down;
//        }
//    }

//    return bonds;
//}

double findLogBinom_old(const uword n, const uword k)
{
    // calculates log(binom(n, k))
    double nsum = 0;
    double ksum = 0;
    double nksum = 0;

    for (uword i = 1; i <= n; i++)
    {
        nsum += log(i);
    }

    for (uword i = 1; i <= k; i++)
    {
        ksum += log(i);
    }

    for (uword i = 1; i <= (n-k); i++)
    {
        nksum += log(i);
    }

    return nsum - (ksum + nksum);
}

double findLogBinom(const uword n, const uword k)
{
    assert(n >= k);

    // calculates log(binom(n, k))
    double nsum = 0;
    double ksum = 0;
    double nksum = 0;

//    // split up loop in three parts
//    uword i = 1;

//    // first do (n-k)!
//    for (; i <= (n-k); i++)
//    {
//        nksum += log(i);
//    }

//    // then k!
//    ksum = nksum;
//    for (; i <= k; i++)
//    {
//        ksum += log(i);
//    }

//    // finally n!
//    nsum = ksum;
//    for (; i <= n; i++)
//    {
//        nsum += log(i);
//    }

    // optimized
    for (uword i = 1; i <= n; i++)
    {
        nsum += log(i);
        if (i == (n-k))
        {
            nksum = nsum;
        }
        if (i == k)
        {
            ksum = nsum;
        }
    }

//    // split up loop in three parts
//    uword i = 1;
//    if ((n-k) >= k)
//    {
//        // do the smallest first
//        for (; i <= k; i++)
//        {
//            ksum += log(i);
//        }

//        nksum = ksum;
//        for (; i <= (n-k); i++)
//        {
//            nksum += log(i);
//        }

//        nsum = nksum;
//    }
//    else // k > (n-k)
//    {
//        for (; i <= (n-k); i++)
//        {
//            nksum += log(i);
//        }

//        ksum = nksum;
//        for (; i <= k; i++)
//        {
//            ksum += log(i);
//        }

//        nsum = ksum;
//    }

//    // finally n!
//    for (; i <= n; i++)
//    {
//        nsum += log(i);
//    }

    return nsum - (ksum + nksum);
}

class LogBinomCoeffGenerator
{
public:
    LogBinomCoeffGenerator(const uword max):
        m_logSum(tabulateLogSum(max))
    {}

    double operator()(const uword n, const uword k) const
    {
        // returns log of binomial coefficient (log(n choose k))
        assert(n < m_logSum.n_elem);
        assert(n >= k);

        double coeff = m_logSum(n) - m_logSum(k) - m_logSum(n - k);
        return coeff;
    }

private:
    // logSum(n) is sum_{i=1}^n log i
    const vec m_logSum;

    static vec tabulateLogSum(const uword max)
    {
        cout << "LogBinomCoeffGenerator() tabulating log sums from 1 to " << max;
        cout.flush();
        vec logSum(max + 1);
        for (uword i = 1; i <= max; i++)
        {
            logSum(i) = logSum(i-1) + log(i);
        }
        cout << " ... Done" << endl;

        return logSum;
    }
};

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

//vector<uvec> shuffleBonds(const vector<uvec>& input)
//{
//    // strategy: convert to matrix, then shuffle using arma::shuffle
//    // this is probably slow, but we'll check that later

//    umat bonds = zeros<umat>(2, input.size());
//    for (uword i = 0; i < input.size(); i++)
//    {
//        bonds.col(i) = input.at(i);
//    }

//    bonds = arma::shuffle(bonds, 1);

//    auto output = vector<uvec>(input.size(), zeros<uvec>(2));
//    for (uword i = 0; i < input.size(); i++)
//    {
//        output.at(i) = bonds.col(i);
//    }

//    return output;
//}

vector<uvec> shuffledBonds(const vector<uvec>& input)
{
//    auto bonds = vector<uvec>(input.size(), zeros<uvec>(2));
    auto bonds = input; // copy
    for (uword i = 0; i < (input.size() - 1); i++)
    {
        std::uniform_int_distribution<uword> uniform(i + 1, input.size() - 1);
        uword j = uniform(rng);
//        cout << "j = " << j << endl;

        // swap element i and j
        auto rowj = bonds.at(j);
        bonds.at(i) = bonds.at(j);
        bonds.at(j) = rowj;
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

int v1()
{
    uword its = 1000;
    uword L = 100;
    uword N = L*L;
    uword nBonds = 2*N;

    if (L > 300)
    {
        cout << "WARNING: Will use a lot of memory" << endl;
    }

    mat P = zeros<mat>(nBonds, its);
    mat s = zeros<mat>(nBonds, its);
    vec p = zeros<vec>(nBonds);

    cout << "Size of matrix is " << 8*nBonds*its << " bytes == " << 8*nBonds*its/(1024*1024) << " Mb" << endl;

    for (uword it = 0; it < its; it++)
    {
        umat bonds = makeBonds(L, L);
        shuffleBonds(bonds);

        Sites sites(L, L);
        for (uword i = 0; i < bonds.n_cols; i++)
        {
            sites.activateMat(bonds.col(i));
            P(i, it) = sites.giantComponent();
            s(i, it) = sites.averageSquaredSize();

            // only need to fill in p the first time, since it's the same every iteration
            if (it == 0)
            {
                p(i) = (i + 1)/double(nBonds);
            }
        }
    }

    vec P_mean = arma::mean(P, 1);
    vec s_mean = arma::mean(s, 1);

    P_mean.save("giant.csv", arma::csv_ascii);
    s_mean.save("averageSquaredSize.csv", arma::csv_ascii);
    p.save("perc.csv", arma::csv_ascii);

    cout << "Done" << endl;

    return 0;
}

class Results
{
public:
    Results(const uword N, const uword nSamples, const uword nIterations, const uword nBonds):
        m_P(nSamples, nIterations),
        m_Psquared(nSamples, nIterations),
        m_s(nSamples, nIterations),
        m_p(arma::linspace<vec>(0, 1, nSamples)),
        m_N(N),
        m_nSamples(nSamples),
        m_nBonds(nBonds),
        m_idx(0),
        m_currentIteration(0)
    {}

    void sample(const Sites& sites, const uword iteration, const uword nActivatedBonds)
    {
        if (iteration != m_currentIteration)
        {
            // restart counter at beginning of each sample cycle
            m_idx = 0;
            m_currentIteration = iteration;
        }
        const double p = nActivatedBonds/double(m_nBonds); // fraction of activated bonds
        if (p*m_nSamples > m_idx)
        {
            m_P(m_idx, iteration) = sites.giantComponent();
            m_Psquared(m_idx, iteration) = sites.giantComponentSquared();
            m_s(m_idx, iteration) = sites.averageSquaredSize();

            if (iteration == 0)
            {
                m_p(m_idx) = p;
            }

            m_idx++;
        }
    }

    void save(const string& folder)
    {
        std::error_code ec;
        bool readyToSave = true;
        if (!std::filesystem::exists(folder))
        {
            readyToSave = false;
            if (!std::filesystem::create_directories(folder, ec)) // DOESN'T WORK WITH TRAILING SLASH
            {
                cout << "WARNING: could not create directory structure \"" << folder << "\"" << endl;
                readyToSave = false;
            }
            else
            {
                readyToSave = true;
            }
        }
        if (readyToSave)
        {
            vec P_mean = arma::mean(m_P, 1);
            vec Psquared_mean = arma::mean(m_Psquared, 1);
            vec s_mean = arma::mean(m_s, 1);

            vec susceptibility = m_N*sqrt(Psquared_mean - pow(P_mean, 2));

            P_mean.save(folder + "/giant.csv", arma::csv_ascii);
            s_mean.save(folder + "/averageSquaredSize.csv", arma::csv_ascii);
            susceptibility.save(folder + "/susceptibility.csv", arma::csv_ascii);

            m_p.save(folder + "/probability.csv", arma::csv_ascii);

            cout << "Saved results to \"" << folder << "\"" << endl;
        }
    }

    double getTotalMemory()
    {
        double size = 0;
        size += sizeof(double)*m_P.n_cols*m_P.n_rows;
        size += sizeof(double)*m_Psquared.n_cols*m_Psquared.n_rows;
        size += sizeof(double)*m_s.n_cols*m_s.n_rows;

        return size;
    }

    string getTotalMemoryMbString()
    {
        std::stringstream ss;
        ss << getTotalMemory()/(1024*1024.0) << " Mb";
        return  ss.str();
    }

//    vec averageGiant() const { return arma::mean(m_P, 1); }
//    vec averageGiantSquared() const { return arma::mean(m_Psquared, 1); }
//    vec averageSize() const { return arma::mean(m_s, 1); }
//    const vec& prob() const { return m_p; }

private:
    mat m_P;
    mat m_Psquared;
    mat m_s;
    vec m_p;

    uword m_N;
    uword m_nSamples;
    uword m_nBonds;

    uword m_idx;
    uword m_currentIteration;
};

int v2()
{
    // with smarter sampling

    vector<uword> sizes = {200};
    uword its = 1000;

    for (uword L : sizes)
    {
        uword N = L*L;
        uword nBonds = 2*N;
        uword nSamples = 1e4;

        Results results(N, nSamples, its, nBonds);

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

class CumulativeResults
{
public:
    CumulativeResults(const uword nBonds, const uword N):
        m_P(nBonds),
        m_Psquared(nBonds),
        m_s(nBonds),
        m_p(nBonds),
        m_N(N),
        m_nBonds(nBonds),
        m_currentIteration(0),
        m_idx(0)
    {}

    void sample(const Sites& sites, const uword iteration, const uword nActivatedBonds)
    {
        if (iteration > m_currentIteration)
        {
            // restart counter at beginning of each sample cycle
            m_idx = 0;
            m_currentIteration = iteration;

            // take sample
            m_giant(m_P);
            m_giantSquared(m_Psquared);
            m_averageSize(m_s);
        }

        const double p = nActivatedBonds/double(m_nBonds); // fraction of activated bonds

        m_P(m_idx) = sites.giantComponent();
        m_Psquared(m_idx) = sites.giantComponentSquared();
        m_s(m_idx) = sites.averageSquaredSize();

        if (iteration == 0)
        {
            m_p(m_idx) = p;
        }

        m_idx++;
    }

    void save(const string& folder)
    {
        std::error_code ec;
        bool readyToSave = true;
        if (!std::filesystem::exists(folder))
        {
            readyToSave = false;
            if (!std::filesystem::create_directories(folder, ec)) // DOESN'T WORK WITH TRAILING SLASH
            {
                cout << "WARNING: could not create directory structure \"" << folder << "\"" << endl;
                readyToSave = false;
            }
            else
            {
                readyToSave = true;
            }
        }
        if (readyToSave)
        {
            vec P_mean = m_giant.mean();
            vec Psquared_mean = m_giantSquared.mean();
            vec s_mean = m_averageSize.mean();

            vec susceptibility = m_N*sqrt(Psquared_mean - pow(P_mean, 2));

            P_mean.save(folder + "/giant.csv", arma::csv_ascii);
            s_mean.save(folder + "/averageSquaredSize.csv", arma::csv_ascii);
            susceptibility.save(folder + "/susceptibility.csv", arma::csv_ascii);
            m_p.save(folder + "/probability.csv", arma::csv_ascii);

            cout << "Saved results to \"" << folder << "\"" << endl;
        }
    }

    vec averageGiant() const { return arma::mean(m_P, 1); }
    vec averageGiantSquared() const { return arma::mean(m_Psquared, 1); }
    vec averageSize() const { return arma::mean(m_s, 1); }
    const vec& prob() const { return m_p; }

private:
    vec m_P;
    vec m_Psquared;
    vec m_s;

    vec m_p; // probability

    running_stat_vec<vec> m_giant;
    running_stat_vec<vec> m_giantSquared;
    running_stat_vec<vec> m_averageSize;

    uword m_N; // size of lattice
    uword m_nBonds;
    uword m_currentIteration;
    uword m_idx;
};

int v3()
{
    // even smarter sampling, using arma::running_stat_vec

    vector<uword> sizes = {100, 250};
    uword its = 1000;

    for (uword L : sizes)
    {
        uword N = L*L;
        uword nBonds = 2*N;

        CumulativeResults results(nBonds, N);

        cout << "Size of matrix is " << 8*nBonds << " bytes == " << 8*nBonds/double(1024) << " Kb == " << 8*nBonds/double(1024*1024) << " Mb" << endl;

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
        ss << "L" << L << "_" << its << "iterations_cumulative";
        const string folder = "../cpp_results/" + ss.str();
        results.save(folder);
    }

    cout << "Done" << endl;

    return 0;
}

class ConvolutionResults
{
public:
    ConvolutionResults(const uword nBonds, const uword nIterations, const uword N, const LogBinomCoeffGenerator& generator):
        m_P(nBonds, nIterations),
        m_Psquared(nBonds, nIterations),
        m_s(nBonds, nIterations),
        m_p(nBonds),
        m_logBinomCoeff(nBonds + 1),
        m_N(N),
        m_nBonds(nBonds),
        m_currentIteration(0),
        m_idx(0)
    {
        // tabulate log(n choose k) for all numbers of bonds using
        // LogBinomCoeffGenerator, which has tabulated values for all required
        // log(i) sums
        for (uword i = 0; i <= nBonds; i++)
        {
            m_logBinomCoeff(i) = generator(nBonds, i);
        }
    }

    void sample(const Sites& sites, const uword iteration, const uword nActivatedBonds)
    {
        if (iteration > m_currentIteration)
        {
            // restart counter at beginning of each sample cycle
            m_idx = 0;
            m_currentIteration = iteration;
        }

        const double p = nActivatedBonds/double(m_nBonds); // fraction of activated bonds

        m_P(m_idx, iteration) = sites.giantComponent();
        m_Psquared(m_idx, iteration) = sites.giantComponentSquared();
        m_s(m_idx, iteration) = sites.averageSquaredSize();

        if (iteration == 0)
        {
            m_p(m_idx) = p;
        }

        m_idx++;
    }

    vec convoluteResults(const mat& Q)
    {
        // strategy: first convolute results with a binomial distribution
        // then average over samples

        vec out(Q.n_rows); // memory not initialized
        const uword M = Q.n_rows; // number of probabilities (bonds in our case)
        const uword nSamples = Q.n_cols;

        cout << "Analyzing results";
        cout.flush();

        for (uword i = 0; i < M; i++) // loop over number of activated bonds
        {
            const double q = m_p(i);
            const double logq = log(q);
            const double logOneMinusQ = log(1-q);

            vec Qq(nSamples); // Q(q)

            for (uword j = 0; j < nSamples; j++) // loop over samples
            {
                const double Qn = Q(i, j);
                const double logQn = log(Qn);

                double sum = 0;
                for (uword n = 0; n < M; n++)
                {
                    const double logBinomCoeff = m_logBinomCoeff(n); // tabulated values for log(M choose n)
                    const double a = n*logq;
                    const double b = (M-n)*logOneMinusQ;
                    const double c = logQn;

                    // do the sum in eq. 4.1
                    const double logTerm = logBinomCoeff + a + b + c;
                    sum += exp(logTerm);
                }

                Qq(j) = sum;
            }

            out(i) = arma::mean(Qq);
        }

        cout << " ... Done!" << endl;

        return out;
    }

    vec calcResults_old(const mat& Q, const vec& pvec)
    {
        const uword nps = pvec.n_elem;
        const uword M = Q.n_rows; // number of bonds
        vec Qp(nps); // results

        // loop over all probabilities we want to get results for
        auto start = chrono::high_resolution_clock::now();
        for (uword i = 0; i < nps; i++)
        {
            const double p = pvec(i);
            double Qpi = 0;

            // loop over all Q's
            for (uword n = 0; n < M; n++)
            {
                const double Qn = Q(n, 0);
                const double a = m_logBinomCoeff(n); // m_logBinomCoeff(n) contains tabulated values for log(M choose n) for all n
                const double b = n*log(p);
                const double c = (M - n)*log(1 - p);
                const double d = log(Qn);

                const double logTerm = a + b + c + d;
                Qpi += exp(logTerm);
            }
            Qp(i) = Qpi;
        }
        auto stop = chrono::high_resolution_clock::now();
        cout << "Time spent calculating statistics " << chrono::duration_cast<chrono::seconds>(stop - start).count() << " seconds" << endl;

        return Qp;
    }

    vec calcResults(const mat& input, const vec& pvec)
    {
        // strategy: take the mean over all samples first, then convolution

        const vec Q = arma::mean(input, 1); // calculate
        const vec logQ = log(Q);

        const uword nps = pvec.n_elem;
        const uword M = Q.n_rows; // number of bonds
        vec Qp(nps); // results

        // loop over all probabilities we want to get results for
        cout << "Calculating statistics" << endl;
        auto start = chrono::high_resolution_clock::now();
        for (uword i = 0; i < nps; i++)
        {
            const double p = pvec(i);
            const double logp = log(p);
            const double logOneMinusP = log(1 - p);

            // calculate Q(p)
            double Qpi = 0;
            for (uword n = 0; n < M; n++) // loop over all Q's
            {
                const double a = m_logBinomCoeff(n); // m_logBinomCoeff(n) contains tabulated values for log(M choose n) for all n
                const double b = n*logp;
                const double c = (M - n)*logOneMinusP;
                const double d = logQ(n);

                const double logTerm = a + b + c + d;
                Qpi += exp(logTerm);
            }
            Qp(i) = Qpi;
        }
        auto stop = chrono::high_resolution_clock::now();
        cout << "Time spent calculating statistics " << chrono::duration_cast<chrono::seconds>(stop - start).count() << " seconds" << endl;

        return Qp;
    }

    void save(const string& folder)
    {
        std::error_code ec;
        bool readyToSave = true;
        if (!std::filesystem::exists(folder))
        {
            readyToSave = false;
            if (!std::filesystem::create_directories(folder, ec)) // DOESN'T WORK WITH TRAILING SLASH
            {
                cout << "WARNING: could not create directory structure \"" << folder << "\"" << endl;
                readyToSave = false;
            }
            else
            {
                readyToSave = true;
            }
        }
        if (readyToSave)
        {
            vec pvec = arma::linspace(0, 1, 1001);
            pvec = pvec(span(1, 1000)); // skip 0, since it gives nan when taking log

            vec giant = calcResults(m_P, pvec);
            vec giant_raw = arma::mean(m_P, 1);

            giant.save(folder + "/giant.csv", arma::csv_ascii);
//            giant_raw.save(folder + "/giant_rawmean.csv", arma::csv_ascii);
            pvec.save(folder + "/probability.csv", arma::csv_ascii);

//            s_mean.save(folder + "/averageSquaredSize.csv", arma::csv_ascii);
//            susceptibility.save(folder + "/susceptibility.csv", arma::csv_ascii);
//            m_p.save(folder + "/probability.csv", arma::csv_ascii);

            cout << "Saved results to \"" << folder << "\"" << endl;
        }
    }

    double getTotalMemory()
    {
        double size = 0;
        size += sizeof(double)*m_P.n_cols*m_P.n_rows;
        size += sizeof(double)*m_Psquared.n_cols*m_Psquared.n_rows;
        size += sizeof(double)*m_s.n_cols*m_s.n_rows;

        return size;
    }

    string getTotalMemoryMbString()
    {
        std::stringstream ss;
        ss << getTotalMemory()/(1024*1024.0) << " Mb";
        return  ss.str();
    }

private:
    mat m_P;
    mat m_Psquared;
    mat m_s;

    vec m_p; // probability

    // table of log(n choose k) for n = nBonds and k in [0, nBonds]
    // m_logBinomCoeff(i) contains log(nBonds choose i)
    vec m_logBinomCoeff;

    uword m_N; // size of lattice
    uword m_nBonds;
    uword m_currentIteration;
    uword m_idx;
};

int v4()
{
    // convolution

//    const vector<uword> sizes = {100, 200, 500, 1000};
    const vector<uword> sizes = {100, 200, 500};
    const uword its = 10; // lim

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

        ConvolutionResults results(nBonds, its, N, gen);

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

int tabulateBinomCoeffs()
{
    const vector<uword> sizes = {100, 200, 300, 400, 500, 700, 900, 1000};
    const uword max = 2*pow(sizes.at(sizes.size() - 1), 2);
    LogBinomCoeffGenerator gen(max);

    cout << gen(10, 5) << endl;
    cout << gen(10, 6) << endl;
    cout << gen(10, 10) << endl;
    cout << gen(10, 1) << endl;
    cout << gen(10, 0) << endl;

    for (uword L : sizes)
    {
        const uword N = L*L;
        const uword nBonds = 2*N;
        (void) nBonds;

        cout << gen(nBonds, 100) << endl;
        cout << gen(nBonds, nBonds) << endl;

//        vec logSums(nBonds + 1); // memory not initialized
//        logSums(0) = 0;
//        double test = 0;
//        for (uword i = 1; i <= nBonds; i++)
//        {
//            logSums(i) = logSums(i-1) + log(i);
//            test += log(i);
//        }
//        cout << "Done with L = " << L << endl;
//        cout << logSums(nBonds) - test << endl;


//        vec logBinomCoeffs(nBonds + 1);

//        // tabulate log binom for all sizes
//        auto start = chrono::high_resolution_clock::now();
//        for (uword i = 0; i <= nBonds; i++)
//        {
//            logBinomCoeffs(i) = findLogBinom(nBonds, i);
//        }
//        auto stop = chrono::high_resolution_clock::now();
//        cout << "Time spent tabulating " << nBonds << " binomial coefficients = " << chrono::duration_cast<chrono::seconds>(stop - start).count() << " seconds" << endl;

//        std::stringstream ss;
//        ss << "L" << L << "_logBinomCoeffs.bin";
//        string filename = "../cpp_results/" + ss.str();
//        logBinomCoeffs.save(filename);
    }

    return 0;
}

int main()
{
//    return v1();
//    return v2();
//    return v3();
    return v4();
//    return tabulateBinomCoeffs();

    {
        auto start = chrono::high_resolution_clock::now();
        double a = findLogBinom(1e6, 1e5);
        auto stop = chrono::high_resolution_clock::now();
        cout << "Time spent = " << chrono::duration_cast<chrono::microseconds>(stop - start).count() << endl;
        cout << a << endl;
    }
    {
        auto start = chrono::high_resolution_clock::now();
        double a = findLogBinom_old(1e6, 1e5);
        auto stop = chrono::high_resolution_clock::now();
        cout << "Time spent = " << chrono::duration_cast<chrono::microseconds>(stop - start).count() << endl;
        cout << a << endl;
    }
    return 0;

    cout << endl;
    cout << findLogBinom_old(10, 6) << endl;
    cout << findLogBinom(10, 6) << endl;
    return 0;


    cout << exp(findLogBinom(10, 5)) << endl;

    cout << findLogBinom(1e4, 5) << endl;
    cout << findLogBinom(1e5, 5) << endl;
    cout << findLogBinom(1e6, 5) << endl;
    cout << findLogBinom(1e6, 1e6) << endl;


    return 0;
}
