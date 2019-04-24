#pragma once

#include <filesystem>
#include <armadillo>

#include "sites.hpp"
#include "logbinom.hpp"

using namespace std;
using namespace arma;

class Results
{
public:
    Results(const uword nBonds, const uword nSites,
            const LogBinomCoeffGenerator& generator):
        m_P(nBonds),
        m_Psquared(nBonds),
        m_s(nBonds),
        m_p(nBonds),
        m_logBinomCoeff(nBonds + 1),
        m_nSites(nSites),
        m_nBonds(nBonds),
        m_currentIteration(0),
        m_idx(0)
    {
        // tabulate log(n choose k) for all numbers of bonds
        for (uword i = 0; i <= nBonds; i++)
        {
            m_logBinomCoeff(i) = generator(nBonds, i);
        }
    }

    void sample(const Sites& sites, const uword iteration,
                const uword nActivatedBonds);
    void save(const string& folder, const uword nSamples=1e3);

private:
    vec m_P;
    vec m_Psquared;
    vec m_s;

    vec m_p; // probability

    running_stat_vec<vec> m_giant;
    running_stat_vec<vec> m_giantSquared;
    running_stat_vec<vec> m_averageSize;

    // table of log(n choose k) for n = nBonds and k in [0, nBonds]
    vec m_logBinomCoeff;

    uword m_nSites;
    uword m_nBonds;
    uword m_currentIteration;
    uword m_idx;

    vec calcResults(const vec& Q, const vec& pvec);
};

//double getTotalMemory()
//{
//    double size = 0;
//    size += sizeof(double)*m_P.n_cols*m_P.n_rows;
//    size += sizeof(double)*m_Psquared.n_cols*m_Psquared.n_rows;
//    size += sizeof(double)*m_s.n_cols*m_s.n_rows;

//    return size;
//}

//string getTotalMemoryMbString()
//{
//    std::stringstream ss;
//    ss << getTotalMemory()/(1024*1024.0) << " Mb";
//    return  ss.str();
//}
