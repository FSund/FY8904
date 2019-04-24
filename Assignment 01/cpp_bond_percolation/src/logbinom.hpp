#pragma once

#include <cassert>
#include <armadillo>

using namespace std;
using namespace arma;

class LogBinomCoeffGenerator
{
public:
    LogBinomCoeffGenerator(const uword max):
        m_logSum(tabulateLogSum(max))
    {}

    double operator()(const uword n, const uword k) const
    {
        // returns log of binomial coefficient (log(n choose k))
        if (!(n < m_logSum.n_elem)) throw runtime_error("!(n < m_logSum.n_elem)");
        assert(n < m_logSum.n_elem);
        if (!(n >= k)) throw runtime_error("!(n >= k)");
        assert(n >= k);

        double coeff = m_logSum(n) - m_logSum(k) - m_logSum(n - k);
        return coeff;
    }

private:
    const vec m_logSum; // this will contain sum_{i=1}^n log i

    static vec tabulateLogSum(const uword max);
};
