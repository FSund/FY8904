#include "results.hpp"

void Results::sample(const Sites& sites, const uword iteration,
                     const uword nActivatedBonds)
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

    const double p = nActivatedBonds/double(m_nBonds); // fraction of activated

    m_P(m_idx) = sites.giantComponent();
    m_Psquared(m_idx) = sites.giantComponentSquared();
    m_s(m_idx) = sites.averageSquaredSize();

    if (iteration == 0)
    {
        m_p(m_idx) = p;
    }

    m_idx++;
}

#if 1
vec Results::calcResults(const vec& Q, const vec& pvec)
{
    const vec logQ = log(Q); // calculate logarithm outside loops

    const uword nps = pvec.n_elem; // number of probabilities
    const uword M = Q.n_rows; // number of bonds
    vec Qp(nps); // results

    // loop over all probabilities we want to get results for
    for (uword i = 0; i < nps; i++)
    {
        const double p = pvec(i);
        const double logp = log(p); // calculate logarithms outside loop
        const double logOneMinusP = log(1 - p);

        // calculate Q(p)
        double Qpi = 0;
        for (uword n = 0; n < M; n++) // loop over all Q's
        {
            const double a = m_logBinomCoeff(n); // tabulated values
            const double b = n*logp;
            const double c = (M - n)*logOneMinusP;
            const double d = logQ(n);

            const double logTerm = a + b + c + d;
            Qpi += exp(logTerm);
        }
        Qp(i) = Qpi;
    }

    return Qp;
}

#else
vec Results::calcResults(const vec& Q, const vec& pvec)
{
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
    cout << "Time spent calculating statistics " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms" << endl;

    return Qp;
}
#endif

void Results::save(const string& folder, const uword nSamples)
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
        vec pvec = arma::linspace(0, 1, nSamples + 1);
        pvec = pvec(span(1, nSamples)); // skip 0, since it gives nan when taking log

        // apply convolution to mean results
        vec giant = calcResults(m_giant.mean(), pvec);
        vec giant2(pvec.size());
        arma::interp1(m_p, m_giant.mean(), pvec, giant2);
        vec giantSquared = calcResults(m_giantSquared.mean(), pvec);
        vec averageSize = calcResults(m_averageSize.mean(), pvec);
        vec averageSize2(pvec.size());
        arma::interp1(m_p, m_averageSize.mean(), pvec, averageSize2);
        vec susceptibility = m_nSites*sqrt(giantSquared - giant%giant);
        vec susceptibility2 = calcResults(m_nSites*sqrt(m_giantSquared.mean() - m_giant.mean()%m_giant.mean()), pvec);

        giant.save(folder + "/giant.csv", arma::csv_ascii);
        giant2.save(folder + "/giant2.csv", arma::csv_ascii);
        averageSize.save(folder + "/averageSize.csv", arma::csv_ascii);
        averageSize2.save(folder + "/averageSize2.csv", arma::csv_ascii);
        susceptibility.save(folder + "/susceptibility.csv", arma::csv_ascii);
        susceptibility2.save(folder + "/susceptibility2.csv", arma::csv_ascii);
        pvec.save(folder + "/probability.csv", arma::csv_ascii);
        m_p.save(folder + "/probability_all.csv", arma::csv_ascii);

        cout << "Saved results to \"" << folder << "\"" << endl;
    }
}
