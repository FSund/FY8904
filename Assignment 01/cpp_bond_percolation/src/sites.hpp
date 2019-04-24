#pragma once

#include <armadillo>

using namespace std;
using namespace arma;

class Sites
{
public:
    Sites(const uword m, const uword n);

    void activate(const uvec& bond);
    void activateMat(const umat& bonds);

    const ivec& sites() const { return m_sites; }
    double giantComponent() const { return m_giantComponent; }
    double giantComponentSquared() const { return m_giantComponentSquared; }
    double averageSquaredSize() const { return m_averageSquaredSize; }
    uword sizeOfLargestCluster() const { return m_sizeOfLargestCluster; }
    umat makeImage();

private:
    uword m_m;
    uword m_n;
    uword m_N;
    ivec m_sites;

    uword m_sizeOfLargestCluster;
    uword m_largestClusterRoot;

    double m_giantComponent;
    double m_giantComponentSquared;
    double m_sizeSum;
    double m_averageSquaredSize;

    uword findRoot(const uword site);
    void mergeClusters(const uword root1, const uword root2);
    void updateLargestCluster(const uword root);
};

inline uword pow2(const uword value)
{
    return value*value;
}

inline sword pow2(const sword value)
{
    return value*value;
}

inline double pow2(const double value)
{
    return value*value;
}
