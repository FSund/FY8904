#include "sites.hpp"

Sites::Sites(const uword m, const uword n):
    m_m(m),
    m_n(n),
    m_N(m*n),
    m_sites(zeros<ivec>(m*n) - 1)
{
    m_sizeOfLargestCluster = 0;
    m_largestClusterRoot = 0;

    m_giantComponent = 0;
    m_giantComponentSquared = 0;
    m_sizeSum = m*n;
    m_averageSquaredSize = 0;
}

void Sites::activate(const uvec& bond)
{
    uword node1 = bond(0);
    uword node2 = bond(1);
    uword root1 = findRoot(node1);
    uword root2 = findRoot(node2);

    if (root1 == root2)
    {
        // nodes belong to the same cluster -- do nothing
    }
    else
    {
        mergeClusters(root1, root2);
    }
}

void Sites::activateMat(const umat& bonds)
{
    for (uword i = 0; i < bonds.n_cols; i++)
    {
        activate(bonds.col(i));
    }
}

void Sites::mergeClusters(const uword root1, const uword root2)
{
    uword larger = root1;
    uword smaller = root2;
    if (m_sites(root2) < m_sites(root1)) // if root2 cluster is larger
    {
        larger = root2;
        smaller = root1;
    }

    // subtract the square of the size of the two clusters that are going to be
    // merged from the sum
    m_sizeSum -= pow2(m_sites(larger)) + pow2(m_sites(smaller));

    // add size of smaller cluster to larger cluster
    m_sites(larger) += m_sites(smaller);

    // point root node of smaller cluster root node of larger cluster
    m_sites(smaller) = sword(larger);

    // add the square of the size of the merged cluster to the sum
    m_sizeSum += pow2(m_sites(larger));

    updateLargestCluster(larger);

    if (m_sizeOfLargestCluster == m_N)
    {
        // avoid division by zero
        m_averageSquaredSize = 0;
    }
    else
    {
        // TODO: check signed-ness of this calculation
        m_averageSquaredSize = (m_sizeSum - pow2(m_N*m_giantComponent))
                /(m_N*(1 - m_giantComponent));
    }
}

void Sites::updateLargestCluster(const uword root)
{
    if (uword(abs(m_sites(root))) > m_sizeOfLargestCluster)
    {
        m_largestClusterRoot = root;
        m_sizeOfLargestCluster = uword(abs(m_sites(root)));
        m_giantComponent = m_sizeOfLargestCluster/double(m_N);
        m_giantComponentSquared = pow2(m_sizeOfLargestCluster/double(m_N));
    }
}

uword Sites::findRoot(const uword site)
{
    // recursive function
    if (m_sites(site) < 0)
    {
        // if sites[i] is negative, this is a root node with size sites[i]
        // return the index of the root node
        return site;
    }
    else
    {
        // this is not a root node, but a node belonging to a cluster with root
        // node sites[i]
        // call findroot again, update sites[i] to point to the root node, and
        // return the root node
        m_sites(site) = sword(findRoot(uword(m_sites(site))));

        return uword(m_sites(site)); // return after updating
    }
}

umat Sites::makeImage()
{
    umat image = zeros<umat>(m_m, m_n); // initialize to zero
    for (uword node = 0; node < m_sites.n_elem; node++)
    {
        if (findRoot(node) == m_largestClusterRoot)
        {
            // this node belongs to largest cluster
            uword i = node/m_n;
            uword j = node%m_m;
            image(i, j) = 1;
        }
    }

    return image;
}
