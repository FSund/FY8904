#include "logbinom.hpp"
#if 1
vec LogBinomCoeffGenerator::tabulateLogSum(const uword max)
{
    vec logSum(max + 1);
    for (uword i = 1; i <= max; i++)
    {
        logSum(i) = logSum(i-1) + log(i);
    }

    return logSum;
}
#else
vec LogBinomCoeffGenerator::tabulateLogSum(const uword max)
{
    cout << "LogBinomCoeffGenerator() tabulating log sums from 1 to " << max;
    cout.flush();
    auto start = chrono::high_resolution_clock::now();
    vec logSum(max + 1);
    for (uword i = 1; i <= max; i++)
    {
        logSum(i) = logSum(i-1) + log(i);
    }
    auto stop = chrono::high_resolution_clock::now();
    cout << " ... Done" << endl;
    cout << "Time spent " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms" << endl;

    return logSum;
}
#endif
