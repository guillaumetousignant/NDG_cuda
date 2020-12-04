#ifndef CHEBYSHEVPOLYNOMIAL_T_H
#define CHEBYSHEVPOLYNOMIAL_T_H

namespace SEM {
    // Algorithm 26
    __global__
    void chebyshev_gauss_nodes_and_weights(int N, float* nodes, float* weights);
}

class ChebyshevPolynomial_t { 
    public: 
        static void nodes_and_weights(int N_max, int blockSize, float* nodes, float* weights);
};

#endif