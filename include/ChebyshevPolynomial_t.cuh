#ifndef NDG_CHEBYSHEVPOLYNOMIAL_T_H
#define NDG_CHEBYSHEVPOLYNOMIAL_T_H

#include "float_types.h"

namespace SEM {
    // Algorithm 26
    __global__
    void chebyshev_gauss_nodes_and_weights(int N, deviceFloat* nodes, deviceFloat* weights);
}

class ChebyshevPolynomial_t { 
    public: 
        static void nodes_and_weights(int N_max, int blockSize, deviceFloat* nodes, deviceFloat* weights);

        __device__
        static void polynomial(int N, deviceFloat x, deviceFloat &T_N);
};

#endif