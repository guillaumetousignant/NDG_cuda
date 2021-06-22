#ifndef NDG_CHEBYSHEVPOLYNOMIAL_T_H
#define NDG_CHEBYSHEVPOLYNOMIAL_T_H

#include "helpers/float_types.h"

namespace SEM { namespace Polynomials {
    // Algorithm 26
    __global__
    void chebyshev_gauss_nodes_and_weights(int N, deviceFloat* nodes, deviceFloat* weights);

    class ChebyshevPolynomial_t { 
        public: 
            static void nodes_and_weights(int N_max, int blockSize, deviceFloat* nodes, deviceFloat* weights, const cudaStream_t &stream);

            __device__
            static deviceFloat polynomial(int N, deviceFloat x);
    };
}}

#endif