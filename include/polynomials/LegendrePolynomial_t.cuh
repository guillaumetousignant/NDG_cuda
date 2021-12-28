#ifndef NDG_POLYNOMIALS_LEGENDREPOLYNOMIAL_T_CUH
#define NDG_POLYNOMIALS_LEGENDREPOLYNOMIAL_T_CUH

#include "helpers/float_types.h"

namespace SEM { namespace Device { namespace Polynomials {
    // Algorithm 23
    __global__
    void legendre_gauss_nodes_and_weights(int N, deviceFloat* nodes, deviceFloat* weights);

    class LegendrePolynomial_t { 
        public: 
            static void nodes_and_weights(int N_max, int blockSize, deviceFloat* nodes, deviceFloat* weights, const cudaStream_t &stream);

            // Algorithm 22
            __device__
            static void polynomial_and_derivative(int N, deviceFloat x, deviceFloat &L_N, deviceFloat &L_N_prime);

            __device__
            static deviceFloat polynomial(int N, deviceFloat x);
    };
}}}

#endif