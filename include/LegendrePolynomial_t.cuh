#ifndef NDG_LEGENDREPOLYNOMIAL_T_H
#define NDG_LEGENDREPOLYNOMIAL_T_H

#include "float_types.h"

namespace SEM {
    // Algorithm 22
    __device__
    void legendre_polynomial_and_derivative(int N, deviceFloat x, deviceFloat &L_N, deviceFloat &L_N_prime);

    // Algorithm 23
    __global__
    void legendre_gauss_nodes_and_weights(int N, deviceFloat* nodes, deviceFloat* weights);
}

class LegendrePolynomial_t { 
    public: 
        static void nodes_and_weights(int N_max, int blockSize, deviceFloat* nodes, deviceFloat* weights);

        // Algorithm 22
        __device__
        void polynomial_and_derivative(int N, deviceFloat x, deviceFloat &L_N, deviceFloat &L_N_prime);
};

#endif