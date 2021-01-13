#ifndef NDG_LEGENDREPOLYNOMIAL_T_H
#define NDG_LEGENDREPOLYNOMIAL_T_H

namespace SEM {
    // Algorithm 22
    __device__
    void legendre_polynomial_and_derivative(int N, float x, float &L_N, float &L_N_prime);

    // Algorithm 23
    __global__
    void legendre_gauss_nodes_and_weights(int N, float* nodes, float* weights);
}

class LegendrePolynomial_t { 
    public: 
        static void nodes_and_weights(int N_max, int blockSize, float* nodes, float* weights);
};

#endif