#ifndef NDG_LEGENDREPOLYNOMIAL_HOST_T_H
#define NDG_LEGENDREPOLYNOMIAL_HOST_T_H

#include <vector>

class LegendrePolynomial_t { 
    public: 
        static void nodes_and_weights(int N, float* nodes, float* weights);

    private:
        // Algorithm 22
        static void legendre_polynomial_and_derivative(int N, float x, float &L_N, float &L_N_prime);

        // Algorithm 23
        static void legendre_gauss_nodes_and_weights(int N, float* nodes, float* weights);
};

#endif