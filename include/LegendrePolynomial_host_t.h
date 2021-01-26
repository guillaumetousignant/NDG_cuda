#ifndef NDG_LEGENDREPOLYNOMIAL_HOST_T_H
#define NDG_LEGENDREPOLYNOMIAL_HOST_T_H

#include "float_types.h"
#include <vector>

class LegendrePolynomial_host_t { 
    public: 
        // Algorithm 23
        static void nodes_and_weights(int N, std::vector<hostFloat>& nodes, std::vector<hostFloat>& weights);

    private:
        // Algorithm 22
        static void legendre_polynomial_and_derivative(int N, hostFloat x, hostFloat &L_N, hostFloat &L_N_prime);
};

#endif