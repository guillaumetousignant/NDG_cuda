#ifndef NDG_POLYNOMIALS_LEGENDREPOLYNOMIAL_T_H
#define NDG_POLYNOMIALS_LEGENDREPOLYNOMIAL_T_H

#include "helpers/float_types.h"
#include <vector>

namespace SEM { namespace Host { namespace Polynomials {
    class LegendrePolynomial_t { 
        public: 
            // Algorithm 23
            static void nodes_and_weights(int N, std::vector<hostFloat>& nodes, std::vector<hostFloat>& weights);

            static hostFloat polynomial(int N, hostFloat x);

        private:
            // Algorithm 22
            static void legendre_polynomial_and_derivative(int N, hostFloat x, hostFloat &L_N, hostFloat &L_N_prime);
    };
}}}

#endif