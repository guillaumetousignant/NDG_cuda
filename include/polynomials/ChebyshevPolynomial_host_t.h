#ifndef NDG_CHEBYSHEVPOLYNOMIAL_HOST_T_H
#define NDG_CHEBYSHEVPOLYNOMIAL_HOST_T_H

#include "helpers/float_types.h"
#include <vector>

namespace SEM { namespace Polynomials {
    class ChebyshevPolynomial_host_t { 
        public: 
            // Algorithm 26
            static void nodes_and_weights(int N, std::vector<hostFloat>& nodes, std::vector<hostFloat>& weights);

            static hostFloat polynomial(int N, hostFloat x);
    };
}}

#endif