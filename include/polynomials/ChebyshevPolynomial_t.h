#ifndef NDG_POLYNOMIALS_CHEBYSHEVPOLYNOMIAL_T_H
#define NDG_POLYNOMIALS_CHEBYSHEVPOLYNOMIAL_T_H

#include "helpers/float_types.h"
#include <vector>

namespace SEM { namespace Host { namespace Polynomials {
    class ChebyshevPolynomial_t { 
        public: 
            // Algorithm 26
            static void nodes_and_weights(int N, std::vector<hostFloat>& nodes, std::vector<hostFloat>& weights);

            static hostFloat polynomial(int N, hostFloat x);
    };
}}}

#endif