#ifndef NDG_CHEBYSHEVPOLYNOMIAL_HOST_T_H
#define NDG_CHEBYSHEVPOLYNOMIAL_HOST_T_H

#include "float_types.h"
#include <vector>

class ChebyshevPolynomial_t { 
    public: 
        // Algorithm 26
        static void nodes_and_weights(int N, std::vector<hostFloat>& nodes, std::vector<hostFloat>& weights);
};

#endif