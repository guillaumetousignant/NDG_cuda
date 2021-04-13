#ifndef SEM_POLYNOMIALS_H
#define SEM_POLYNOMIALS_H

namespace SEM { 
    /**
     * @brief Contains the different polynomials used through the library.
     * 
     * Different polynomials can be used as basis for the solvers.
     */
    namespace Polynomials {}
}

#include "ChebyshevPolynomial_t.cuh"
#include "ChebyshevPolynomial_host_t.h"
#include "LegendrePolynomial_t.cuh"
#include "LegendrePolynomial_host_t.h"

#endif