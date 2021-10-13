#ifndef SEM_SOLVERS_H
#define SEM_SOLVERS_H

namespace SEM { 
    /**
     * @brief Contains the different solvers used through the library.
     * 
     * These solvers can be used with the different meshes and polynomials, and have different properties.
     */
    namespace Solvers {}
}

#include "Solver2D_t.cuh"
#include "Solver2D_host_t.h"

#endif