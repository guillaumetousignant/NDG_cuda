#ifndef SEM_FUNCTIONS_FUNCTIONS_CUH
#define SEM_FUNCTIONS_FUNCTIONS_CUH

namespace SEM { namespace Device {
    /**
     * @brief Contains functions relating to Hillbert curves.
     * 
     * These functions are used to create and use Hillbert curves for meshing and adaptivity.
     */
    namespace Hilbert {}
}}

#include "analytical_solution.cuh"
#include "Hilbert.h"
#include "Hilbert_splitting.cuh"
#include "inverse_quad_map.cuh"
#include "quad_map.cuh"
#include "quad_metrics.cuh"
#include "Utilities.h"

#endif