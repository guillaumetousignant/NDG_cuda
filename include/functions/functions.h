#ifndef SEM_FUNCTIONS_H
#define SEM_FUNCTIONS_H

namespace SEM { 
    /**
     * @brief Contains functions relating to Hillbert curves.
     * 
     * These functions are used to create and use Hillbert curves for meshing and adaptivity.
     */
    namespace Hilbert {}
}

#include "Hilbert.h"
#include "inverse_quad_map.cuh"
#include "quad_map.cuh"
#include "quad_metrics.cuh"
#include "Utilities.h"

#endif