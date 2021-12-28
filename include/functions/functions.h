#ifndef SEM_FUNCTIONS_FUNCTIONS_H
#define SEM_FUNCTIONS_FUNCTIONS_H

namespace SEM { namespace Host {
    /**
     * @brief Contains functions relating to Hillbert curves.
     * 
     * These functions are used to create and use Hillbert curves for meshing and adaptivity.
     */
    namespace Hilbert {}
}}

#include "analytical_solution.h"
#include "Hilbert.h"
#include "Hilbert_splitting.h"
#include "inverse_quad_map.h"
#include "quad_map.h"
#include "quad_metrics.h"
#include "Utilities.h"

#endif