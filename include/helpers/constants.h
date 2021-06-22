#ifndef NDG_CONSTANTS_H
#define NDG_CONSTANTS_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include <cmath>

namespace SEM { 
    /**
     * @brief Contains constants used through the library.
     * 
     * Most are aero constants or initial conditions.
     */
    namespace Constants {
        constexpr deviceFloat c = 1;
        constexpr SEM::Entities::Vec2<deviceFloat> xy0 {0.2, 0.2};
        constexpr SEM::Entities::Vec2<deviceFloat> k {0.707106781186548, 0.707106781186548}; // √2/2, √2/2
        constexpr deviceFloat d = 0.120112240878645; // 0.2/(2 * √ln(2))
        constexpr deviceFloat tolerance_min = 1e-6;     // Refine above this
        constexpr deviceFloat tolerance_max = 1e-14;    // Coarsen below this
    }
}

#endif