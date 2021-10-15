#ifndef NDG_HELPERS_CONSTANTS_H
#define NDG_HELPERS_CONSTANTS_H

#include "helpers/float_types.h"
#include "entities/Vec2.h"
#include <cmath>

namespace SEM { namespace Host {
    /**
     * @brief Contains constants used through the library.
     * 
     * Most are aero constants or initial conditions.
     */
    namespace Constants {
        constexpr hostFloat c = 1;
        constexpr SEM::Host::Entities::Vec2<hostFloat> xy0 {-0.2, -0.2};
        constexpr SEM::Host::Entities::Vec2<hostFloat> k {0.707106781186548, 0.707106781186548}; // √2/2, √2/2
        constexpr hostFloat d = 0.120112240878645; // 0.2/(2 * √ln(2))
        constexpr int n_points_least_squares_max = 4; // Don't use more than the last n points from the spectrum to estimate error
    }
}}

#endif