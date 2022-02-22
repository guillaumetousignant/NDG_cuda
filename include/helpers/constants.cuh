#ifndef NDG_HELPERS_CONSTANTS_CUH
#define NDG_HELPERS_CONSTANTS_CUH

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include <cmath>

namespace SEM { namespace Device {
    /**
     * @brief Contains constants used through the library.
     * 
     * Most are aero constants or initial conditions.
     */
    namespace Constants {
        constexpr deviceFloat c = 1;
        constexpr SEM::Device::Entities::Vec2<deviceFloat> xy0 {0.45, 0.45};
        constexpr SEM::Device::Entities::Vec2<deviceFloat> k {0.707106781186548, 0.707106781186548}; // √2/2, √2/2
        constexpr deviceFloat d = 0.030112240878645; // 0.2/(2 * √ln(2))
        __device__ const int n_points_least_squares_max = 4; // Don't use more than the last n points from the spectrum to estimate error
    }
}}

#endif