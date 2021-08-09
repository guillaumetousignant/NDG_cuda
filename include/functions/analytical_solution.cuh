#ifndef NDG_ANALYTICAL_SOLUTION_H
#define NDG_ANALYTICAL_SOLUTION_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include <array>

namespace SEM {
    /**
     * @brief Returns the analytical solution at a specific coordinate in global space and time.
     * 
     * @param xy 2D global coordinates, x and y.
     * @param t Time coordinate.
     * @return std::array<deviceFloat, 3> Array of the state at these coordinates, [p, u, v].
     */
    __host__ __device__
    auto g(SEM::Entities::Vec2<deviceFloat> xy, deviceFloat t) -> std::array<deviceFloat, 3>;
}

#endif
