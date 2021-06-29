#ifndef NDG_QUAD_MAP_H
#define NDG_QUAD_MAP_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include <array>

namespace SEM {
    /**
     * @brief Returns the global coordinates of a point in local coordinates, with the global coordinates of the four corners.
     *
     * Algorithm 95
     * 
     * @param local_coordinates 2D local coordinates, xi and eta, both in range [-1, 1].
     * @param points Array of the four points in global coordinates defining the quadrilateral, counter-clockwise order.
     * @return SEM::Entities::Vec2<deviceFloat> 2D global coordinates, x and y.
     */
    __host__ __device__
    auto quad_map(SEM::Entities::Vec2<deviceFloat> local_coordinates, const std::array<SEM::Entities::Vec2<deviceFloat>, 4>& points) -> SEM::Entities::Vec2<deviceFloat>;
}

#endif
