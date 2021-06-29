#ifndef NDG_INVERSE_QUAD_MAP_H
#define NDG_INVERSE_QUAD_MAP_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include <array>

namespace SEM {
    /**
     * @brief Returns the local coordinates of a point in global coordinates, with the global coordinates of the four corners.
     *
     * From: 
     * https://www.iquilezles.org/www/articles/ibilinear/ibilinear.htm
     * 
     * @param global_coordinates 2D global coordinates, x and y.
     * @param points Array of the four points in global coordinates defining the quadrilateral, counter-clockwise order.
     * @return SEM::Entities::Vec2<deviceFloat> 2D local coordinates, xi and eta.
     */
    __host__ __device__
    auto inverse_quad_map(SEM::Entities::Vec2<deviceFloat> global_coordinates, const std::array<SEM::Entities::Vec2<deviceFloat>, 4>& points) -> SEM::Entities::Vec2<deviceFloat>;
}

#endif
