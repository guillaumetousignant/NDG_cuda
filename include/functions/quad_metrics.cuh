#ifndef NDG_QUAD_METRICS_H
#define NDG_QUAD_METRICS_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include <array>

namespace SEM {
    /**
     * @brief Returns the metric terms on a straight sided quadrilateral
     * 
     * Algorithm 100
     *
     * @param local_coordinates 2D local coordinates, xi and eta.
     * @param points Array of the four points in global coordinates defining the quadrilateral, counter-clockwise order.
     * @return std::array<SEM::Entities::Vec2<deviceFloat>, 2> Array of the x any y derivatives. {[dx_dxi, dx_deta], [dy_dxi, dy_deta]}
     */
    __host__ __device__
    auto quad_metrics(SEM::Entities::Vec2<deviceFloat> local_coordinates, const std::array<SEM::Entities::Vec2<deviceFloat>, 4>& points) -> std::array<SEM::Entities::Vec2<deviceFloat>, 2>;
}

#endif