#ifndef NDG_QUAD_METRICS_H
#define NDG_QUAD_METRICS_H

#include "entities/Vec2.cuh"
#include <array>

namespace SEM {
    /**
     * @brief Returns the metric terms on a straight sided quadrilateral
     * 
     * Algorithm 100
     *
     * @param local_coordinates 2D local coordinates, xi and eta, both in range [-1, 1].
     * @param points Array of the four points in global coordinates defining the quadrilateral, counter-clockwise order.
     * @return std::array<SEM::Entities::Vec2<T>, 2> Array of the x any y derivatives. {[dx_dxi, dx_deta], [dy_dxi, dy_deta]}
     */
    #if defined(__CUDA__)
    __host__ __device__
    #endif
    template <class T>
    auto quad_metrics(SEM::Entities::Vec2<T> local_coordinates, const std::array<SEM::Entities::Vec2<T>, 4>& points) -> std::array<SEM::Entities::Vec2<T>, 2>;
}

#include "functions/quad_metrics.tcu"

#endif