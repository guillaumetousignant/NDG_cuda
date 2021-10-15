#ifndef NDG_FUNCTIONS_QUAD_MAP_H
#define NDG_FUNCTIONS_QUAD_MAP_H

#include "entities/Vec2.h"
#include <array>

namespace SEM { namespace Host {
    /**
     * @brief Returns the global coordinates of a point in local coordinates, with the global coordinates of the four corners.
     *
     * Algorithm 95
     * 
     * @param local_coordinates 2D local coordinates, xi and eta, both in range [-1, 1].
     * @param points Array of the four points in global coordinates defining the quadrilateral, counter-clockwise order.
     * @return SEM::Host::Entities::Vec2<T> 2D global coordinates, x and y.
     */
    template <class T>
    auto quad_map(SEM::Host::Entities::Vec2<T> local_coordinates, const std::array<SEM::Host::Entities::Vec2<T>, 4>& points) -> SEM::Host::Entities::Vec2<T>;
}}

#include "functions/quad_map.tpp"

#endif
