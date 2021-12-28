#ifndef NDG_FUNCTIONS_INVERSE_QUAD_MAP_H
#define NDG_FUNCTIONS_INVERSE_QUAD_MAP_H

#include "entities/Vec2.h"
#include <array>

namespace SEM { namespace Host {
    /**
     * @brief Returns the local coordinates of a point in global coordinates, with the global coordinates of the four corners.
     *
     * From: 
     * https://www.iquilezles.org/www/articles/ibilinear/ibilinear.htm
     * 
     * @param global_coordinates 2D global coordinates, x and y.
     * @param points Array of the four points in global coordinates defining the quadrilateral, counter-clockwise order.
     * @return SEM::Host::Entities::Vec2<T> 2D local coordinates, xi and eta.
     */
    template <class T>
    auto inverse_quad_map(SEM::Host::Entities::Vec2<T> global_coordinates, const std::array<SEM::Host::Entities::Vec2<T>, 4>& points) -> SEM::Host::Entities::Vec2<T>;
}}

#include "functions/inverse_quad_map.tpp"

#endif
