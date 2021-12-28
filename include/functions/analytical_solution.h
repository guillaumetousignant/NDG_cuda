#ifndef NDG_FUNCTIONS_ANALYTICAL_SOLUTION_H
#define NDG_FUNCTIONS_ANALYTICAL_SOLUTION_H

#include "entities/Vec2.h"
#include <array>

namespace SEM { namespace Host {
    /**
     * @brief Returns the analytical solution at a specific coordinate in global space and time.
     * 
     * @param xy 2D global coordinates, x and y.
     * @param t Time coordinate.
     * @return std::array<T, 3> Array of the state at these coordinates, [p, u, v].
     */
    template <class T>
    auto g(SEM::Host::Entities::Vec2<T> xy, T t) -> std::array<T, 3>;
}}

#include "functions/analytical_solution.tpp"

#endif
