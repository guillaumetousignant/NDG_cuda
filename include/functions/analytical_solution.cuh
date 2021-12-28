#ifndef NDG_FUNCTIONS_ANALYTICAL_SOLUTION_CUH
#define NDG_FUNCTIONS_ANALYTICAL_SOLUTION_CUH

#include "entities/Vec2.cuh"
#include <array>

namespace SEM { namespace Device {
    /**
     * @brief Returns the analytical solution at a specific coordinate in global space and time.
     * 
     * @param xy 2D global coordinates, x and y.
     * @param t Time coordinate.
     * @return std::array<T, 3> Array of the state at these coordinates, [p, u, v].
     */
    template <class T>
    __host__ __device__
    auto g(SEM::Device::Entities::Vec2<T> xy, T t) -> std::array<T, 3>;
}}

#include "functions/analytical_solution.tcu"

#endif
