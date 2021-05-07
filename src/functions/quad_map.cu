#include "functions/quad_map.cuh"

// Algorithm 95
__host__ __device__
auto SEM::quad_map(SEM::Entities::Vec2<deviceFloat> local_coordinates, const std::array<SEM::Entities::Vec2<deviceFloat>, 4>& points) -> SEM::Entities::Vec2<deviceFloat> {
    return (points[0] * (1 - local_coordinates.x()) * (1 - local_coordinates.y()) 
            + points[1] * (local_coordinates.x() + 1) * (1 - local_coordinates.y())
            + points[2] * (local_coordinates.x() + 1) * (local_coordinates.y() + 1)
            + points[3] * (1 - local_coordinates.x()) * (local_coordinates.y() + 1)) / 4;
}