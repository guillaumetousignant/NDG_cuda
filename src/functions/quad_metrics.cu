#include "functions/quad_metrics.cuh"

// Algorithm 100
__host__ __device__
auto SEM::quad_metrics(SEM::Entities::Vec2<deviceFloat> local_coordinates, const std::array<SEM::Entities::Vec2<deviceFloat>, 4>& points) -> std::array<SEM::Entities::Vec2<deviceFloat>, 2> {
    return {SEM::Entities::Vec2<deviceFloat>{((1 - local_coordinates.y()) * (points[1].x() - points[0].x()) + (1 + local_coordinates.y()) * (points[2].x() - points[3].x())) / 4, 
                                             ((1 - local_coordinates.x()) * (points[3].x() - points[0].x()) + (1 + local_coordinates.x()) * (points[2].x() - points[1].x())) / 4},
            SEM::Entities::Vec2<deviceFloat>{((1 - local_coordinates.y()) * (points[1].y() - points[0].y()) + (1 + local_coordinates.y()) * (points[2].y() - points[3].y())) / 4,
                                             ((1 - local_coordinates.x()) * (points[3].y() - points[0].y()) + (1 + local_coordinates.x()) * (points[2].y() - points[1].y())) / 4}};
}