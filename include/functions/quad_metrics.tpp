// Algorithm 100
template <class T>
auto SEM::Host::quad_metrics(SEM::Host::Entities::Vec2<T> local_coordinates, const std::array<SEM::Host::Entities::Vec2<T>, 4>& points) -> std::array<SEM::Host::Entities::Vec2<T>, 2> {
    return {SEM::Host::Entities::Vec2<T>{((1 - local_coordinates.y()) * (points[1].x() - points[0].x()) + (1 + local_coordinates.y()) * (points[2].x() - points[3].x())) / 4, 
                                   ((1 - local_coordinates.x()) * (points[3].x() - points[0].x()) + (1 + local_coordinates.x()) * (points[2].x() - points[1].x())) / 4},
            SEM::Host::Entities::Vec2<T>{((1 - local_coordinates.y()) * (points[1].y() - points[0].y()) + (1 + local_coordinates.y()) * (points[2].y() - points[3].y())) / 4,
                                   ((1 - local_coordinates.x()) * (points[3].y() - points[0].y()) + (1 + local_coordinates.x()) * (points[2].y() - points[1].y())) / 4}};
}