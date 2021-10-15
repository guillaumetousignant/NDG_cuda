// Algorithm 95
template <class T>
auto SEM::Host::quad_map(SEM::Host::Entities::Vec2<T> local_coordinates, const std::array<SEM::Host::Entities::Vec2<T>, 4>& points) -> SEM::Host::Entities::Vec2<T> {
    return (points[0] * (1 - local_coordinates.x()) * (1 - local_coordinates.y()) 
            + points[1] * (local_coordinates.x() + 1) * (1 - local_coordinates.y())
            + points[2] * (local_coordinates.x() + 1) * (local_coordinates.y() + 1)
            + points[3] * (1 - local_coordinates.x()) * (local_coordinates.y() + 1)) / 4;
}