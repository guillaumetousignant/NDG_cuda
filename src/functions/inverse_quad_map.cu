#include "functions/inverse_quad_map.cuh"
#include <cmath>

using SEM::Entities::Vec2;

// Algorithm 95
__host__ __device__
auto SEM::inverse_quad_map(Vec2<deviceFloat> global_coordinates, const std::array<Vec2<deviceFloat>, 4>& points) -> Vec2<deviceFloat> {
    const Vec2<deviceFloat> e = points[1] - points[0];
    const Vec2<deviceFloat> f = points[3] - points[0];
    const Vec2<deviceFloat> g = points[0] - points[1] + points[2] - points[3];
    const Vec2<deviceFloat> h = global_coordinates - points[0];

    const deviceFloat k2 = g.cross(f);
    const deviceFloat k1 = e.cross(f) + h.cross(g);
    const deviceFloat k0 = h.cross(e);

    if(std::abs(k2) < static_cast<deviceFloat>(0.00001)) {
        return {(h.x() * k1 + f.x() * k0) / (e.x() * k1 - g.x() * k0), -k0 / k1};
    }
    else {
        const deviceFloat w = std::sqrt(k1 * k1 - 4 * k0 * k2); // CHECK this could nan

        const deviceFloat ik2 = 1/(k2 * 2);
        const deviceFloat v = (-k1 - w) * ik2;
        const deviceFloat u = (h.x() - f.x() * v)/(e.x() + g.x() * v);

        return {u, v}; // CHECK originally this checks if outside reference square.
    }
}