#include <cmath>
#include <limits>

using SEM::Host::Entities::Vec2;

// Algorithm 95
template <class T>
auto SEM::Host::inverse_quad_map(Vec2<T> global_coordinates, const std::array<Vec2<T>, 4>& points) -> Vec2<T> {
    /*const deviceFloat A = (points[0] - global_coordinates).cross(points[0] - points[2]);
    const deviceFloat B = ((points[0] - global_coordinates).cross(points[1] - points[3]) + (points[1] - global_coordinates).cross(points[0] - points[2])) / 2;
    const deviceFloat C = (points[1] - global_coordinates).cross(points[1] - points[3]);

    const deviceFloat denominator = (A - 2 * B + C);

    if (std::abs(denominator) < static_cast<deviceFloat>(0.00001)) {
        const deviceFloat s = A / (A - C);

        const deviceFloat t_denominator_x = (1 - s) * (points[0].x() - points[2].x()) + s * (points[1].x() - points[3].x());
        const deviceFloat t_denominator_y = (1 - s) * (points[0].y() - points[2].y()) + s * (points[1].y() - points[3].y());

        const deviceFloat t = (std::abs(t_denominator_x) > std::abs(t_denominator_y)) ? ((1 - s) * (points[0].x() - global_coordinates.x()) + s * (points[1].x() - global_coordinates.x())) / t_denominator_x : ((1 - s) * (points[0].y() - global_coordinates.y()) + s * (points[1].y() - global_coordinates.y())) / t_denominator_y;

        return {s, t};
    }
    else {
        const deviceFloat s_plus = ((A - B) + std::sqrt(B * B - A * C)) / denominator;
        const deviceFloat s_minus = ((A - B) - std::sqrt(B * B - A * C)) / denominator;

        const deviceFloat t_plus_denominator_x = (1 - s_plus) * (points[0].x() - points[2].x()) + s_plus * (points[1].x() - points[3].x());
        const deviceFloat t_plus_denominator_y = (1 - s_plus) * (points[0].y() - points[2].y()) + s_plus * (points[1].y() - points[3].y());

        const deviceFloat t_plus = (std::abs(t_plus_denominator_x) > std::abs(t_plus_denominator_y)) ? ((1 - s_plus) * (points[0].x() - global_coordinates.x()) + s_plus * (points[1].x() - global_coordinates.x())) / t_plus_denominator_x : ((1 - s_plus) * (points[0].y() - global_coordinates.y()) + s_plus * (points[1].y() - global_coordinates.y())) / t_plus_denominator_y;

        const deviceFloat t_minus_denominator_x = (1 - s_minus) * (points[0].x() - points[2].x()) + s_minus * (points[1].x() - points[3].x());
        const deviceFloat t_minus_denominator_y = (1 - s_minus) * (points[0].y() - points[2].y()) + s_minus * (points[1].y() - points[3].y());

        const deviceFloat t_minus = (std::abs(t_minus_denominator_x) > std::abs(t_minus_denominator_y)) ? ((1 - s_minus) * (points[0].x() - global_coordinates.x()) + s_minus * (points[1].x() - global_coordinates.x())) / t_minus_denominator_x : ((1 - s_minus) * (points[0].y() - global_coordinates.y()) + s_minus * (points[1].y() - global_coordinates.y())) / t_minus_denominator_y;
    
        const deviceFloat s = s_plus;
        const deviceFloat t = t_plus;

        return {s, t};
    }*/

    const Vec2<T> e = points[1] - points[0];
    const Vec2<T> f = points[3] - points[0];
    const Vec2<T> g = points[0] - points[1] + points[2] - points[3];
    const Vec2<T> h = global_coordinates - points[0];
        
    const T k2 = g.cross(f);
    const T k1 = e.cross(f) + h.cross(g);
    const T k0 = h.cross(e);
    
    // if edges are parallel, this is a linear equation
    if(std::abs(k2) < std::numeric_limits<T>::min()) {
        const T u_denominator_x = e.x() * k1 - g.x() * k0;
        const T u_denominator_y = e.y() * k1 - g.y() * k0;

        const T u = (std::abs(u_denominator_x) > std::abs(u_denominator_y)) ? (h.x() * k1 + f.x() * k0) / u_denominator_x : (h.y() * k1 + f.y() * k0) / u_denominator_y;

        return {u * 2 - 1, -k0/k1 * 2 - 1};
    }
    // otherwise, it's a quadratic
	else
    {
        T w = k1 * k1 - 4 * k0 * k2;
        if(w < 0.0) return {-1, -1};
        w = std::sqrt(w);

        const T ik2 = 1 / (2 * k2);
        T v = (-k1 - w) * ik2;

        T u_denominator_x = e.x() + g.x() * v;
        T u_denominator_y = e.y() + g.y() * v; 

        T u = (std::abs(u_denominator_x) > std::abs(u_denominator_y)) ? (h.x() - f.x() * v) / u_denominator_x : (h.y() - f.y() * v) / u_denominator_y;
        
        if(u < static_cast<T>(0.0) || u > static_cast<T>(1.0) || v < static_cast<T>(0.0) || v > static_cast<T>(1.0)) {
            v = (-k1 + w) * ik2;

            u_denominator_x = e.x() + g.x() * v;
            u_denominator_y = e.y() + g.y() * v; 

            u = (std::abs(u_denominator_x) > std::abs(u_denominator_y)) ? (h.x() - f.x() * v) / u_denominator_x : (h.y() - f.y() * v) / u_denominator_y;
        }
        return {u * 2 - 1, v * 2 - 1};
    }
}
