#include "entities/Face2D_t.h"
#include <utility>

SEM::Host::Entities::Face2D_t::Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements, std::array<size_t, 2> elements_side) : 
        N_{N},
        nodes_{nodes},
        elements_{elements},
        elements_side_{elements_side},
        normal_{0, 0},
        tangent_{0, 0},
        length_{0},
        offset_{0.0, 0.0},
        scale_{0.0, 0.0},
        refine_{false},
        p_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        u_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        v_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        p_flux_(N_ + 1),
        u_flux_(N_ + 1),
        v_flux_(N_ + 1) {}

SEM::Host::Entities::Face2D_t::Face2D_t() :
        N_{0},
        nodes_{0, 0},
        elements_{0, 0},
        elements_side_{0, 0},
        normal_{0, 0},
        tangent_{0, 0},
        length_{0},
        offset_{0.0, 0.0},
        scale_{0.0, 0.0},
        refine_{false},
        p_{},
        u_{},
        v_{} {}

auto SEM::Host::Entities::Face2D_t::allocate_storage() -> void {
    p_ = {std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)};
    u_ = {std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)};
    v_ = {std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)};
    p_flux_ = std::vector<hostFloat>(N_ + 1);
    u_flux_ = std::vector<hostFloat>(N_ + 1);
    v_flux_ = std::vector<hostFloat>(N_ + 1);
}

auto SEM::Host::Entities::Face2D_t::resize_storage(int N) -> void {
    N_ = N;
    p_ = {std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)};
    u_ = {std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)};
    v_ = {std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)};
    p_flux_ = std::vector<hostFloat>(N_ + 1);
    u_flux_ = std::vector<hostFloat>(N_ + 1);
    v_flux_ = std::vector<hostFloat>(N_ + 1);
}

auto SEM::Host::Entities::Face2D_t::compute_geometry(const std::array<SEM::Host::Entities::Vec2<hostFloat>, 2>& elements_centres, const std::array<SEM::Host::Entities::Vec2<hostFloat>, 2>& nodes, const std::array<std::array<SEM::Host::Entities::Vec2<hostFloat>, 2>, 2>& element_nodes) -> void {
    tangent_ = nodes[1] - nodes[0]; 
    length_ = tangent_.magnitude();
    tangent_ /= length_; // CHECK should be normalized or not?
    normal_ = SEM::Host::Entities::Vec2<hostFloat>(tangent_.y(), -tangent_.x());     

    const SEM::Host::Entities::Vec2<hostFloat> center = (nodes[0] + nodes[1])/2;
    const SEM::Host::Entities::Vec2<hostFloat> delta = center - elements_centres[0]; // CHECK doesn't work with ghost cells
    const hostFloat sign = std::copysign(static_cast<hostFloat>(1), normal_.dot(delta));
    normal_ *= sign;
    tangent_ *= sign;

    const std::array<SEM::Host::Entities::Vec2<hostFloat>, 2> elements_delta {
        element_nodes[0][1] - element_nodes[0][0], 
        element_nodes[1][1] - element_nodes[1][0]
    };

    const SEM::Host::Entities::Vec2<hostFloat> face_delta = nodes[1] - nodes[0];

    const std::array<SEM::Host::Entities::Vec2<hostFloat>, 2> edge_delta {
        nodes[0] - element_nodes[0][0], 
        nodes[1] - element_nodes[1][0]
    };

    offset_ = {edge_delta[0].magnitude()/elements_delta[0].magnitude(), edge_delta[1].magnitude()/elements_delta[1].magnitude()};
    scale_ = {face_delta.magnitude()/elements_delta[0].magnitude(), face_delta.magnitude()/elements_delta[1].magnitude()};
}
