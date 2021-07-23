#include "entities/Face2D_t.cuh"
#include <utility>

__device__ 
SEM::Entities::Face2D_t::Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements, std::array<size_t, 2> elements_side) : 
        N_{N},
        nodes_{nodes},
        elements_{elements},
        elements_side_{elements_side},
        normal_{0, 0},
        tangent_{0, 0},
        length_{0},
        offset_{0.0, 0.0},
        scale_{0.0, 0.0},
        p_{N_ + 1, N_ + 1},
        u_{N_ + 1, N_ + 1},
        v_{N_ + 1, N_ + 1},
        p_flux_{N_ + 1},
        u_flux_{N_ + 1},
        v_flux_{N_ + 1} {}

__host__ __device__
SEM::Entities::Face2D_t::Face2D_t() :
        N_{0},
        nodes_{0, 0},
        elements_{0, 0},
        elements_side_{0, 0},
        normal_{0, 0},
        tangent_{0, 0},
        length_{0},
        offset_{0.0, 0.0},
        scale_{0.0, 0.0},
        p_{},
        u_{},
        v_{} {}

__device__
auto SEM::Entities::Face2D_t::allocate_storage() -> void {
    p_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    u_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    v_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    p_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    u_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    v_flux_ = cuda_vector<deviceFloat>(N_ + 1);
}

__device__
auto SEM::Entities::Face2D_t::resize_storage(int N) -> void {
    N_ = N;
    p_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    u_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    v_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    p_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    u_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    v_flux_ = cuda_vector<deviceFloat>(N_ + 1);
}

__device__
auto SEM::Entities::Face2D_t::compute_geometry(const SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes) -> void {
    const std::array<SEM::Entities::Vec2<deviceFloat>, 2> points {nodes[nodes_[0]], nodes[nodes_[1]]};
    
    tangent_ = points[1] - points[0]; 
    length_ = tangent_.magnitude();
    tangent_ /= length_; // CHECK should be normalized or not?
    normal_ = SEM::Entities::Vec2<deviceFloat>(tangent_.y(), -tangent_.x());     
    
    const SEM::Entities::Element2D_t& element_L = elements[elements_[0]];
    const SEM::Entities::Element2D_t& element_R = elements[elements_[1]];

    const SEM::Entities::Vec2<deviceFloat> center = (points[0] + points[1])/2;
    const SEM::Entities::Vec2<deviceFloat> delta = center - element_L.center_; // CHECK doesn't work with ghost cells
    const deviceFloat sign = std::copysign(static_cast<deviceFloat>(1.0), normal_.dot(delta));
    normal_ *= sign;
    tangent_ *= sign;

    const std::array<std::array<SEM::Entities::Vec2<deviceFloat>, 2>, 2> elements_nodes {
        std::array<SEM::Entities::Vec2<deviceFloat>, 2>{nodes[element_L.nodes_[elements_side_[0]]], (elements_side_[0] < element_L.faces_.size() - 1) ? nodes[element_L.nodes_[elements_side_[0] + 1]] : nodes[element_L.nodes_[0]]},
        std::array<SEM::Entities::Vec2<deviceFloat>, 2>{nodes[element_R.nodes_[elements_side_[1]]], (elements_side_[1] < element_R.faces_.size() - 1) ? nodes[element_R.nodes_[elements_side_[1] + 1]] : nodes[element_R.nodes_[0]]}
    };

    const std::array<SEM::Entities::Vec2<deviceFloat>, 2> elements_delta {
        elements_nodes[0][1] - elements_nodes[0][0], 
        elements_nodes[1][1] - elements_nodes[1][0]
    };

    const SEM::Entities::Vec2<deviceFloat> face_delta = points[1] - points[0];

    const std::array<SEM::Entities::Vec2<deviceFloat>, 2> edge_delta {
        points[0] - elements_nodes[0][0], 
        points[1] - elements_nodes[1][0]
    };

    offset_ = {edge_delta[0].magnitude()/elements_delta[0].magnitude(), edge_delta[1].magnitude()/elements_delta[1].magnitude()};
    scale_ = {face_delta.magnitude()/elements_delta[0].magnitude(), face_delta.magnitude()/elements_delta[1].magnitude()};
}
