#include "entities/Face2D_t.cuh"
#include <utility>

__device__ 
SEM::Device::Entities::Face2D_t::Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements, std::array<size_t, 2> elements_side) : 
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
        p_{N_ + 1, N_ + 1},
        u_{N_ + 1, N_ + 1},
        v_{N_ + 1, N_ + 1},
        p_flux_{N_ + 1},
        u_flux_{N_ + 1},
        v_flux_{N_ + 1} {}

__host__ __device__
SEM::Device::Entities::Face2D_t::Face2D_t() :
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

__device__
auto SEM::Device::Entities::Face2D_t::allocate_storage() -> void {
    p_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    u_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    v_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    p_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    u_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    v_flux_ = cuda_vector<deviceFloat>(N_ + 1);
}

__device__
auto SEM::Device::Entities::Face2D_t::resize_storage(int N) -> void {
    N_ = N;
    p_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    u_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    v_ = {cuda_vector<deviceFloat>(N_ + 1), cuda_vector<deviceFloat>(N_ + 1)};
    p_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    u_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    v_flux_ = cuda_vector<deviceFloat>(N_ + 1);
}

__device__
auto SEM::Device::Entities::Face2D_t::clear_storage() -> void {
    p_[0].data_ = nullptr;
    p_[1].data_ = nullptr;
    u_[0].data_ = nullptr;
    u_[1].data_ = nullptr;
    v_[0].data_ = nullptr;
    v_[1].data_ = nullptr;
    p_flux_.data_ = nullptr;
    u_flux_.data_ = nullptr;
    v_flux_.data_ = nullptr;
}

__device__
auto SEM::Device::Entities::Face2D_t::compute_geometry(const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2>& elements_centres, const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2>& nodes, const std::array<std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2>, 2>& element_nodes) -> void {
    tangent_ = nodes[1] - nodes[0]; 
    length_ = tangent_.magnitude();
    tangent_ /= length_; // CHECK should be normalized or not?
    normal_ = SEM::Device::Entities::Vec2<deviceFloat>(tangent_.y(), -tangent_.x());     

    const SEM::Device::Entities::Vec2<deviceFloat> left_element_tangent = element_nodes[0][1] - element_nodes[0][0];
    const deviceFloat sign = std::copysign(deviceFloat{1}, tangent_.dot(left_element_tangent)); // CHECK doesn't work for zero-length edges, if it is an issue do the thing with centres
    normal_ *= sign;
    tangent_ *= sign;

    const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2> elements_delta {
        element_nodes[0][1] - element_nodes[0][0], 
        element_nodes[1][1] - element_nodes[1][0]
    };

    const SEM::Device::Entities::Vec2<deviceFloat> face_delta = nodes[1] - nodes[0];

    const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 2> edge_delta {
        nodes[0] - element_nodes[0][0], 
        nodes[1] - element_nodes[1][0]
    };

    offset_ = {edge_delta[0].magnitude()/elements_delta[0].magnitude(), edge_delta[1].magnitude()/elements_delta[1].magnitude()};
    scale_ = {face_delta.magnitude()/elements_delta[0].magnitude(), face_delta.magnitude()/elements_delta[1].magnitude()};
}
