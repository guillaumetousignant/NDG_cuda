#include "entities/Element2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include <cmath>

constexpr deviceFloat pi = 3.14159265358979323846;

__device__ 
SEM::Entities::Element2D_t::Element2D_t(int N, std::array<size_t, 4> faces, std::array<size_t, 4> nodes) : 
        N_(N),
        faces_{faces},
        nodes_{nodes},
        delta_xy_{0.0, 0.0},
        phi_extrapolated_{0.0, 0.0, 0.0, 0.0},
        phi_prime_extrapolated_{0.0, 0.0, 0.0, 0.0},
        p_((N_ + 1) * (N_ + 1)),
        u_((N_ + 1) * (N_ + 1)),
        v_((N_ + 1) * (N_ + 1)),
        G_p_((N_ + 1) * (N_ + 1)),
        G_u_((N_ + 1) * (N_ + 1)),
        G_v_((N_ + 1) * (N_ + 1)),
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {}

__host__ __device__
SEM::Entities::Element2D_t::Element2D_t() :
        N_(0),
        faces_{0, 0, 0, 0},
        nodes_{0, 0, 0, 0},
        delta_xy_{0.0, 0.0},
        phi_extrapolated_{0.0, 0.0, 0.0, 0.0},
        phi_prime_extrapolated_{0.0, 0.0, 0.0, 0.0},
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {};

// Algorithm 61
__device__
auto SEM::Entities::Element2D_t::interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void {
    
}

// Algorithm 61
__device__
auto SEM::Entities::Element2D_t::interpolate_q_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void {
    
}

template __device__ auto SEM::Entities::Element2D_t::estimate_error<SEM::Polynomials::ChebyshevPolynomial_t>(const deviceFloat* nodes, const deviceFloat* weights) -> void;
template __device__ auto SEM::Entities::Element2D_t::estimate_error<SEM::Polynomials::LegendrePolynomial_t>(const deviceFloat* nodes, const deviceFloat* weights) -> void;

template<typename Polynomial>
__device__
auto SEM::Entities::Element2D_t::estimate_error<Polynomial>(const deviceFloat* nodes, const deviceFloat* weights) -> void {
    
}

__device__
auto SEM::Entities::Element2D_t::exponential_decay() -> deviceFloat {
    return 0.0;
}

__device__
auto SEM::Entities::Element2D_t::interpolate_from(const SEM::Entities::Element2D_t& other, const deviceFloat* nodes, const deviceFloat* barycentric_weights) -> void {

}
