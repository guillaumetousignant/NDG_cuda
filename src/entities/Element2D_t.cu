#include "entities/Element2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "functions/quad_map.cuh"
#include <cmath>

using SEM::Entities::cuda_vector;

constexpr deviceFloat pi = 3.14159265358979323846;

__device__ 
SEM::Entities::Element2D_t::Element2D_t(int N, std::array<cuda_vector<size_t>, 4> faces, std::array<size_t, 4> nodes) : 
        N_(N),
        faces_{faces},
        nodes_{nodes},
        delta_xy_min_{0.0},
        p_((N_ + 1) * (N_ + 1)),
        u_((N_ + 1) * (N_ + 1)),
        v_((N_ + 1) * (N_ + 1)),
        G_p_((N_ + 1) * (N_ + 1)),
        G_u_((N_ + 1) * (N_ + 1)),
        G_v_((N_ + 1) * (N_ + 1)),
        p_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        u_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        v_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {}

__host__ __device__
SEM::Entities::Element2D_t::Element2D_t() :
        N_(0),
        faces_{},
        nodes_{0, 0, 0, 0},
        delta_xy_min_{0.0},
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
auto SEM::Entities::Element2D_t::estimate_error<Polynomial>(const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void {
    
}

__device__
auto SEM::Entities::Element2D_t::exponential_decay() -> deviceFloat {
    return 0.0;
}

__device__
auto SEM::Entities::Element2D_t::interpolate_from(const SEM::Entities::Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void {

}

__device__
auto SEM::Entities::Element2D_t::interpolate_solution(size_t N_interpolation_points, const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    for (size_t i = 0; i < N_interpolation_points; ++i) {
        for (size_t j = 0; j < N_interpolation_points; ++j) {
            // x and y
            const Vec2<deviceFloat> coordinates {static_cast<deviceFloat>(i)/static_cast<deviceFloat>(N_interpolation_points - 1) * 2 - 1, static_cast<deviceFloat>(j)/static_cast<deviceFloat>(N_interpolation_points - 1) * 2 - 1};
            const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points);

            x[i * N_interpolation_points + j] = global_coordinates.x();
            y[i * N_interpolation_points + j] = global_coordinates.y();

            // Pressure, u, and v
            p[i * N_interpolation_points + j] = 0.0;
            u[i * N_interpolation_points + j] = 0.0;
            v[i * N_interpolation_points + j] = 0.0;
            for (int m = 0; m <= N_; ++m) {
                for (int n = 0; n <= N_; ++n) {
                    p[i * N_interpolation_points + j] += p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u[i * N_interpolation_points + j] += u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v[i * N_interpolation_points + j] += v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                }
            }
        }
    }
}

__device__
auto SEM::Entities::Element2D_t::allocate_storage() -> void {
    faces_ = {cuda_vector<size_t>(1),
              cuda_vector<size_t>(1),
              cuda_vector<size_t>(1),
              cuda_vector<size_t>(1)};

    p_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    u_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    v_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));

    G_p_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    G_u_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    G_v_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));

    p_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1)};
    u_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1)};
    v_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1)};
}
