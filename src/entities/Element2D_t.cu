#include "entities/Element2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "functions/quad_map.cuh"
#include "helpers/constants.h"
#include <cmath>
#include <limits>

using SEM::Entities::cuda_vector;

__device__ 
SEM::Entities::Element2D_t::Element2D_t(int N, std::array<cuda_vector<size_t>, 4> faces, std::array<size_t, 4> nodes) : 
        N_(N),
        faces_{faces},
        nodes_{nodes},
        delta_xy_min_{0.0},
        dxi_dx_((N_ + 1) * (N_ + 1)),
        deta_dx_((N_ + 1) * (N_ + 1)),
        dxi_dy_((N_ + 1) * (N_ + 1)),
        deta_dy_((N_ + 1) * (N_ + 1)),
        jacobian_((N_ + 1) * (N_ + 1)),
        scaling_factor_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        p_((N_ + 1) * (N_ + 1)),
        u_((N_ + 1) * (N_ + 1)),
        v_((N_ + 1) * (N_ + 1)),
        G_p_((N_ + 1) * (N_ + 1)),
        G_u_((N_ + 1) * (N_ + 1)),
        G_v_((N_ + 1) * (N_ + 1)),
        p_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        u_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        v_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        p_flux_(N_ + 1),
        u_flux_(N_ + 1),
        v_flux_(N_ + 1),
        p_flux_derivative_(N_ + 1),
        u_flux_derivative_(N_ + 1),
        v_flux_derivative_(N_ + 1),
        p_flux_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        u_flux_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        v_flux_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        p_intermediate_((N_ + 1) * (N_ + 1)),
        u_intermediate_((N_ + 1) * (N_ + 1)),
        v_intermediate_((N_ + 1) * (N_ + 1)),
        spectrum_(N_ + 1),
        refine_(false),
        coarsen_(false),
        p_error_(0.0),
        u_error_(0.0),
        v_error_(0.0) {}

__host__ __device__
SEM::Entities::Element2D_t::Element2D_t() :
        N_(0),
        faces_{},
        nodes_{0, 0, 0, 0},
        delta_xy_min_{0.0},
        scaling_factor_{},
        p_extrapolated_{},
        u_extrapolated_{},
        v_extrapolated_{},
        p_flux_extrapolated_{},
        u_flux_extrapolated_{},
        v_flux_extrapolated_{},
        refine_(false),
        coarsen_(false),
        p_error_(0.0),
        u_error_(0.0),
        v_error_(0.0) {};

// Algorithm 61
__device__
auto SEM::Entities::Element2D_t::interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void {
    const int offset_1D = N_ * (N_ + 1) /2;

    for (int i = 0; i <= N_; ++i) {
        p_extrapolated_[0][i] = 0.0;
        p_extrapolated_[2][N_ - i] = 0.0;
        p_extrapolated_[1][i] = 0.0;
        p_extrapolated_[3][N_ - i] = 0.0;

        u_extrapolated_[0][i] = 0.0;
        u_extrapolated_[2][N_ - i] = 0.0;
        u_extrapolated_[1][i] = 0.0;
        u_extrapolated_[3][N_ - i] = 0.0;

        v_extrapolated_[0][i] = 0.0;
        v_extrapolated_[2][N_ - i] = 0.0;
        v_extrapolated_[1][i] = 0.0;
        v_extrapolated_[3][N_ - i] = 0.0;
        
        // For the boundaries, the numbering increases from the first node to the second. 
        // Inside the element, the ksi and eta coordinates increase from left to right, bottom to top.
        // This means that there is an inconsistency on the top and left edges, and the numbering has to be reversed.
        // This way, the projection from the element edge to the face(s) can always be done in the same way.
        // The same process has to be done when adding back to the solution, but I bet I'll forget.
        for (int j = 0; j <= N_; ++j) {
            p_extrapolated_[0][i] += lagrange_interpolant_minus[offset_1D + j] * p_[i * (N_ + 1) + j];
            p_extrapolated_[2][N_ - i] += lagrange_interpolant_plus[offset_1D + j] * p_[i * (N_ + 1) + j];
            p_extrapolated_[1][i] += lagrange_interpolant_plus[offset_1D + j] * p_[j * (N_ + 1) + i];
            p_extrapolated_[3][N_ - i] += lagrange_interpolant_minus[offset_1D + j] * p_[j * (N_ + 1) + i];

            u_extrapolated_[0][i] += lagrange_interpolant_minus[offset_1D + j] * u_[i * (N_ + 1) + j];
            u_extrapolated_[2][N_ - i] += lagrange_interpolant_plus[offset_1D + j] * u_[i * (N_ + 1) + j];
            u_extrapolated_[1][i] += lagrange_interpolant_plus[offset_1D + j] * u_[j * (N_ + 1) + i];
            u_extrapolated_[3][N_ - i] += lagrange_interpolant_minus[offset_1D + j] * u_[j * (N_ + 1) + i];

            v_extrapolated_[0][i] += lagrange_interpolant_minus[offset_1D + j] * v_[i * (N_ + 1) + j];
            v_extrapolated_[2][N_ - i] += lagrange_interpolant_plus[offset_1D + j] * v_[i * (N_ + 1) + j];
            v_extrapolated_[1][i] += lagrange_interpolant_plus[offset_1D + j] * v_[j * (N_ + 1) + i];
            v_extrapolated_[3][N_ - i] += lagrange_interpolant_minus[offset_1D + j] * v_[j * (N_ + 1) + i];
        }
    }
}

// Algorithm 61
__device__
auto SEM::Entities::Element2D_t::interpolate_q_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void {
    printf("Warning, SEM::Entities::Element2D_t::interpolate_q_to_boundaries is not implemented.\n");
}

template __device__ auto SEM::Entities::Element2D_t::estimate_error<SEM::Polynomials::ChebyshevPolynomial_t>(const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;
template __device__ auto SEM::Entities::Element2D_t::estimate_error<SEM::Polynomials::LegendrePolynomial_t>(const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

template<typename Polynomial>
__device__
auto SEM::Entities::Element2D_t::estimate_error<Polynomial>(const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void {
    const int offset_1D = N_ * (N_ + 1) /2;
    const int n_points_least_squares = min(N_ + 1, SEM::Constants::n_points_least_squares_max); // Number of points to use for thew least squares reduction, but don't go above N.

    refine_ = false;
    coarsen_ = true;

    // Pressure
    for (int node_index = 0; node_index < n_points_least_squares; ++node_index) {
        spectrum_[node_index] = 0.0;
        const deviceFloat p = N_ + node_index + 1 - n_points_least_squares;

        // x direction
        for (int i = 0; i <= p; ++i) {
            deviceFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const deviceFloat L_N_x = Polynomial::polynomial(i, polynomial_nodes[offset_1D + k]);

                for (int l = 0; l <= N_; ++l) {
                    const deviceFloat L_N_y = Polynomial::polynomial(p, polynomial_nodes[offset_1D + l]);

                    local_spectrum += (2 * i + 1) * (2 * p + 1) * static_cast<deviceFloat>(0.25) *
                                      p_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[offset_1D + k] * weights[offset_1D + l];
                }

            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }

        // y direction
        for (int j = 0; j < p; ++j) { // No need to include the last point here
            deviceFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const deviceFloat L_N_x = Polynomial::polynomial(p, polynomial_nodes[offset_1D + k]);

                for (int l = 0; l <= N_; ++l) {
                    const deviceFloat L_N_y = Polynomial::polynomial(j, polynomial_nodes[offset_1D + l]);

                    local_spectrum += (2 * j + 1) * (2 * (double)p + 1.0) * static_cast<deviceFloat>(0.25) *
                                      p_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[offset_1D + k] * weights[offset_1D + l];
                }
            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }
    }

    const auto [C_p, sigma_p] = exponential_decay(n_points_least_squares);

    // Sum of error
    p_error_ = std::sqrt(spectrum_[n_points_least_squares - 1] * spectrum_[n_points_least_squares - 1] // Why this part?
                         + C_p * C_p * 0.5 / sigma_p * std::exp(-2 * sigma_p * (N_ + 1)));

    if(p_error_ > SEM::Constants::tolerance_min) {	// need refinement
        refine_ = true;
    }
    if(p_error_ > SEM::Constants::tolerance_max) {	// need coarsening
        coarsen_ = false;
    }

    // Velocity x
    for (int node_index = 0; node_index < n_points_least_squares; ++node_index) {
        spectrum_[node_index] = 0.0;
        const deviceFloat p = N_ + node_index + 1 - n_points_least_squares;

        // x direction
        for (int i = 0; i <= p; ++i) {
            deviceFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const deviceFloat L_N_x = Polynomial::polynomial(i, polynomial_nodes[offset_1D + k]);

                for (int l = 0; l <= N_; ++l) {
                    const deviceFloat L_N_y = Polynomial::polynomial(p, polynomial_nodes[offset_1D + l]);

                    local_spectrum += (2 * i + 1) * (2 * p + 1) * static_cast<deviceFloat>(0.25) *
                                      u_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[offset_1D + k] * weights[offset_1D + l];
                }

            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }

        // y direction
        for (int j = 0; j < p; ++j) { // No need to include the last point here
            deviceFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const deviceFloat L_N_x = Polynomial::polynomial(p, polynomial_nodes[offset_1D + k]);

                for (int l = 0; l <= N_; ++l) {
                    const deviceFloat L_N_y = Polynomial::polynomial(j, polynomial_nodes[offset_1D + l]);

                    local_spectrum += (2 * j + 1) * (2 * (double)p + 1.0) * static_cast<deviceFloat>(0.25) *
                                      u_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[offset_1D + k] * weights[offset_1D + l];
                }
            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }
    }

    const auto [C_u, sigma_u] = exponential_decay(n_points_least_squares);

    // Sum of error
    u_error_ = std::sqrt(spectrum_[n_points_least_squares - 1] * spectrum_[n_points_least_squares - 1] // Why this part?
                         + C_u * C_u * 0.5 / sigma_u * std::exp(-2 * sigma_u * (N_ + 1)));

    if(u_error_ > SEM::Constants::tolerance_min) {	// need refinement
        refine_ = true;
    }
    if(u_error_ > SEM::Constants::tolerance_max) {	// need coarsening
        coarsen_ = false;
    }

    // Velocity y
    for (int node_index = 0; node_index < n_points_least_squares; ++node_index) {
        spectrum_[node_index] = 0.0;
        const deviceFloat p = N_ + node_index + 1 - n_points_least_squares;

        // x direction
        for (int i = 0; i <= p; ++i) {
            deviceFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const deviceFloat L_N_x = Polynomial::polynomial(i, polynomial_nodes[offset_1D + k]);

                for (int l = 0; l <= N_; ++l) {
                    const deviceFloat L_N_y = Polynomial::polynomial(p, polynomial_nodes[offset_1D + l]);

                    local_spectrum += (2 * i + 1) * (2 * p + 1) * static_cast<deviceFloat>(0.25) *
                                      v_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[offset_1D + k] * weights[offset_1D + l];
                }

            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }

        // y direction
        for (int j = 0; j < p; ++j) { // No need to include the last point here
            deviceFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const deviceFloat L_N_x = Polynomial::polynomial(p, polynomial_nodes[offset_1D + k]);

                for (int l = 0; l <= N_; ++l) {
                    const deviceFloat L_N_y = Polynomial::polynomial(j, polynomial_nodes[offset_1D + l]);

                    local_spectrum += (2 * j + 1) * (2 * (double)p + 1.0) * static_cast<deviceFloat>(0.25) *
                                      v_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[offset_1D + k] * weights[offset_1D + l];
                }
            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }
    }

    const auto [C_v, sigma_v] = exponential_decay(n_points_least_squares);

    // Sum of error
    v_error_ = std::sqrt(spectrum_[n_points_least_squares - 1] * spectrum_[n_points_least_squares - 1] // Why this part?
                         + C_v * C_v * 0.5 / sigma_v * std::exp(-2 * sigma_v * (N_ + 1)));

    if(v_error_ > SEM::Constants::tolerance_min) {	// need refinement
        refine_ = true;
    }
    if(v_error_ > SEM::Constants::tolerance_max) {	// need coarsening
        coarsen_ = false;
    }
}

__device__
auto SEM::Entities::Element2D_t::exponential_decay(int n_points_least_squares) -> std::array<deviceFloat, 2> {
    deviceFloat x_avg = 0.0;
    deviceFloat y_avg = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        x_avg += N_ + i + 1 - n_points_least_squares;
        y_avg += std::log(spectrum_[i]);
    }

    x_avg /= n_points_least_squares;
    y_avg /= n_points_least_squares;

    deviceFloat numerator = 0.0;
    deviceFloat denominator = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        const int p = N_ + i + 1 - n_points_least_squares;
        numerator += (p - x_avg) * (std::log(spectrum_[i]) - y_avg);
        denominator += (p - x_avg) * (p - x_avg);
    }

    const deviceFloat sigma = numerator/denominator;
    const deviceFloat C = std::exp(y_avg - sigma * x_avg);
    return {C, std::abs(sigma)};
}

__device__
auto SEM::Entities::Element2D_t::interpolate_from(const SEM::Entities::Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void {
    printf("Warning, SEM::Entities::Element2D_t::interpolate_from is not implemented.\n");
}

__device__
auto SEM::Entities::Element2D_t::interpolate_solution(size_t N_interpolation_points, const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt) const -> void {
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
            dp_dt[i * N_interpolation_points + j] = 0.0;
            du_dt[i * N_interpolation_points + j] = 0.0;
            dv_dt[i * N_interpolation_points + j] = 0.0;
            for (int m = 0; m <= N_; ++m) {
                for (int n = 0; n <= N_; ++n) {
                    p[i * N_interpolation_points + j] += p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u[i * N_interpolation_points + j] += u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v[i * N_interpolation_points + j] += v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    dp_dt[i * N_interpolation_points + j] += G_p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    du_dt[i * N_interpolation_points + j] += G_u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    dv_dt[i * N_interpolation_points + j] += G_v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
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

    dxi_dx_  = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    deta_dx_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    dxi_dy_  = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    deta_dy_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    jacobian_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    scaling_factor_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(N_ + 1)};

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

    p_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    u_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    v_flux_ = cuda_vector<deviceFloat>(N_ + 1);
    p_flux_derivative_ = cuda_vector<deviceFloat>(N_ + 1);
    u_flux_derivative_ = cuda_vector<deviceFloat>(N_ + 1);
    v_flux_derivative_ = cuda_vector<deviceFloat>(N_ + 1);

    p_flux_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1)};
    u_flux_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1)};
    v_flux_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1),
                            cuda_vector<deviceFloat>(N_ + 1)};
    
    p_intermediate_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    u_intermediate_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    v_intermediate_ = cuda_vector<deviceFloat>((N_ + 1) * (N_ + 1));
    spectrum_ = cuda_vector<deviceFloat>(N_ + 1);
}

__device__
auto SEM::Entities::Element2D_t::allocate_boundary_storage() -> void {
    faces_ = {cuda_vector<size_t>(1),
              cuda_vector<size_t>(1),
              cuda_vector<size_t>(1),
              cuda_vector<size_t>(1)};

    p_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>()};
    u_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>()};
    v_extrapolated_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>()};
}
