#include "entities/Element2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "functions/quad_map.cuh"
#include "functions/inverse_quad_map.cuh"
#include "functions/quad_metrics.cuh"
#include "functions/analytical_solution.cuh"
#include "helpers/constants.h"
#include <cmath>
#include <limits>
#include <cstddef>

using SEM::Entities::cuda_vector;
using namespace SEM::Hilbert;

__device__ 
SEM::Entities::Element2D_t::Element2D_t(int N, int split_level, Hilbert::Status status, int rotation, const std::array<cuda_vector<size_t>, 4>& faces, std::array<size_t, 4> nodes) : 
        N_{N},
        faces_{faces},
        nodes_{nodes},
        status_{status},
        delta_xy_min_{0.0},
        center_{0.0, 0.0},
        dxi_dx_{(N_ + 1) * (N_ + 1)},
        deta_dx_{(N_ + 1) * (N_ + 1)},
        dxi_dy_{(N_ + 1) * (N_ + 1)},
        deta_dy_{(N_ + 1) * (N_ + 1)},
        jacobian_{(N_ + 1) * (N_ + 1)},
        scaling_factor_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        rotation_{rotation},
        p_{(N_ + 1) * (N_ + 1)},
        u_{(N_ + 1) * (N_ + 1)},
        v_{(N_ + 1) * (N_ + 1)},
        G_p_{(N_ + 1) * (N_ + 1)},
        G_u_{(N_ + 1) * (N_ + 1)},
        G_v_{(N_ + 1) * (N_ + 1)},
        p_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        u_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        v_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        p_flux_{N_ + 1},
        u_flux_{N_ + 1},
        v_flux_{N_ + 1},
        p_flux_derivative_{N_ + 1},
        u_flux_derivative_{N_ + 1},
        v_flux_derivative_{N_ + 1},
        p_flux_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        u_flux_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        v_flux_extrapolated_{N_ + 1, N_ + 1, N_ + 1, N_ + 1},
        p_intermediate_{(N_ + 1) * (N_ + 1)},
        u_intermediate_{(N_ + 1) * (N_ + 1)},
        v_intermediate_{(N_ + 1) * (N_ + 1)},
        spectrum_{N_ + 1},
        refine_{false},
        coarsen_{false},
        p_error_{0.0},
        u_error_{0.0},
        v_error_{0.0},
        p_sigma_{0.0},
        u_sigma_{0.0},
        v_sigma_{0.0},
        split_level_{split_level},
        additional_nodes_{false, false, false, false} {}

__host__ __device__
SEM::Entities::Element2D_t::Element2D_t() :
        N_{0},
        faces_{},
        nodes_{0, 0, 0, 0},
        status_{Hilbert::Status::A},
        delta_xy_min_{0.0},
        center_{0.0, 0.0},
        scaling_factor_{},
        rotation_{0},
        p_extrapolated_{},
        u_extrapolated_{},
        v_extrapolated_{},
        p_flux_extrapolated_{},
        u_flux_extrapolated_{},
        v_flux_extrapolated_{},
        refine_{false},
        coarsen_{false},
        p_error_{0.0},
        u_error_{0.0},
        v_error_{0.0},
        p_sigma_{0.0},
        u_sigma_{0.0},
        v_sigma_{0.0},
        split_level_{0},
        additional_nodes_{false, false, false, false} {};

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

template __device__ auto SEM::Entities::Element2D_t::estimate_error<SEM::Polynomials::ChebyshevPolynomial_t>(deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;
template __device__ auto SEM::Entities::Element2D_t::estimate_error<SEM::Polynomials::LegendrePolynomial_t>(deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

template<typename Polynomial>
__device__
auto SEM::Entities::Element2D_t::estimate_error<Polynomial>(deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void {
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
    p_sigma_ = sigma_p;

    // Sum of error
    p_error_ = std::sqrt(spectrum_[n_points_least_squares - 1] * spectrum_[n_points_least_squares - 1] // Why this part?
                         + C_p * C_p * 0.5 / sigma_p * std::exp(-2 * sigma_p * (N_ + 1)));

    if(p_error_ > tolerance_min) {	// need refinement
        refine_ = true;
    }
    if(p_error_ > tolerance_max) {	// need coarsening
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
    u_sigma_ = sigma_u;

    // Sum of error
    u_error_ = std::sqrt(spectrum_[n_points_least_squares - 1] * spectrum_[n_points_least_squares - 1] // Why this part?
                         + C_u * C_u * 0.5 / sigma_u * std::exp(-2 * sigma_u * (N_ + 1)));

    if(u_error_ > tolerance_min) {	// need refinement
        refine_ = true;
    }
    if(u_error_ > tolerance_max) {	// need coarsening
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
    v_sigma_ = sigma_v;

    // Sum of error
    v_error_ = std::sqrt(spectrum_[n_points_least_squares - 1] * spectrum_[n_points_least_squares - 1] // Why this part?
                         + C_v * C_v * 0.5 / sigma_v * std::exp(-2 * sigma_v * (N_ + 1)));

    if(v_error_ > tolerance_min) {	// need refinement
        refine_ = true;
    }
    if(v_error_ > tolerance_max) {	// need coarsening
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
auto SEM::Entities::Element2D_t::interpolate_from(const std::array<Vec2<deviceFloat>, 4>& points, const std::array<Vec2<deviceFloat>, 4>& points_other, const SEM::Entities::Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void {
    const int offset_1D = N_ * (N_ + 1) /2;
    const int offset_1D_other = other.N_ * (other.N_ + 1) /2;

    for (int i = 0; i <= N_; ++i) {
        for (int j = 0; j <= N_; ++j) {
            // x and y
            const Vec2<deviceFloat> local_coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
            const Vec2<deviceFloat> global_coordinates = SEM::quad_map(local_coordinates, points);
            const Vec2<deviceFloat> local_coordinates_in_other = SEM::inverse_quad_map(global_coordinates, points_other);

            int row_found = -1;
            int column_found = -1;
            for (int m = 0; m <= other.N_; ++m) {
                if (SEM::Entities::Element2D_t::almost_equal(local_coordinates_in_other.x(), polynomial_nodes[offset_1D_other + m])) {
                    column_found = m;
                }
                if (SEM::Entities::Element2D_t::almost_equal(local_coordinates_in_other.y(), polynomial_nodes[offset_1D_other + m])) {
                    row_found = m;
                }
            }

            // A point fits exactly
            if (row_found != -1 && column_found != -1) {
                p_[i * (N_ + 1) + j] = other.p_[column_found * (other.N_ + 1) + row_found];
                u_[i * (N_ + 1) + j] = other.u_[column_found * (other.N_ + 1) + row_found];
                v_[i * (N_ + 1) + j] = other.v_[column_found * (other.N_ + 1) + row_found]; 
                G_p_[i * (N_ + 1) + j] = other.G_p_[column_found * (other.N_ + 1) + row_found];
                G_u_[i * (N_ + 1) + j] = other.G_u_[column_found * (other.N_ + 1) + row_found];
                G_v_[i * (N_ + 1) + j] = other.G_v_[column_found * (other.N_ + 1) + row_found]; 
            }
            // A row fits exactly
            else if (row_found != -1) {
                deviceFloat p_numerator = 0.0;
                deviceFloat u_numerator = 0.0;
                deviceFloat v_numerator = 0.0;
                deviceFloat G_p_numerator = 0.0;
                deviceFloat G_u_numerator = 0.0;
                deviceFloat G_v_numerator = 0.0;
                deviceFloat denominator = 0.0;

                for (int m = 0; m <= other.N_; ++m) {
                    const deviceFloat t = barycentric_weights[offset_1D_other + m]/(local_coordinates_in_other.x() - polynomial_nodes[offset_1D_other + m]);
                    p_numerator += t * other.p_[m * (other.N_ + 1) + row_found];
                    u_numerator += t * other.u_[m * (other.N_ + 1) + row_found];
                    v_numerator += t * other.v_[m * (other.N_ + 1) + row_found];
                    G_p_numerator += t * other.G_p_[m * (other.N_ + 1) + row_found];
                    G_u_numerator += t * other.G_u_[m * (other.N_ + 1) + row_found];
                    G_v_numerator += t * other.G_v_[m * (other.N_ + 1) + row_found];
                    denominator += t;
                }

                p_[i * (N_ + 1) + j] = p_numerator/denominator;
                u_[i * (N_ + 1) + j] = u_numerator/denominator;
                v_[i * (N_ + 1) + j] = v_numerator/denominator;
                G_p_[i * (N_ + 1) + j] = G_p_numerator/denominator;
                G_u_[i * (N_ + 1) + j] = G_u_numerator/denominator;
                G_v_[i * (N_ + 1) + j] = G_v_numerator/denominator;
            }
            // A column fits exactly
            else if (column_found != -1) {
                deviceFloat p_numerator = 0.0;
                deviceFloat u_numerator = 0.0;
                deviceFloat v_numerator = 0.0;
                deviceFloat G_p_numerator = 0.0;
                deviceFloat G_u_numerator = 0.0;
                deviceFloat G_v_numerator = 0.0;
                deviceFloat denominator = 0.0;

                for (int n = 0; n <= other.N_; ++n) {
                    const deviceFloat t = barycentric_weights[offset_1D_other + n]/(local_coordinates_in_other.y() - polynomial_nodes[offset_1D_other + n]);
                    p_numerator += t * other.p_[column_found * (other.N_ + 1) + n];
                    u_numerator += t * other.u_[column_found * (other.N_ + 1) + n];
                    v_numerator += t * other.v_[column_found * (other.N_ + 1) + n];
                    G_p_numerator += t * other.G_p_[column_found * (other.N_ + 1) + n];
                    G_u_numerator += t * other.G_u_[column_found * (other.N_ + 1) + n];
                    G_v_numerator += t * other.G_v_[column_found * (other.N_ + 1) + n];
                    denominator += t;
                }
                
                p_[i * (N_ + 1) + j] = p_numerator/denominator;
                u_[i * (N_ + 1) + j] = u_numerator/denominator;
                v_[i * (N_ + 1) + j] = v_numerator/denominator;
                G_p_[i * (N_ + 1) + j] = G_p_numerator/denominator;
                G_u_[i * (N_ + 1) + j] = G_u_numerator/denominator;
                G_v_[i * (N_ + 1) + j] = G_v_numerator/denominator;
            }
            // Complete interpolation
            else {
                p_[i * (N_ + 1) + j] = 0;
                u_[i * (N_ + 1) + j] = 0;
                v_[i * (N_ + 1) + j] = 0;
                G_p_[i * (N_ + 1) + j] = 0;
                G_u_[i * (N_ + 1) + j] = 0;
                G_v_[i * (N_ + 1) + j] = 0;

                deviceFloat denominator_x = 0.0;
                deviceFloat denominator_y = 0.0;

                for (int k = 0; k <= other.N_; ++k) {
                    denominator_x += barycentric_weights[offset_1D_other + k]/(local_coordinates_in_other.x() - polynomial_nodes[offset_1D_other + k]);
                    denominator_y += barycentric_weights[offset_1D_other + k]/(local_coordinates_in_other.y() - polynomial_nodes[offset_1D_other + k]);
                }

                for (int m = 0; m <= other.N_; ++m) {
                    const deviceFloat t_x = barycentric_weights[offset_1D_other + m]/((local_coordinates_in_other.x() - polynomial_nodes[offset_1D_other + m]) * denominator_x);
                    for (int n = 0; n <= other.N_; ++n) {
                        const deviceFloat t_y = barycentric_weights[offset_1D_other + n]/((local_coordinates_in_other.y() - polynomial_nodes[offset_1D_other + n]) * denominator_y);

                        p_[i * (N_ + 1) + j] += other.p_[m * (other.N_ + 1) + n] * t_x * t_y;
                        u_[i * (N_ + 1) + j] += other.u_[m * (other.N_ + 1) + n] * t_x * t_y;
                        v_[i * (N_ + 1) + j] += other.v_[m * (other.N_ + 1) + n] * t_x * t_y;
                        G_p_[i * (N_ + 1) + j] += other.G_p_[m * (other.N_ + 1) + n] * t_x * t_y;
                        G_u_[i * (N_ + 1) + j] += other.G_u_[m * (other.N_ + 1) + n] * t_x * t_y;
                        G_v_[i * (N_ + 1) + j] += other.G_v_[m * (other.N_ + 1) + n] * t_x * t_y;
                    }
                }
            }  
        }
    }
}

__device__
auto SEM::Entities::Element2D_t::interpolate_from(const SEM::Entities::Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void {
    const int offset_1D = N_ * (N_ + 1) /2;
    const int offset_1D_other = other.N_ * (other.N_ + 1) /2;

    for (int i = 0; i <= N_; ++i) {
        for (int j = 0; j <= N_; ++j) {
            // x and y
            const Vec2<deviceFloat> local_coordinates_in_other = {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};

            int row_found = -1;
            int column_found = -1;
            for (int m = 0; m <= other.N_; ++m) {
                if (SEM::Entities::Element2D_t::almost_equal(local_coordinates_in_other.x(), polynomial_nodes[offset_1D_other + m])) {
                    column_found = m;
                }
                if (SEM::Entities::Element2D_t::almost_equal(local_coordinates_in_other.y(), polynomial_nodes[offset_1D_other + m])) {
                    row_found = m;
                }
            }

            // A point fits exactly
            if (row_found != -1 && column_found != -1) {
                p_[i * (N_ + 1) + j] = other.p_[column_found * (other.N_ + 1) + row_found];
                u_[i * (N_ + 1) + j] = other.u_[column_found * (other.N_ + 1) + row_found];
                v_[i * (N_ + 1) + j] = other.v_[column_found * (other.N_ + 1) + row_found]; 
                G_p_[i * (N_ + 1) + j] = other.G_p_[column_found * (other.N_ + 1) + row_found];
                G_u_[i * (N_ + 1) + j] = other.G_u_[column_found * (other.N_ + 1) + row_found];
                G_v_[i * (N_ + 1) + j] = other.G_v_[column_found * (other.N_ + 1) + row_found]; 
            }
            // A row fits exactly
            else if (row_found != -1) {
                deviceFloat p_numerator = 0.0;
                deviceFloat u_numerator = 0.0;
                deviceFloat v_numerator = 0.0;
                deviceFloat G_p_numerator = 0.0;
                deviceFloat G_u_numerator = 0.0;
                deviceFloat G_v_numerator = 0.0;
                deviceFloat denominator = 0.0;

                for (int m = 0; m <= other.N_; ++m) {
                    const deviceFloat t = barycentric_weights[offset_1D_other + m]/(local_coordinates_in_other.x() - polynomial_nodes[offset_1D_other + m]);
                    p_numerator += t * other.p_[m * (other.N_ + 1) + row_found];
                    u_numerator += t * other.u_[m * (other.N_ + 1) + row_found];
                    v_numerator += t * other.v_[m * (other.N_ + 1) + row_found];
                    G_p_numerator += t * other.G_p_[m * (other.N_ + 1) + row_found];
                    G_u_numerator += t * other.G_u_[m * (other.N_ + 1) + row_found];
                    G_v_numerator += t * other.G_v_[m * (other.N_ + 1) + row_found];
                    denominator += t;
                }

                p_[i * (N_ + 1) + j] = p_numerator/denominator;
                u_[i * (N_ + 1) + j] = u_numerator/denominator;
                v_[i * (N_ + 1) + j] = v_numerator/denominator;
                G_p_[i * (N_ + 1) + j] = G_p_numerator/denominator;
                G_u_[i * (N_ + 1) + j] = G_u_numerator/denominator;
                G_v_[i * (N_ + 1) + j] = G_v_numerator/denominator;
            }
            // A column fits exactly
            else if (column_found != -1) {
                deviceFloat p_numerator = 0.0;
                deviceFloat u_numerator = 0.0;
                deviceFloat v_numerator = 0.0;
                deviceFloat G_p_numerator = 0.0;
                deviceFloat G_u_numerator = 0.0;
                deviceFloat G_v_numerator = 0.0;
                deviceFloat denominator = 0.0;

                for (int n = 0; n <= other.N_; ++n) {
                    const deviceFloat t = barycentric_weights[offset_1D_other + n]/(local_coordinates_in_other.y() - polynomial_nodes[offset_1D_other + n]);
                    p_numerator += t * other.p_[column_found * (other.N_ + 1) + n];
                    u_numerator += t * other.u_[column_found * (other.N_ + 1) + n];
                    v_numerator += t * other.v_[column_found * (other.N_ + 1) + n];
                    G_p_numerator += t * other.G_p_[column_found * (other.N_ + 1) + n];
                    G_u_numerator += t * other.G_u_[column_found * (other.N_ + 1) + n];
                    G_v_numerator += t * other.G_v_[column_found * (other.N_ + 1) + n];
                    denominator += t;
                }
                
                p_[i * (N_ + 1) + j] = p_numerator/denominator;
                u_[i * (N_ + 1) + j] = u_numerator/denominator;
                v_[i * (N_ + 1) + j] = v_numerator/denominator;
                G_p_[i * (N_ + 1) + j] = G_p_numerator/denominator;
                G_u_[i * (N_ + 1) + j] = G_u_numerator/denominator;
                G_v_[i * (N_ + 1) + j] = G_v_numerator/denominator;
            }
            // Complete interpolation
            else {
                p_[i * (N_ + 1) + j] = 0;
                u_[i * (N_ + 1) + j] = 0;
                v_[i * (N_ + 1) + j] = 0;
                G_p_[i * (N_ + 1) + j] = 0;
                G_u_[i * (N_ + 1) + j] = 0;
                G_v_[i * (N_ + 1) + j] = 0;

                deviceFloat denominator_x = 0.0;
                deviceFloat denominator_y = 0.0;

                for (int k = 0; k <= other.N_; ++k) {
                    denominator_x += barycentric_weights[offset_1D_other + k]/(local_coordinates_in_other.x() - polynomial_nodes[offset_1D_other + k]);
                    denominator_y += barycentric_weights[offset_1D_other + k]/(local_coordinates_in_other.y() - polynomial_nodes[offset_1D_other + k]);
                }

                for (int m = 0; m <= other.N_; ++m) {
                    const deviceFloat t_x = barycentric_weights[offset_1D_other + m]/((local_coordinates_in_other.x() - polynomial_nodes[offset_1D_other + m]) * denominator_x);
                    for (int n = 0; n <= other.N_; ++n) {
                        const deviceFloat t_y = barycentric_weights[offset_1D_other + n]/((local_coordinates_in_other.y() - polynomial_nodes[offset_1D_other + n]) * denominator_y);

                        p_[i * (N_ + 1) + j] += other.p_[m * (other.N_ + 1) + n] * t_x * t_y;
                        u_[i * (N_ + 1) + j] += other.u_[m * (other.N_ + 1) + n] * t_x * t_y;
                        v_[i * (N_ + 1) + j] += other.v_[m * (other.N_ + 1) + n] * t_x * t_y;
                        G_p_[i * (N_ + 1) + j] += other.G_p_[m * (other.N_ + 1) + n] * t_x * t_y;
                        G_u_[i * (N_ + 1) + j] += other.G_u_[m * (other.N_ + 1) + n] * t_x * t_y;
                        G_v_[i * (N_ + 1) + j] += other.G_v_[m * (other.N_ + 1) + n] * t_x * t_y;
                    }
                }
            }  
        }
    }
}

__device__
auto SEM::Entities::Element2D_t::interpolate_solution(size_t n_interpolation_points, const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) const -> void {
    for (size_t i = 0; i < n_interpolation_points; ++i) {
        for (size_t j = 0; j < n_interpolation_points; ++j) {
            // x and y
            const Vec2<deviceFloat> coordinates {static_cast<deviceFloat>(i)/static_cast<deviceFloat>(n_interpolation_points - 1) * 2 - 1, static_cast<deviceFloat>(j)/static_cast<deviceFloat>(n_interpolation_points - 1) * 2 - 1};
            const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points);

            x[i * n_interpolation_points + j] = global_coordinates.x();
            y[i * n_interpolation_points + j] = global_coordinates.y();

            // Pressure, u, and v
            p[i * n_interpolation_points + j] = 0.0;
            u[i * n_interpolation_points + j] = 0.0;
            v[i * n_interpolation_points + j] = 0.0;
            for (int m = 0; m <= N_; ++m) {
                for (int n = 0; n <= N_; ++n) {
                    p[i * n_interpolation_points + j] += p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u[i * n_interpolation_points + j] += u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v[i * n_interpolation_points + j] += v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                }
            }
        }
    }
}

__device__
auto SEM::Entities::Element2D_t::interpolate_complete_solution(size_t n_interpolation_points, deviceFloat time, const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* polynomial_nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt, deviceFloat* p_analytical_error, deviceFloat* u_analytical_error, deviceFloat* v_analytical_error) const -> void {
    const int offset_1D = N_ * (N_ + 1) /2;

    for (size_t i = 0; i < n_interpolation_points; ++i) {
        for (size_t j = 0; j < n_interpolation_points; ++j) {
            // x and y
            const Vec2<deviceFloat> coordinates {static_cast<deviceFloat>(i)/static_cast<deviceFloat>(n_interpolation_points - 1) * 2 - 1, static_cast<deviceFloat>(j)/static_cast<deviceFloat>(n_interpolation_points - 1) * 2 - 1};
            const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points);

            x[i * n_interpolation_points + j] = global_coordinates.x();
            y[i * n_interpolation_points + j] = global_coordinates.y();

            // Pressure, u, and v
            p[i * n_interpolation_points + j] = 0.0;
            u[i * n_interpolation_points + j] = 0.0;
            v[i * n_interpolation_points + j] = 0.0;
            dp_dt[i * n_interpolation_points + j] = 0.0;
            du_dt[i * n_interpolation_points + j] = 0.0;
            dv_dt[i * n_interpolation_points + j] = 0.0;
            p_analytical_error[i * n_interpolation_points + j] = 0.0;
            u_analytical_error[i * n_interpolation_points + j] = 0.0;
            v_analytical_error[i * n_interpolation_points + j] = 0.0;
            for (int m = 0; m <= N_; ++m) {
                for (int n = 0; n <= N_; ++n) {
                    p[i * n_interpolation_points + j] += p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u[i * n_interpolation_points + j] += u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v[i * n_interpolation_points + j] += v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    dp_dt[i * n_interpolation_points + j] += G_p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    du_dt[i * n_interpolation_points + j] += G_u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    dv_dt[i * n_interpolation_points + j] += G_v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                
                    const Vec2<deviceFloat> other_coordinates {polynomial_nodes[offset_1D + m], polynomial_nodes[offset_1D + n]};
                    const Vec2<deviceFloat> other_global_coordinates = SEM::quad_map(other_coordinates, points);

                    const std::array<deviceFloat, 3> state = SEM::g(other_global_coordinates, time);
                    p_analytical_error[i * n_interpolation_points + j] += std::abs(state[0] - p_[m * (N_ + 1) + n]) * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u_analytical_error[i * n_interpolation_points + j] += std::abs(state[1] - u_[m * (N_ + 1) + n]) * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v_analytical_error[i * n_interpolation_points + j] += std::abs(state[2] - v_[m * (N_ + 1) + n]) * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
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
              cuda_vector<size_t>(),
              cuda_vector<size_t>(),
              cuda_vector<size_t>()};

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
    scaling_factor_ = {cuda_vector<deviceFloat>(N_ + 1),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>(),
                       cuda_vector<deviceFloat>()};
}

__device__
auto SEM::Entities::Element2D_t::resize_boundary_storage(int N) -> void {
    N_ = N;
    p_extrapolated_[0] = cuda_vector<deviceFloat>(N_ + 1);
    u_extrapolated_[0] = cuda_vector<deviceFloat>(N_ + 1);
    v_extrapolated_[0] = cuda_vector<deviceFloat>(N_ + 1);
    scaling_factor_[0] = cuda_vector<deviceFloat>(N_ + 1);
}

__device__
auto SEM::Entities::Element2D_t::compute_geometry(const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* polynomial_nodes) -> void {
    const size_t offset_1D = N_ * (N_ + 1) /2;

    for (int i = 0; i <= N_; ++i) {
        for (int j = 0; j <= N_; ++j) {
            const Vec2<deviceFloat> coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
            const std::array<Vec2<deviceFloat>, 2> metrics = SEM::quad_metrics(coordinates, points);

            dxi_dx_[i * (N_ + 1) + j]  = metrics[0].x();
            deta_dx_[i * (N_ + 1) + j] = metrics[0].y();
            dxi_dy_[i * (N_ + 1) + j]  = metrics[1].x();
            deta_dy_[i * (N_ + 1) + j] = metrics[1].y();
            jacobian_[i * (N_ + 1) + j] = metrics[0].x() * metrics[1].y() - metrics[0].y() * metrics[1].x();
        }

        const Vec2<deviceFloat> coordinates_bottom {polynomial_nodes[offset_1D + i], -1};
        const Vec2<deviceFloat> coordinates_right  {1, polynomial_nodes[offset_1D + i]};
        const Vec2<deviceFloat> coordinates_top    {polynomial_nodes[offset_1D + i], 1};
        const Vec2<deviceFloat> coordinates_left   {-1, polynomial_nodes[offset_1D + i]};

        const std::array<Vec2<deviceFloat>, 2> metrics_bottom = SEM::quad_metrics(coordinates_bottom, points);
        const std::array<Vec2<deviceFloat>, 2> metrics_right  = SEM::quad_metrics(coordinates_right, points);
        const std::array<Vec2<deviceFloat>, 2> metrics_top    = SEM::quad_metrics(coordinates_top, points);
        const std::array<Vec2<deviceFloat>, 2> metrics_left   = SEM::quad_metrics(coordinates_left, points);

        scaling_factor_[0][i] = std::sqrt(metrics_bottom[0].x() * metrics_bottom[0].x() + metrics_bottom[1].x() * metrics_bottom[1].x());
        scaling_factor_[1][i] = std::sqrt(metrics_right[0].y() * metrics_right[0].y() + metrics_right[1].y() * metrics_right[1].y());
        scaling_factor_[2][i] = std::sqrt(metrics_top[0].x() * metrics_top[0].x() + metrics_top[1].x() * metrics_top[1].x());
        scaling_factor_[3][i] = std::sqrt(metrics_left[0].y() * metrics_left[0].y() + metrics_left[1].y() * metrics_left[1].y());
    }

    delta_xy_min_ = std::min(std::min(
        std::min((points[1] - points[0]).magnitude(), (points[2] - points[3]).magnitude()), 
        std::min((points[1] - points[2]).magnitude(), (points[0] - points[3]).magnitude())), 
        std::min((points[1] - points[3]).magnitude(), (points[2] - points[0]).magnitude()));
    center_ = (points[0] + points[1] + points[2] + points[3])/4;
}

__device__
auto SEM::Entities::Element2D_t::compute_boundary_geometry(const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* polynomial_nodes) -> void {
    const size_t offset_1D = N_ * (N_ + 1) /2;

    for (int i = 0; i <= N_; ++i) {
        const Vec2<deviceFloat> coordinates_bottom {polynomial_nodes[offset_1D + i], -1};

        const std::array<Vec2<deviceFloat>, 2> metrics_bottom = SEM::quad_metrics(coordinates_bottom, points);

        scaling_factor_[0][i] = std::sqrt(metrics_bottom[0].x() * metrics_bottom[0].x() + metrics_bottom[1].x() * metrics_bottom[1].x());
    }
    
    delta_xy_min_ = (points[1] - points[0]).magnitude();
    center_ = (points[0] + points[1])/2;
}

// From cppreference.com
__device__
auto SEM::Entities::Element2D_t::almost_equal(deviceFloat x, deviceFloat y) -> bool {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<deviceFloat>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<deviceFloat>::min();
}

__device__
auto SEM::Entities::Element2D_t::would_p_refine(int max_N) const -> bool {
    return refine_ 
        && (p_sigma_ + u_sigma_ + v_sigma_)/3 >= static_cast<deviceFloat>(1) 
        && N_ + 2 <= max_N;
}

__device__
auto SEM::Entities::Element2D_t::would_h_refine(int max_split_level) const -> bool {
    return refine_ 
        && (p_sigma_ + u_sigma_ + v_sigma_)/3 < static_cast<deviceFloat>(1) 
        && split_level_ < max_split_level;
}

__host__ __device__
auto SEM::Entities::Element2D_t::clear_storage() -> void {
    faces_[0].data_ = nullptr;
    faces_[1].data_ = nullptr;
    faces_[2].data_ = nullptr;
    faces_[3].data_ = nullptr;

    dxi_dx_.data_ = nullptr;
    deta_dx_.data_ = nullptr;
    dxi_dy_.data_ = nullptr;
    deta_dy_.data_ = nullptr;
    jacobian_.data_ = nullptr;
    scaling_factor_[0].data_ = nullptr;
    scaling_factor_[1].data_ = nullptr;
    scaling_factor_[2].data_ = nullptr;
    scaling_factor_[3].data_ = nullptr;

    p_.data_ = nullptr;
    u_.data_ = nullptr;
    v_.data_ = nullptr;

    G_p_.data_ = nullptr;
    G_u_.data_ = nullptr;
    G_v_.data_ = nullptr;

    p_extrapolated_[0].data_ = nullptr;
    p_extrapolated_[1].data_ = nullptr;
    p_extrapolated_[2].data_ = nullptr;
    p_extrapolated_[3].data_ = nullptr;
    u_extrapolated_[0].data_ = nullptr;
    u_extrapolated_[1].data_ = nullptr;
    u_extrapolated_[2].data_ = nullptr;
    u_extrapolated_[3].data_ = nullptr;
    v_extrapolated_[0].data_ = nullptr;
    v_extrapolated_[1].data_ = nullptr;
    v_extrapolated_[2].data_ = nullptr;
    v_extrapolated_[3].data_ = nullptr;

    p_flux_.data_ = nullptr;
    u_flux_.data_ = nullptr;
    v_flux_.data_ = nullptr;
    p_flux_derivative_.data_ = nullptr;
    u_flux_derivative_.data_ = nullptr;
    v_flux_derivative_.data_ = nullptr;

    p_flux_extrapolated_[0].data_ = nullptr;
    p_flux_extrapolated_[1].data_ = nullptr;
    p_flux_extrapolated_[2].data_ = nullptr;
    p_flux_extrapolated_[3].data_ = nullptr;
    u_flux_extrapolated_[0].data_ = nullptr;
    u_flux_extrapolated_[1].data_ = nullptr;
    u_flux_extrapolated_[2].data_ = nullptr;
    u_flux_extrapolated_[3].data_ = nullptr;
    v_flux_extrapolated_[0].data_ = nullptr;
    v_flux_extrapolated_[1].data_ = nullptr;
    v_flux_extrapolated_[2].data_ = nullptr;
    v_flux_extrapolated_[3].data_ = nullptr;
    
    p_intermediate_.data_ = nullptr;
    u_intermediate_.data_ = nullptr;
    v_intermediate_.data_ = nullptr;
    spectrum_.data_ = nullptr;
}

__host__
SEM::Entities::Element2D_t::Datatype::Datatype() {
    constexpr int n = 3;

    constexpr std::array<int, n> lengths {1, 1, 1};
    constexpr std::array<MPI_Aint, n> displacements {offsetof(SEM::Entities::Element2D_t, status_), offsetof(SEM::Entities::Element2D_t, rotation_), offsetof(SEM::Entities::Element2D_t, split_level_)};
    constexpr std::array<MPI_Datatype, n> types {MPI_INT, MPI_INT, MPI_INT}; // Ok I could just send those as packed ints, but who knows if something will have to be added.
    
    MPI_Type_create_struct(n, lengths.data(), displacements.data(), types.data(), &datatype_);
    MPI_Type_commit(&datatype_);
}

__host__
SEM::Entities::Element2D_t::Datatype::~Datatype() {
    MPI_Type_free(&datatype_);
}

__host__
auto SEM::Entities::Element2D_t::Datatype::data() const -> const MPI_Datatype& {
    return datatype_;
}

__host__
auto SEM::Entities::Element2D_t::Datatype::data() -> MPI_Datatype& {
    return datatype_;
}
