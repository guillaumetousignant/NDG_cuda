#include "entities/Element2D_t.h"
#include "functions/quad_map.h"
#include "functions/inverse_quad_map.h"
#include "functions/quad_metrics.h"
#include "functions/analytical_solution.h"
#include <cmath>
#include <limits>
#include <cstddef>

using namespace SEM::Host::Hilbert;

SEM::Host::Entities::Element2D_t::Element2D_t(int N, int split_level, Hilbert::Status status, int rotation, const std::array<std::vector<size_t>, 4>& faces, std::array<size_t, 4> nodes) : 
        N_{N},
        faces_{faces},
        nodes_{nodes},
        status_{status},
        delta_xy_min_{0.0},
        center_{0.0, 0.0},
        dxi_dx_((N_ + 1) * (N_ + 1)),
        deta_dx_((N_ + 1) * (N_ + 1)),
        dxi_dy_((N_ + 1) * (N_ + 1)),
        deta_dy_((N_ + 1) * (N_ + 1)),
        jacobian_((N_ + 1) * (N_ + 1)),
        scaling_factor_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        rotation_{rotation},
        p_((N_ + 1) * (N_ + 1)),
        u_((N_ + 1) * (N_ + 1)),
        v_((N_ + 1) * (N_ + 1)),
        G_p_((N_ + 1) * (N_ + 1)),
        G_u_((N_ + 1) * (N_ + 1)),
        G_v_((N_ + 1) * (N_ + 1)),
        p_extrapolated_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        u_extrapolated_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        v_extrapolated_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        p_flux_(N_ + 1),
        u_flux_(N_ + 1),
        v_flux_(N_ + 1),
        p_flux_derivative_(N_ + 1),
        u_flux_derivative_(N_ + 1),
        v_flux_derivative_(N_ + 1),
        p_flux_extrapolated_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        u_flux_extrapolated_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        v_flux_extrapolated_{std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1), std::vector<hostFloat>(N_ + 1)},
        p_intermediate_((N_ + 1) * (N_ + 1)),
        u_intermediate_((N_ + 1) * (N_ + 1)),
        v_intermediate_((N_ + 1) * (N_ + 1)),
        spectrum_(N_ + 1),
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

SEM::Host::Entities::Element2D_t::Element2D_t() :
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
auto SEM::Host::Entities::Element2D_t::interpolate_to_boundaries(const std::vector<hostFloat>& lagrange_interpolant_minus, const std::vector<hostFloat>& lagrange_interpolant_plus) -> void {
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
            p_extrapolated_[0][i] += lagrange_interpolant_minus[j] * p_[i * (N_ + 1) + j];
            p_extrapolated_[2][N_ - i] += lagrange_interpolant_plus[j] * p_[i * (N_ + 1) + j];
            p_extrapolated_[1][i] += lagrange_interpolant_plus[j] * p_[j * (N_ + 1) + i];
            p_extrapolated_[3][N_ - i] += lagrange_interpolant_minus[j] * p_[j * (N_ + 1) + i];

            u_extrapolated_[0][i] += lagrange_interpolant_minus[j] * u_[i * (N_ + 1) + j];
            u_extrapolated_[2][N_ - i] += lagrange_interpolant_plus[j] * u_[i * (N_ + 1) + j];
            u_extrapolated_[1][i] += lagrange_interpolant_plus[j] * u_[j * (N_ + 1) + i];
            u_extrapolated_[3][N_ - i] += lagrange_interpolant_minus[j] * u_[j * (N_ + 1) + i];

            v_extrapolated_[0][i] += lagrange_interpolant_minus[j] * v_[i * (N_ + 1) + j];
            v_extrapolated_[2][N_ - i] += lagrange_interpolant_plus[j] * v_[i * (N_ + 1) + j];
            v_extrapolated_[1][i] += lagrange_interpolant_plus[j] * v_[j * (N_ + 1) + i];
            v_extrapolated_[3][N_ - i] += lagrange_interpolant_minus[j] * v_[j * (N_ + 1) + i];
        }
    }
}

// Algorithm 61
auto SEM::Host::Entities::Element2D_t::interpolate_q_to_boundaries(const std::vector<hostFloat>& lagrange_interpolant_minus, const std::vector<hostFloat>& lagrange_interpolant_plus) -> void {
    printf("Warning, SEM::Host::Entities::Element2D_t::interpolate_q_to_boundaries is not implemented.\n");
}

auto SEM::Host::Entities::Element2D_t::exponential_decay(int n_points_least_squares) -> std::array<hostFloat, 2> {
    hostFloat x_avg = 0.0;
    hostFloat y_avg = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        x_avg += N_ + i + 1 - n_points_least_squares;
        y_avg += std::log(spectrum_[i]);
    }

    x_avg /= n_points_least_squares;
    y_avg /= n_points_least_squares;

    hostFloat numerator = 0.0;
    hostFloat denominator = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        const int p = N_ + i + 1 - n_points_least_squares;
        numerator += (p - x_avg) * (std::log(spectrum_[i]) - y_avg);
        denominator += (p - x_avg) * (p - x_avg);
    }

    const hostFloat sigma = numerator/denominator;
    const hostFloat C = std::exp(y_avg - sigma * x_avg);
    return {C, std::abs(sigma)};
}

auto SEM::Host::Entities::Element2D_t::interpolate_from(const std::array<Vec2<hostFloat>, 4>& points, const std::array<Vec2<hostFloat>, 4>& points_other, const SEM::Host::Entities::Element2D_t& other, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    for (int i = 0; i <= N_; ++i) {
        for (int j = 0; j <= N_; ++j) {
            // x and y
            const Vec2<hostFloat> local_coordinates {polynomial_nodes[N_][i], polynomial_nodes[N_][j]};
            const Vec2<hostFloat> global_coordinates = SEM::Host::quad_map(local_coordinates, points);
            const Vec2<hostFloat> local_coordinates_in_other = SEM::Host::inverse_quad_map(global_coordinates, points_other);

            int row_found = -1;
            int column_found = -1;
            for (int m = 0; m <= other.N_; ++m) {
                if (SEM::Host::Entities::Element2D_t::almost_equal(local_coordinates_in_other.x(), polynomial_nodes[other.N_][m])) {
                    column_found = m;
                }
                if (SEM::Host::Entities::Element2D_t::almost_equal(local_coordinates_in_other.y(), polynomial_nodes[other.N_][m])) {
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
                hostFloat p_numerator = 0.0;
                hostFloat u_numerator = 0.0;
                hostFloat v_numerator = 0.0;
                hostFloat G_p_numerator = 0.0;
                hostFloat G_u_numerator = 0.0;
                hostFloat G_v_numerator = 0.0;
                hostFloat denominator = 0.0;

                for (int m = 0; m <= other.N_; ++m) {
                    const hostFloat t = barycentric_weights[other.N_][m]/(local_coordinates_in_other.x() - polynomial_nodes[other.N_][m]);
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
                hostFloat p_numerator = 0.0;
                hostFloat u_numerator = 0.0;
                hostFloat v_numerator = 0.0;
                hostFloat G_p_numerator = 0.0;
                hostFloat G_u_numerator = 0.0;
                hostFloat G_v_numerator = 0.0;
                hostFloat denominator = 0.0;

                for (int n = 0; n <= other.N_; ++n) {
                    const hostFloat t = barycentric_weights[other.N_][n]/(local_coordinates_in_other.y() - polynomial_nodes[other.N_][n]);
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

                hostFloat denominator_x = 0.0;
                hostFloat denominator_y = 0.0;

                for (int k = 0; k <= other.N_; ++k) {
                    denominator_x += barycentric_weights[other.N_][k]/(local_coordinates_in_other.x() - polynomial_nodes[other.N_][k]);
                    denominator_y += barycentric_weights[other.N_][k]/(local_coordinates_in_other.y() - polynomial_nodes[other.N_][k]);
                }

                for (int m = 0; m <= other.N_; ++m) {
                    const hostFloat t_x = barycentric_weights[other.N_][m]/((local_coordinates_in_other.x() - polynomial_nodes[other.N_][m]) * denominator_x);
                    for (int n = 0; n <= other.N_; ++n) {
                        const hostFloat t_y = barycentric_weights[other.N_][n]/((local_coordinates_in_other.y() - polynomial_nodes[other.N_][n]) * denominator_y);

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

auto SEM::Host::Entities::Element2D_t::interpolate_from(const SEM::Host::Entities::Element2D_t& other, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    for (int i = 0; i <= N_; ++i) {
        for (int j = 0; j <= N_; ++j) {
            // x and y
            const Vec2<hostFloat> local_coordinates_in_other = {polynomial_nodes[N_][i], polynomial_nodes[N_][j]};

            int row_found = -1;
            int column_found = -1;
            for (int m = 0; m <= other.N_; ++m) {
                if (SEM::Host::Entities::Element2D_t::almost_equal(local_coordinates_in_other.x(), polynomial_nodes[other.N_][m])) {
                    column_found = m;
                }
                if (SEM::Host::Entities::Element2D_t::almost_equal(local_coordinates_in_other.y(), polynomial_nodes[other.N_][m])) {
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
                hostFloat p_numerator = 0.0;
                hostFloat u_numerator = 0.0;
                hostFloat v_numerator = 0.0;
                hostFloat G_p_numerator = 0.0;
                hostFloat G_u_numerator = 0.0;
                hostFloat G_v_numerator = 0.0;
                hostFloat denominator = 0.0;

                for (int m = 0; m <= other.N_; ++m) {
                    const hostFloat t = barycentric_weights[other.N_][m]/(local_coordinates_in_other.x() - polynomial_nodes[other.N_][m]);
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
                hostFloat p_numerator = 0.0;
                hostFloat u_numerator = 0.0;
                hostFloat v_numerator = 0.0;
                hostFloat G_p_numerator = 0.0;
                hostFloat G_u_numerator = 0.0;
                hostFloat G_v_numerator = 0.0;
                hostFloat denominator = 0.0;

                for (int n = 0; n <= other.N_; ++n) {
                    const hostFloat t = barycentric_weights[other.N_][n]/(local_coordinates_in_other.y() - polynomial_nodes[other.N_][n]);
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

                hostFloat denominator_x = 0.0;
                hostFloat denominator_y = 0.0;

                for (int k = 0; k <= other.N_; ++k) {
                    denominator_x += barycentric_weights[other.N_][k]/(local_coordinates_in_other.x() - polynomial_nodes[other.N_][k]);
                    denominator_y += barycentric_weights[other.N_][k]/(local_coordinates_in_other.y() - polynomial_nodes[other.N_][k]);
                }

                for (int m = 0; m <= other.N_; ++m) {
                    const hostFloat t_x = barycentric_weights[other.N_][m]/((local_coordinates_in_other.x() - polynomial_nodes[other.N_][m]) * denominator_x);
                    for (int n = 0; n <= other.N_; ++n) {
                        const hostFloat t_y = barycentric_weights[other.N_][n]/((local_coordinates_in_other.y() - polynomial_nodes[other.N_][n]) * denominator_y);

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

auto SEM::Host::Entities::Element2D_t::interpolate_solution(size_t n_interpolation_points, size_t output_offset, const std::array<Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& interpolation_matrices, std::vector<hostFloat>& x, std::vector<hostFloat>& y, std::vector<hostFloat>& p, std::vector<hostFloat>& u, std::vector<hostFloat>& v) const -> void {
    for (size_t i = 0; i < n_interpolation_points; ++i) {
        for (size_t j = 0; j < n_interpolation_points; ++j) {
            // x and y
            const Vec2<hostFloat> coordinates {static_cast<hostFloat>(i)/static_cast<hostFloat>(n_interpolation_points - 1) * 2 - 1, static_cast<hostFloat>(j)/static_cast<hostFloat>(n_interpolation_points - 1) * 2 - 1};
            const Vec2<hostFloat> global_coordinates = SEM::Host::quad_map(coordinates, points);

            x[output_offset + i * n_interpolation_points + j] = global_coordinates.x();
            y[output_offset + i * n_interpolation_points + j] = global_coordinates.y();

            // Pressure, u, and v
            p[output_offset + i * n_interpolation_points + j] = 0.0;
            u[output_offset + i * n_interpolation_points + j] = 0.0;
            v[output_offset + i * n_interpolation_points + j] = 0.0;
            for (int m = 0; m <= N_; ++m) {
                for (int n = 0; n <= N_; ++n) {
                    p[output_offset + i * n_interpolation_points + j] += p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u[output_offset + i * n_interpolation_points + j] += u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v[output_offset + i * n_interpolation_points + j] += v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                }
            }
        }
    }
}

auto SEM::Host::Entities::Element2D_t::interpolate_complete_solution(size_t n_interpolation_points, hostFloat time, size_t output_offset, const std::array<Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& polynomial_nodes, const std::vector<hostFloat>& interpolation_matrices, std::vector<hostFloat>& x, std::vector<hostFloat>& y, std::vector<hostFloat>& p, std::vector<hostFloat>& u, std::vector<hostFloat>& v, std::vector<hostFloat>& dp_dt, std::vector<hostFloat>& du_dt, std::vector<hostFloat>& dv_dt, std::vector<hostFloat>& p_analytical_error, std::vector<hostFloat>& u_analytical_error, std::vector<hostFloat>& v_analytical_error) const -> void {
    for (size_t i = 0; i < n_interpolation_points; ++i) {
        for (size_t j = 0; j < n_interpolation_points; ++j) {
            // x and y
            const Vec2<hostFloat> coordinates {static_cast<hostFloat>(i)/static_cast<hostFloat>(n_interpolation_points - 1) * 2 - 1, static_cast<hostFloat>(j)/static_cast<hostFloat>(n_interpolation_points - 1) * 2 - 1};
            const Vec2<hostFloat> global_coordinates = SEM::Host::quad_map(coordinates, points);

            x[output_offset + i * n_interpolation_points + j] = global_coordinates.x();
            y[output_offset + i * n_interpolation_points + j] = global_coordinates.y();

            // Pressure, u, and v
            p[output_offset + i * n_interpolation_points + j] = 0.0;
            u[output_offset + i * n_interpolation_points + j] = 0.0;
            v[output_offset + i * n_interpolation_points + j] = 0.0;
            dp_dt[output_offset + i * n_interpolation_points + j] = 0.0;
            du_dt[output_offset + i * n_interpolation_points + j] = 0.0;
            dv_dt[output_offset + i * n_interpolation_points + j] = 0.0;
            p_analytical_error[output_offset + i * n_interpolation_points + j] = 0.0;
            u_analytical_error[output_offset + i * n_interpolation_points + j] = 0.0;
            v_analytical_error[output_offset + i * n_interpolation_points + j] = 0.0;
            for (int m = 0; m <= N_; ++m) {
                for (int n = 0; n <= N_; ++n) {
                    p[output_offset + i * n_interpolation_points + j] += p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u[output_offset + i * n_interpolation_points + j] += u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v[output_offset + i * n_interpolation_points + j] += v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    dp_dt[output_offset + i * n_interpolation_points + j] += G_p_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    du_dt[output_offset + i * n_interpolation_points + j] += G_u_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    dv_dt[output_offset + i * n_interpolation_points + j] += G_v_[m * (N_ + 1) + n] * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                
                    const Vec2<hostFloat> other_coordinates {polynomial_nodes[m], polynomial_nodes[n]};
                    const Vec2<hostFloat> other_global_coordinates = SEM::Host::quad_map(other_coordinates, points);

                    const std::array<hostFloat, 3> state = SEM::Host::g(other_global_coordinates, time);
                    p_analytical_error[output_offset + i * n_interpolation_points + j] += std::abs(state[0] - p_[m * (N_ + 1) + n]) * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    u_analytical_error[output_offset + i * n_interpolation_points + j] += std::abs(state[1] - u_[m * (N_ + 1) + n]) * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                    v_analytical_error[output_offset + i * n_interpolation_points + j] += std::abs(state[2] - v_[m * (N_ + 1) + n]) * interpolation_matrices[i * (N_ + 1) + m] * interpolation_matrices[j * (N_ + 1) + n];
                }
            }
        }
    }
}

auto SEM::Host::Entities::Element2D_t::allocate_storage() -> void {
    faces_ = {std::vector<size_t>(1),
              std::vector<size_t>(1),
              std::vector<size_t>(1),
              std::vector<size_t>(1)};

    dxi_dx_  = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    deta_dx_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    dxi_dy_  = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    deta_dy_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    jacobian_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    scaling_factor_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1)};

    p_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    u_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    v_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));

    G_p_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    G_u_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    G_v_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));

    p_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1)};
    u_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1)};
    v_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(N_ + 1)};

    p_flux_ = std::vector<hostFloat>(N_ + 1);
    u_flux_ = std::vector<hostFloat>(N_ + 1);
    v_flux_ = std::vector<hostFloat>(N_ + 1);
    p_flux_derivative_ = std::vector<hostFloat>(N_ + 1);
    u_flux_derivative_ = std::vector<hostFloat>(N_ + 1);
    v_flux_derivative_ = std::vector<hostFloat>(N_ + 1);

    p_flux_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1)};
    u_flux_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1)};
    v_flux_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1),
                            std::vector<hostFloat>(N_ + 1)};
    
    p_intermediate_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    u_intermediate_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    v_intermediate_ = std::vector<hostFloat>((N_ + 1) * (N_ + 1));
    spectrum_ = std::vector<hostFloat>(N_ + 1);
}

auto SEM::Host::Entities::Element2D_t::allocate_boundary_storage() -> void {
    faces_ = {std::vector<size_t>(1),
              std::vector<size_t>(),
              std::vector<size_t>(),
              std::vector<size_t>()};

    p_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>()};
    u_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>()};
    v_extrapolated_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>()};
    scaling_factor_ = {std::vector<hostFloat>(N_ + 1),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>(),
                       std::vector<hostFloat>()};
}

auto SEM::Host::Entities::Element2D_t::resize_boundary_storage(int N) -> void {
    N_ = N;
    p_extrapolated_[0] = std::vector<hostFloat>(N_ + 1);
    u_extrapolated_[0] = std::vector<hostFloat>(N_ + 1);
    v_extrapolated_[0] = std::vector<hostFloat>(N_ + 1);
    scaling_factor_[0] = std::vector<hostFloat>(N_ + 1);
}

auto SEM::Host::Entities::Element2D_t::compute_geometry(const std::array<Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& polynomial_nodes) -> void {
    for (int i = 0; i <= N_; ++i) {
        for (int j = 0; j <= N_; ++j) {
            const Vec2<hostFloat> coordinates {polynomial_nodes[i], polynomial_nodes[j]};
            const std::array<Vec2<hostFloat>, 2> metrics = SEM::Host::quad_metrics(coordinates, points);

            dxi_dx_[i * (N_ + 1) + j]  = metrics[0].x();
            deta_dx_[i * (N_ + 1) + j] = metrics[0].y();
            dxi_dy_[i * (N_ + 1) + j]  = metrics[1].x();
            deta_dy_[i * (N_ + 1) + j] = metrics[1].y();
            jacobian_[i * (N_ + 1) + j] = metrics[0].x() * metrics[1].y() - metrics[0].y() * metrics[1].x();
        }

        const Vec2<hostFloat> coordinates_bottom {polynomial_nodes[i], -1};
        const Vec2<hostFloat> coordinates_right  {1, polynomial_nodes[i]};
        const Vec2<hostFloat> coordinates_top    {polynomial_nodes[i], 1};
        const Vec2<hostFloat> coordinates_left   {-1, polynomial_nodes[i]};

        const std::array<Vec2<hostFloat>, 2> metrics_bottom = SEM::Host::quad_metrics(coordinates_bottom, points);
        const std::array<Vec2<hostFloat>, 2> metrics_right  = SEM::Host::quad_metrics(coordinates_right, points);
        const std::array<Vec2<hostFloat>, 2> metrics_top    = SEM::Host::quad_metrics(coordinates_top, points);
        const std::array<Vec2<hostFloat>, 2> metrics_left   = SEM::Host::quad_metrics(coordinates_left, points);

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

auto SEM::Host::Entities::Element2D_t::compute_boundary_geometry(const std::array<Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& polynomial_nodes) -> void {
    for (int i = 0; i <= N_; ++i) {
        const Vec2<hostFloat> coordinates_bottom {polynomial_nodes[i], -1};

        const std::array<Vec2<hostFloat>, 2> metrics_bottom = SEM::Host::quad_metrics(coordinates_bottom, points);

        scaling_factor_[0][i] = std::sqrt(metrics_bottom[0].x() * metrics_bottom[0].x() + metrics_bottom[1].x() * metrics_bottom[1].x());
    }
    
    delta_xy_min_ = (points[1] - points[0]).magnitude();
    center_ = (points[0] + points[1])/2;
}

// From cppreference.com
auto SEM::Host::Entities::Element2D_t::almost_equal(hostFloat x, hostFloat y) -> bool {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<hostFloat>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<hostFloat>::min();
}

auto SEM::Host::Entities::Element2D_t::would_p_refine(int max_N) const -> bool {
    return refine_ 
        && (p_sigma_ + u_sigma_ + v_sigma_)/3 >= static_cast<hostFloat>(1) 
        && N_ + 2 <= max_N;
}

auto SEM::Host::Entities::Element2D_t::would_h_refine(int max_split_level) const -> bool {
    return refine_ 
        && (p_sigma_ + u_sigma_ + v_sigma_)/3 < static_cast<hostFloat>(1) 
        && split_level_ < max_split_level;
}

SEM::Host::Entities::Element2D_t::Datatype::Datatype() {
    constexpr int n = 3;

    constexpr std::array<int, n> lengths {1, 1, 1};
    constexpr std::array<MPI_Aint, n> displacements {offsetof(SEM::Host::Entities::Element2D_t, status_), offsetof(SEM::Host::Entities::Element2D_t, rotation_), offsetof(SEM::Host::Entities::Element2D_t, split_level_)};
    const std::array<MPI_Datatype, n> types {MPI_INT, MPI_INT, MPI_INT}; // Ok I could just send those as packed ints, but who knows if something will have to be added.
    
    MPI_Type_create_struct(n, lengths.data(), displacements.data(), types.data(), &datatype_);
    MPI_Type_commit(&datatype_);
}

SEM::Host::Entities::Element2D_t::Datatype::~Datatype() {
    MPI_Type_free(&datatype_);
}

auto SEM::Host::Entities::Element2D_t::Datatype::data() const -> const MPI_Datatype& {
    return datatype_;
}

auto SEM::Host::Entities::Element2D_t::Datatype::data() -> MPI_Datatype& {
    return datatype_;
}
