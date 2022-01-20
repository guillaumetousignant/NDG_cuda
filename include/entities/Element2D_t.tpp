#include "helpers/constants.h"

template<typename Polynomial>
auto SEM::Host::Entities::Element2D_t::estimate_error(hostFloat tolerance_min, hostFloat tolerance_max, const std::vector<hostFloat>& polynomial_nodes, const std::vector<hostFloat>& weights) -> void {
    const int n_points_least_squares = std::min(N_ + 1, SEM::Host::Constants::n_points_least_squares_max); // Number of points to use for thew least squares reduction, but don't go above N.

    refine_ = false;
    coarsen_ = true;

    // Pressure
    for (int node_index = 0; node_index < n_points_least_squares; ++node_index) {
        spectrum_[node_index] = 0.0;
        const hostFloat p = N_ + node_index + 1 - n_points_least_squares;

        // x direction
        for (int i = 0; i <= p; ++i) {
            hostFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const hostFloat L_N_x = Polynomial::polynomial(i, polynomial_nodes[k]);

                for (int l = 0; l <= N_; ++l) {
                    const hostFloat L_N_y = Polynomial::polynomial(p, polynomial_nodes[l]);

                    local_spectrum += (2 * i + 1) * (2 * p + 1) * static_cast<hostFloat>(0.25) *
                                      p_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[k] * weights[l];
                }

            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }

        // y direction
        for (int j = 0; j < p; ++j) { // No need to include the last point here
            hostFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const hostFloat L_N_x = Polynomial::polynomial(p, polynomial_nodes[k]);

                for (int l = 0; l <= N_; ++l) {
                    const hostFloat L_N_y = Polynomial::polynomial(j, polynomial_nodes[l]);

                    local_spectrum += (2 * j + 1) * (2 * (double)p + 1.0) * static_cast<hostFloat>(0.25) *
                                      p_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[k] * weights[l];
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
        const hostFloat p = N_ + node_index + 1 - n_points_least_squares;

        // x direction
        for (int i = 0; i <= p; ++i) {
            hostFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const hostFloat L_N_x = Polynomial::polynomial(i, polynomial_nodes[k]);

                for (int l = 0; l <= N_; ++l) {
                    const hostFloat L_N_y = Polynomial::polynomial(p, polynomial_nodes[l]);

                    local_spectrum += (2 * i + 1) * (2 * p + 1) * static_cast<hostFloat>(0.25) *
                                      u_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[k] * weights[l];
                }

            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }

        // y direction
        for (int j = 0; j < p; ++j) { // No need to include the last point here
            hostFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const hostFloat L_N_x = Polynomial::polynomial(p, polynomial_nodes[k]);

                for (int l = 0; l <= N_; ++l) {
                    const hostFloat L_N_y = Polynomial::polynomial(j, polynomial_nodes[l]);

                    local_spectrum += (2 * j + 1) * (2 * (double)p + 1.0) * static_cast<hostFloat>(0.25) *
                                      u_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[k] * weights[l];
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
        const hostFloat p = N_ + node_index + 1 - n_points_least_squares;

        // x direction
        for (int i = 0; i <= p; ++i) {
            hostFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const hostFloat L_N_x = Polynomial::polynomial(i, polynomial_nodes[k]);

                for (int l = 0; l <= N_; ++l) {
                    const hostFloat L_N_y = Polynomial::polynomial(p, polynomial_nodes[l]);

                    local_spectrum += (2 * i + 1) * (2 * p + 1) * static_cast<hostFloat>(0.25) *
                                      v_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[k] * weights[l];
                }

            }
            spectrum_[node_index] += std::abs(local_spectrum);
        }

        // y direction
        for (int j = 0; j < p; ++j) { // No need to include the last point here
            hostFloat local_spectrum = 0.0;

            for (int k = 0; k <= N_; ++k) {
                const hostFloat L_N_x = Polynomial::polynomial(p, polynomial_nodes[k]);

                for (int l = 0; l <= N_; ++l) {
                    const hostFloat L_N_y = Polynomial::polynomial(j, polynomial_nodes[l]);

                    local_spectrum += (2 * j + 1) * (2 * (double)p + 1.0) * static_cast<hostFloat>(0.25) *
                                      v_[k * (N_ + 1) + l] *
                                      L_N_x * L_N_y * 
                                      weights[k] * weights[l];
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