#include "solvers/Solver2D_t.h"
#include "helpers/ProgressBar_t.h"
#include "helpers/constants.h"
#include <mpi.h>
#include <limits>
#include <sstream>

using SEM::Host::Entities::Vec2;
using SEM::Host::Entities::Element2D_t;
using SEM::Host::Entities::Face2D_t;

SEM::Host::Solvers::Solver2D_t::Solver2D_t(hostFloat CFL, std::vector<hostFloat> output_times, hostFloat viscosity) :
        CFL_{CFL},
        output_times_{output_times},
        viscosity_{viscosity} {}

auto SEM::Host::Solvers::Solver2D_t::get_delta_t(SEM::Host::Meshes::Mesh2D_t& mesh) const -> hostFloat {   
    hostFloat delta_t_min_local = std::numeric_limits<hostFloat>::infinity();
    for (int i = 0; i < mesh.n_elements_; ++i) {
        delta_t_min_local = std::min(CFL_ * mesh.elements_[i].delta_xy_min_/(mesh.elements_[i].N_ * mesh.elements_[i].N_), delta_t_min_local);
    }

    hostFloat delta_t_min;
    constexpr MPI_Datatype data_type = (sizeof(hostFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE; // CHECK this is a bad way of doing this
    MPI_Allreduce(&delta_t_min_local, &delta_t_min, 1, data_type, MPI_MIN, MPI_COMM_WORLD);
    return delta_t_min;
}

auto SEM::Host::Solvers::Solver2D_t::x_flux(hostFloat p, hostFloat u, hostFloat v) -> std::array<hostFloat, 3> {
    return {SEM::Host::Constants::c * u, p, 0};
}

auto SEM::Host::Solvers::Solver2D_t::y_flux(hostFloat p, hostFloat u, hostFloat v) -> std::array<hostFloat, 3> {
    return {SEM::Host::Constants::c * v, 0, p};
}

void SEM::Host::Solvers::Solver2D_t::matrix_vector_multiply(int N, const std::vector<hostFloat>& matrix, const std::vector<hostFloat>& vector, std::vector<hostFloat>& result) {
    for (int i = 0; i <= N; ++i) {
        result[i] = 0.0;
        for (int j = 0; j <= N; ++j) {
            result[i] +=  matrix[i * (N + 1) + j] * vector[j];
        }
    }
}

auto SEM::Host::Solvers::Solver2D_t::calculate_wave_fluxes(std::vector<Face2D_t>& faces) -> void {
    for (auto& face : faces) {
        // Computing fluxes
        for (int i = 0; i <= face.N_; ++i) {
            const Vec2<hostFloat> u_L {face.u_[0][i], face.v_[0][i]};
            const Vec2<hostFloat> u_R {face.u_[1][i], face.v_[1][i]};

            const hostFloat w_L = face.p_[0][i] + SEM::Host::Constants::c * u_L.dot(face.normal_);
            const hostFloat w_R = face.p_[1][i] - SEM::Host::Constants::c * u_R.dot(face.normal_);

            face.p_flux_[i] = SEM::Host::Constants::c * (w_L - w_R) / 2;
            face.u_flux_[i] = face.normal_.x() * (w_L + w_R) / 2;
            face.v_flux_[i] = face.normal_.y() * (w_L + w_R) / 2;
        }
    }
}

// Algorithm 114
auto SEM::Host::Solvers::compute_dg_wave_derivative(size_t N_elements, Element2D_t* elements, const Face2D_t* faces, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) -> void {
    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        Element2D_t& element = elements[element_index];

        // Horizontal direction
        for (int j = 0; j <= element.N_; ++j) {
            for (int i = 0; i <= element.N_; ++i) {
                const std::array<hostFloat, 3> flux_x = SEM::Host::Solvers::Solver2D_t::x_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);
                const std::array<hostFloat, 3> flux_y = SEM::Host::Solvers::Solver2D_t::y_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);
            
                element.p_flux_[i] = element.deta_dy_[i * (element.N_ + 1) + j] * flux_x[0] - element.deta_dx_[i * (element.N_ + 1) + j] * flux_y[0];
                element.u_flux_[i] = element.deta_dy_[i * (element.N_ + 1) + j] * flux_x[1] - element.deta_dx_[i * (element.N_ + 1) + j] * flux_y[1];
                element.v_flux_[i] = element.deta_dy_[i * (element.N_ + 1) + j] * flux_x[2] - element.deta_dx_[i * (element.N_ + 1) + j] * flux_y[2];
            }

            SEM::Host::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat[element.N_], element.p_flux_, element.p_flux_derivative_);
            SEM::Host::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat[element.N_], element.u_flux_, element.u_flux_derivative_);
            SEM::Host::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat[element.N_], element.v_flux_, element.v_flux_derivative_);

            // For the boundaries, the numbering increases from the first node to the second. 
            // Inside the element, the ksi and eta coordinates increase from left to right, bottom to top.
            // This means that there is an inconsistency on the top and left edges, and the numbering has to be reversed.
            // This way, the projection from the element edge to the face(s) can always be done in the same way.
            // The same process has to be done when interpolating to the boundaries.
            for (int i = 0; i <= element.N_; ++i) {
                element.p_flux_derivative_[i] += (element.p_flux_extrapolated_[1][j] * lagrange_interpolant_right[element.N_][i] + element.p_flux_extrapolated_[3][element.N_ - j] * lagrange_interpolant_left[element.N_][i]) / weights[element.N_][i];
                element.u_flux_derivative_[i] += (element.u_flux_extrapolated_[1][j] * lagrange_interpolant_right[element.N_][i] + element.u_flux_extrapolated_[3][element.N_ - j] * lagrange_interpolant_left[element.N_][i]) / weights[element.N_][i];
                element.v_flux_derivative_[i] += (element.v_flux_extrapolated_[1][j] * lagrange_interpolant_right[element.N_][i] + element.v_flux_extrapolated_[3][element.N_ - j] * lagrange_interpolant_left[element.N_][i]) / weights[element.N_][i];
            }

            for (int i = 0; i <= element.N_; ++i) {
                element.G_p_[i * (element.N_ + 1) + j] = -element.p_flux_derivative_[i];
                element.G_u_[i * (element.N_ + 1) + j] = -element.u_flux_derivative_[i];
                element.G_v_[i * (element.N_ + 1) + j] = -element.v_flux_derivative_[i];
            }
        }

        // Vertical direction
        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const std::array<hostFloat, 3> flux_x = SEM::Host::Solvers::Solver2D_t::x_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);
                const std::array<hostFloat, 3> flux_y = SEM::Host::Solvers::Solver2D_t::y_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);

                element.p_flux_[j] = -element.dxi_dy_[i * (element.N_ + 1) + j] * flux_x[0] + element.dxi_dx_[i * (element.N_ + 1) + j] * flux_y[0];
                element.u_flux_[j] = -element.dxi_dy_[i * (element.N_ + 1) + j] * flux_x[1] + element.dxi_dx_[i * (element.N_ + 1) + j] * flux_y[1];
                element.v_flux_[j] = -element.dxi_dy_[i * (element.N_ + 1) + j] * flux_x[2] + element.dxi_dx_[i * (element.N_ + 1) + j] * flux_y[2];
            }

            SEM::Host::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat[element.N_], element.p_flux_, element.p_flux_derivative_);
            SEM::Host::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat[element.N_], element.u_flux_, element.u_flux_derivative_);
            SEM::Host::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat[element.N_], element.v_flux_, element.v_flux_derivative_);

            // For the boundaries, the numbering increases from the first node to the second. 
            // Inside the element, the ksi and eta coordinates increase from left to right, bottom to top.
            // This means that there is an inconsistency on the top and left edges, and the numbering has to be reversed.
            // This way, the projection from the element edge to the face(s) can always be done in the same way.
            // The same process has to be done when interpolating to the boundaries.
            for (int j = 0; j <= element.N_; ++j) {
                element.p_flux_derivative_[j] += (element.p_flux_extrapolated_[2][element.N_ - i] * lagrange_interpolant_right[element.N_][j] + element.p_flux_extrapolated_[0][i] * lagrange_interpolant_left[element.N_][j]) / weights[element.N_][j];
                element.u_flux_derivative_[j] += (element.u_flux_extrapolated_[2][element.N_ - i] * lagrange_interpolant_right[element.N_][j] + element.u_flux_extrapolated_[0][i] * lagrange_interpolant_left[element.N_][j]) / weights[element.N_][j];
                element.v_flux_derivative_[j] += (element.v_flux_extrapolated_[2][element.N_ - i] * lagrange_interpolant_right[element.N_][j] + element.v_flux_extrapolated_[0][i] * lagrange_interpolant_left[element.N_][j]) / weights[element.N_][j];
            }

            for (int j = 0; j <= element.N_; ++j) {
                element.G_p_[i * (element.N_ + 1) + j] = (element.G_p_[i * (element.N_ + 1) + j] - element.p_flux_derivative_[j]) / element.jacobian_[i * (element.N_ + 1) + j];
                element.G_u_[i * (element.N_ + 1) + j] = (element.G_u_[i * (element.N_ + 1) + j] - element.u_flux_derivative_[j]) / element.jacobian_[i * (element.N_ + 1) + j];
                element.G_v_[i * (element.N_ + 1) + j] = (element.G_v_[i * (element.N_ + 1) + j] - element.v_flux_derivative_[j]) / element.jacobian_[i * (element.N_ + 1) + j];
            }
        }
    }
}

auto SEM::Host::Solvers::rk3_first_step(size_t N_elements, SEM::Host::Entities::Element2D_t* elements, hostFloat delta_t, hostFloat g) -> void {
    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        Element2D_t& element = elements[element_index];

        for (int i = 0; i <= element.N_; ++i){
            for (int j = 0; j <= element.N_; ++j){
                element.p_intermediate_[i * (element.N_ + 1) + j] = element.G_p_[i * (element.N_ + 1) + j];
                element.u_intermediate_[i * (element.N_ + 1) + j] = element.G_u_[i * (element.N_ + 1) + j];
                element.v_intermediate_[i * (element.N_ + 1) + j] = element.G_v_[i * (element.N_ + 1) + j];

                element.p_[i * (element.N_ + 1) + j] += g * delta_t * element.p_intermediate_[i * (element.N_ + 1) + j];
                element.u_[i * (element.N_ + 1) + j] += g * delta_t * element.u_intermediate_[i * (element.N_ + 1) + j];
                element.v_[i * (element.N_ + 1) + j] += g * delta_t * element.v_intermediate_[i * (element.N_ + 1) + j];
            }
        }
    }
}

auto SEM::Host::Solvers::rk3_step(size_t N_elements, SEM::Host::Entities::Element2D_t* elements, hostFloat delta_t, hostFloat a, hostFloat g) -> void {
    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        Element2D_t& element = elements[element_index];

        for (int i = 0; i <= element.N_; ++i){
            for (int j = 0; j <= element.N_; ++j){
                element.p_intermediate_[i * (element.N_ + 1) + j] = a * element.p_intermediate_[i * (element.N_ + 1) + j] + element.G_p_[i * (element.N_ + 1) + j];
                element.u_intermediate_[i * (element.N_ + 1) + j] = a * element.u_intermediate_[i * (element.N_ + 1) + j] + element.G_u_[i * (element.N_ + 1) + j];
                element.v_intermediate_[i * (element.N_ + 1) + j] = a * element.v_intermediate_[i * (element.N_ + 1) + j] + element.G_v_[i * (element.N_ + 1) + j];

                element.p_[i * (element.N_ + 1) + j] += g * delta_t * element.p_intermediate_[i * (element.N_ + 1) + j];
                element.u_[i * (element.N_ + 1) + j] += g * delta_t * element.u_intermediate_[i * (element.N_ + 1) + j];
                element.v_[i * (element.N_ + 1) + j] += g * delta_t * element.v_intermediate_[i * (element.N_ + 1) + j];
            }
        }
    }
}
