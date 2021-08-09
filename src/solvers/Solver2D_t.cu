#include "solvers/Solver2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "helpers/ProgressBar_t.h"
#include "helpers/constants.h"
#include <mpi.h>
#include <limits>
#include <sstream>

using SEM::Entities::Vec2;
using SEM::Entities::Element2D_t;
using SEM::Entities::Face2D_t;

SEM::Solvers::Solver2D_t::Solver2D_t(deviceFloat CFL, std::vector<deviceFloat> output_times, deviceFloat viscosity) :
        CFL_{CFL},
        output_times_{output_times},
        viscosity_{viscosity} {}

template auto SEM::Solvers::Solver2D_t::solve(const SEM::Entities::NDG_t<SEM::Polynomials::ChebyshevPolynomial_t> &NDG, SEM::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) const -> void; // Get with the times c++, it's crazy I have to do this
template auto SEM::Solvers::Solver2D_t::solve(const SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> &NDG, SEM::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) const -> void;

template<typename Polynomial>
auto SEM::Solvers::Solver2D_t::solve(const SEM::Entities::NDG_t<Polynomial> &NDG, SEM::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) const -> void {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    deviceFloat time = 0.0;
    const deviceFloat t_end = output_times_.back();
    SEM::Helpers::ProgressBar_t bar;
    size_t timestep = 0;
    constexpr std::array<deviceFloat, 3> am {0, -5.0/9.0, -153.0/128.0};
    constexpr std::array<deviceFloat, 3> bm {0, 1.0/3.0, 0.75};
    constexpr std::array<deviceFloat, 3> gm {1.0/3.0, 15.0/16.0, 8.0/15.0};

    deviceFloat delta_t = get_delta_t(mesh);

    for (auto const& e : std::as_const(output_times_)) {
        if ((time >= e) && (time < e + delta_t)) {
            if (global_rank == 0) {
                bar.set_status_text("Writing solution");
                bar.update(0.0);
            }
            mesh.write_complete_data(time, NDG.nodes_, NDG.interpolation_matrices_, data_writer);
        }
    }
    
    if (global_rank == 0) {
        bar.set_status_text("Iteration 0");
        bar.update(0.0);
    }
    
    while (time < t_end) {
        ++timestep;
        delta_t = get_delta_t(mesh);
        if (time + delta_t > t_end) {
            delta_t = t_end - time;
        }

        // Kinda algorithm 62
        deviceFloat t = time + bm[0] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        SEM::Solvers::calculate_wave_fluxes<<<mesh.faces_numBlocks_, mesh.faces_blockSize_, 0, mesh.stream_>>>(mesh.faces_.size(), mesh.faces_.data(), mesh.elements_.data());
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Solvers::compute_dg_wave_derivative<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        SEM::Solvers::rk3_first_step<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.n_elements_, mesh.elements_.data(), delta_t, gm[0]);

        t = time + bm[1] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        SEM::Solvers::calculate_wave_fluxes<<<mesh.faces_numBlocks_, mesh.faces_blockSize_, 0, mesh.stream_>>>(mesh.faces_.size(), mesh.faces_.data(), mesh.elements_.data());
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Solvers::compute_dg_wave_derivative<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        SEM::Solvers::rk3_step<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.n_elements_, mesh.elements_.data(), delta_t, am[1], gm[1]);

        t = time + bm[2] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        SEM::Solvers::calculate_wave_fluxes<<<mesh.faces_numBlocks_, mesh.faces_blockSize_, 0, mesh.stream_>>>(mesh.faces_.size(), mesh.faces_.data(), mesh.elements_.data());
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Solvers::compute_dg_wave_derivative<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        SEM::Solvers::rk3_step<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.n_elements_, mesh.elements_.data(), delta_t, am[2], gm[2]);
        
        time += delta_t;
        for (auto const& e : std::as_const(output_times_)) {
            if ((time >= e) && (time < e + delta_t)) {
                if (global_rank == 0) {
                    bar.set_status_text("Writing solution");
                    bar.update(time/t_end);
                }
                mesh.estimate_error<Polynomial>(NDG.nodes_, NDG.weights_);
                mesh.write_complete_data(time, NDG.nodes_, NDG.interpolation_matrices_, data_writer);
                break;
            }
        }

        if (timestep % mesh.adaptivity_interval_ == 0) {
            if (global_rank == 0) {
                bar.set_status_text("Adapting");
                bar.update(time/t_end);
            }

            mesh.estimate_error<Polynomial>(NDG.nodes_, NDG.weights_);
            mesh.adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
        }

        if (global_rank == 0) {
            std::stringstream ss;
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
            bar.update(time/t_end);
        }
    }

    bool did_write = false;
    for (auto const& e : std::as_const(output_times_)) {
        if ((time >= e) && (time < e + delta_t)) {
            did_write = true;
            break;
        }
    }

    if (!did_write) {
        mesh.estimate_error<Polynomial>(NDG.nodes_, NDG.weights_);
        if (global_rank == 0) {
            bar.set_status_text("Writing solution");
            bar.update(1.0);
        }
        mesh.write_complete_data(time, NDG.nodes_, NDG.interpolation_matrices_, data_writer);
    }
    if (global_rank == 0) {
        bar.set_status_text("Done");
        bar.update(1.0);
    }
    if (global_rank == 0) {
        std::cout << std::endl;
    }
}

auto SEM::Solvers::Solver2D_t::get_delta_t(SEM::Meshes::Mesh2D_t& mesh) const -> deviceFloat {   
    SEM::Solvers::reduce_wave_delta_t<mesh.elements_blockSize_/2><<<mesh.elements_numBlocks_, mesh.elements_blockSize_/2, 0, mesh.stream_>>>(CFL_, mesh.n_elements_, mesh.elements_.data(), mesh.device_delta_t_array_.data());
    mesh.device_delta_t_array_.copy_to(mesh.host_delta_t_array_, mesh.stream_);
    cudaStreamSynchronize(mesh.stream_);

    deviceFloat delta_t_min_local = std::numeric_limits<deviceFloat>::infinity();
    for (int i = 0; i < mesh.elements_numBlocks_; ++i) {
        delta_t_min_local = min(delta_t_min_local, mesh.host_delta_t_array_[i]);
    }

    deviceFloat delta_t_min;
    constexpr MPI_Datatype data_type = (sizeof(deviceFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE; // CHECK this is a bad way of doing this
    MPI_Allreduce(&delta_t_min_local, &delta_t_min, 1, data_type, MPI_MIN, MPI_COMM_WORLD);
    return delta_t_min;
}

__host__ __device__
auto SEM::Solvers::Solver2D_t::x_flux(deviceFloat p, deviceFloat u, deviceFloat v) -> std::array<deviceFloat, 3> {
    return {SEM::Constants::c * u, p, 0};
}

__host__ __device__
auto SEM::Solvers::Solver2D_t::y_flux(deviceFloat p, deviceFloat u, deviceFloat v) -> std::array<deviceFloat, 3> {
    return {SEM::Constants::c * v, 0, p};
}

__device__
void SEM::Solvers::Solver2D_t::matrix_vector_multiply(int N, const deviceFloat* matrix, const deviceFloat* vector, deviceFloat* result) {
    for (int i = 0; i <= N; ++i) {
        result[i] = 0.0;
        for (int j = 0; j <= N; ++j) {
            result[i] +=  matrix[i * (N + 1) + j] * vector[j];
        }
    }
}

__global__
auto SEM::Solvers::calculate_wave_fluxes(size_t N_faces, Face2D_t* faces, const Element2D_t* elements) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < N_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];

        // Computing fluxes
        for (int i = 0; i <= face.N_; ++i) {
            const Vec2<deviceFloat> u_L {face.u_[0][i], face.v_[0][i]};
            const Vec2<deviceFloat> u_R {face.u_[1][i], face.v_[1][i]};

            const deviceFloat w_L = face.p_[0][i] + SEM::Constants::c * u_L.dot(face.normal_);
            const deviceFloat w_R = face.p_[1][i] - SEM::Constants::c * u_R.dot(face.normal_);

            face.p_flux_[i] = SEM::Constants::c * (w_L - w_R) / 2;
            face.u_flux_[i] = face.normal_.x() * (w_L + w_R) / 2;
            face.v_flux_[i] = face.normal_.y() * (w_L + w_R) / 2;
        }
    }
}

// Algorithm 114
__global__
auto SEM::Solvers::compute_dg_wave_derivative(size_t N_elements, Element2D_t* elements, const Face2D_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
        Element2D_t& element = elements[element_index];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2; // CHECK cache?
        const size_t offset_2D = element.N_ * (element.N_ + 1) * (2 * element.N_ + 1) /6;

        // Horizontal direction
        for (int j = 0; j <= element.N_; ++j) {
            for (int i = 0; i <= element.N_; ++i) {
                const std::array<deviceFloat, 3> flux_x = SEM::Solvers::Solver2D_t::x_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);
                const std::array<deviceFloat, 3> flux_y = SEM::Solvers::Solver2D_t::y_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);
            
                element.p_flux_[i] = element.deta_dy_[i * (element.N_ + 1) + j] * flux_x[0] - element.deta_dx_[i * (element.N_ + 1) + j] * flux_y[0];
                element.u_flux_[i] = element.deta_dy_[i * (element.N_ + 1) + j] * flux_x[1] - element.deta_dx_[i * (element.N_ + 1) + j] * flux_y[1];
                element.v_flux_[i] = element.deta_dy_[i * (element.N_ + 1) + j] * flux_x[2] - element.deta_dx_[i * (element.N_ + 1) + j] * flux_y[2];
            }

            SEM::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat + offset_2D, element.p_flux_.data(), element.p_flux_derivative_.data());
            SEM::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat + offset_2D, element.u_flux_.data(), element.u_flux_derivative_.data());
            SEM::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat + offset_2D, element.v_flux_.data(), element.v_flux_derivative_.data());

            // For the boundaries, the numbering increases from the first node to the second. 
            // Inside the element, the ksi and eta coordinates increase from left to right, bottom to top.
            // This means that there is an inconsistency on the top and left edges, and the numbering has to be reversed.
            // This way, the projection from the element edge to the face(s) can always be done in the same way.
            // The same process has to be done when interpolating to the boundaries.
            for (int i = 0; i <= element.N_; ++i) {
                element.p_flux_derivative_[i] += (element.p_flux_extrapolated_[1][j] * lagrange_interpolant_right[offset_1D + i] + element.p_flux_extrapolated_[3][element.N_ - j] * lagrange_interpolant_left[offset_1D + i]) / weights[offset_1D + i];
                element.u_flux_derivative_[i] += (element.u_flux_extrapolated_[1][j] * lagrange_interpolant_right[offset_1D + i] + element.u_flux_extrapolated_[3][element.N_ - j] * lagrange_interpolant_left[offset_1D + i]) / weights[offset_1D + i];
                element.v_flux_derivative_[i] += (element.v_flux_extrapolated_[1][j] * lagrange_interpolant_right[offset_1D + i] + element.v_flux_extrapolated_[3][element.N_ - j] * lagrange_interpolant_left[offset_1D + i]) / weights[offset_1D + i];
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
                const std::array<deviceFloat, 3> flux_x = SEM::Solvers::Solver2D_t::x_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);
                const std::array<deviceFloat, 3> flux_y = SEM::Solvers::Solver2D_t::y_flux(element.p_[i * (element.N_ + 1) + j], element.u_[i * (element.N_ + 1) + j], element.v_[i * (element.N_ + 1) + j]);

                element.p_flux_[j] = -element.dxi_dy_[i * (element.N_ + 1) + j] * flux_x[0] + element.dxi_dx_[i * (element.N_ + 1) + j] * flux_y[0];
                element.u_flux_[j] = -element.dxi_dy_[i * (element.N_ + 1) + j] * flux_x[1] + element.dxi_dx_[i * (element.N_ + 1) + j] * flux_y[1];
                element.v_flux_[j] = -element.dxi_dy_[i * (element.N_ + 1) + j] * flux_x[2] + element.dxi_dx_[i * (element.N_ + 1) + j] * flux_y[2];
            }

            SEM::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat + offset_2D, element.p_flux_.data(), element.p_flux_derivative_.data());
            SEM::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat + offset_2D, element.u_flux_.data(), element.u_flux_derivative_.data());
            SEM::Solvers::Solver2D_t::matrix_vector_multiply(element.N_, derivative_matrices_hat + offset_2D, element.v_flux_.data(), element.v_flux_derivative_.data());

            // For the boundaries, the numbering increases from the first node to the second. 
            // Inside the element, the ksi and eta coordinates increase from left to right, bottom to top.
            // This means that there is an inconsistency on the top and left edges, and the numbering has to be reversed.
            // This way, the projection from the element edge to the face(s) can always be done in the same way.
            // The same process has to be done when interpolating to the boundaries.
            for (int j = 0; j <= element.N_; ++j) {
                element.p_flux_derivative_[j] += (element.p_flux_extrapolated_[2][element.N_ - i] * lagrange_interpolant_right[offset_1D + j] + element.p_flux_extrapolated_[0][i] * lagrange_interpolant_left[offset_1D + j]) / weights[offset_1D + j];
                element.u_flux_derivative_[j] += (element.u_flux_extrapolated_[2][element.N_ - i] * lagrange_interpolant_right[offset_1D + j] + element.u_flux_extrapolated_[0][i] * lagrange_interpolant_left[offset_1D + j]) / weights[offset_1D + j];
                element.v_flux_derivative_[j] += (element.v_flux_extrapolated_[2][element.N_ - i] * lagrange_interpolant_right[offset_1D + j] + element.v_flux_extrapolated_[0][i] * lagrange_interpolant_left[offset_1D + j]) / weights[offset_1D + j];
            }

            for (int j = 0; j <= element.N_; ++j) {
                element.G_p_[i * (element.N_ + 1) + j] = (element.G_p_[i * (element.N_ + 1) + j] - element.p_flux_derivative_[j]) / element.jacobian_[i * (element.N_ + 1) + j];
                element.G_u_[i * (element.N_ + 1) + j] = (element.G_u_[i * (element.N_ + 1) + j] - element.u_flux_derivative_[j]) / element.jacobian_[i * (element.N_ + 1) + j];
                element.G_v_[i * (element.N_ + 1) + j] = (element.G_v_[i * (element.N_ + 1) + j] - element.v_flux_derivative_[j]) / element.jacobian_[i * (element.N_ + 1) + j];
            }
        }
    }
}

__global__
auto SEM::Solvers::rk3_first_step(size_t N_elements, SEM::Entities::Element2D_t* elements, deviceFloat delta_t, deviceFloat g) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
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

__global__
auto SEM::Solvers::rk3_step(size_t N_elements, SEM::Entities::Element2D_t* elements, deviceFloat delta_t, deviceFloat a, deviceFloat g) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
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
