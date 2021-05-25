#include "solvers/Solver2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "helpers/ProgressBar_t.h"
#include "helpers/constants.h"
#include <mpi.h>
#include <limits>

using SEM::Entities::device_vector;
using SEM::Entities::Vec2;
using SEM::Entities::Element2D_t;
using SEM::Entities::Face2D_t;

SEM::Solvers::Solver2D_t::Solver2D_t(deviceFloat CFL, std::vector<deviceFloat> output_times, deviceFloat viscosity) :
        CFL_{CFL},
        output_times_{output_times},
        viscosity_{viscosity} {}

template auto SEM::Solvers::Solver2D_t::solve(const SEM::Entities::NDG_t<SEM::Polynomials::ChebyshevPolynomial_t> &NDG, SEM::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) -> void; // Get with the times c++, it's crazy I have to do this
template auto SEM::Solvers::Solver2D_t::solve(const SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> &NDG, SEM::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) -> void;

template<typename Polynomial>
auto SEM::Solvers::Solver2D_t::solve(const SEM::Entities::NDG_t<Polynomial> &NDG, SEM::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) -> void {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    deviceFloat time = 0.0;
    const deviceFloat t_end = output_times_.back();
    SEM::Helpers::ProgressBar_t bar;
    size_t timestep = 0;

    deviceFloat delta_t = get_delta_t(mesh);
    if (global_rank == 0) {
        bar.set_status_text("Writing solution");
        bar.update(0.0);
    }
    mesh.write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_.data(), data_writer);
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
        deviceFloat t = time;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions();
        SEM::Solvers::calculate_wave_fluxes<<<mesh.faces_numBlocks_, mesh.faces_blockSize_, 0, mesh.stream_>>>(mesh.faces_.size(), mesh.faces_.data(), mesh.elements_.data());
        SEM::Solvers::compute_dg_wave_derivative<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        //SEM::Meshes::rk3_first_step<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_, delta_t, 1.0/3.0);

        t = time + 0.33333333333f * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions();
        SEM::Solvers::calculate_wave_fluxes<<<mesh.faces_numBlocks_, mesh.faces_blockSize_, 0, mesh.stream_>>>(mesh.faces_.size(), mesh.faces_.data(), mesh.elements_.data());
        SEM::Solvers::compute_dg_wave_derivative<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        //SEM::Meshes::rk3_step<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_, delta_t, -5.0/9.0, 15.0/16.0);

        t = time + 0.75f * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions();
        SEM::Solvers::calculate_wave_fluxes<<<mesh.faces_numBlocks_, mesh.faces_blockSize_, 0, mesh.stream_>>>(mesh.faces_.size(), mesh.faces_.data(), mesh.elements_.data());
        SEM::Solvers::compute_dg_wave_derivative<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        //SEM::Meshes::rk3_step<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_, delta_t, -153.0/128.0, 8.0/15.0);
        
        time += delta_t;
        for (auto const& e : std::as_const(output_times_)) {
            if ((time >= e) && (time < e + delta_t)) {
                //SEM::Meshes::estimate_error<Polynomial><<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_.data(), NDG.nodes_.data(), NDG.weights_.data());
                if (global_rank == 0) {
                    bar.set_status_text("Writing solution");
                    bar.update(time/t_end);
                }
                mesh.write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_.data(), data_writer);
                break;
            }
        }
        if (global_rank == 0) {
            std::stringstream ss;
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
            bar.update(time/t_end);
        }

        if (timestep % mesh.adaptivity_interval_ == 0) {
            //SEM::Meshes::estimate_error<Polynomial><<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_.data(), NDG.nodes_.data(), NDG.weights_.data());
            mesh.adapt(NDG.N_max_, NDG.nodes_.data(), NDG.barycentric_weights_.data());
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
        //SEM::Meshes::estimate_error<Polynomial><<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, mesh.stream_>>>(mesh.N_elements_, mesh.elements_.data(), NDG.nodes_.data(), NDG.weights_.data());
        if (global_rank == 0) {
            bar.set_status_text("Writing solution");
            bar.update(1.0);
        }
        mesh.write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_.data(), data_writer);
    }
    if (global_rank == 0) {
        bar.set_status_text("Done");
        bar.update(1.0);
    }
    if (global_rank == 0) {
        std::cout << std::endl;
    }
}

auto SEM::Solvers::Solver2D_t::get_delta_t(SEM::Meshes::Mesh2D_t& mesh) -> deviceFloat {   
    SEM::Solvers::reduce_wave_delta_t<mesh.elements_blockSize_/2><<<mesh.elements_numBlocks_, mesh.elements_blockSize_/2, 0, mesh.stream_>>>(CFL_, mesh.N_elements_, mesh.elements_.data(), mesh.device_delta_t_array_.data());
    mesh.device_delta_t_array_.copy_to(mesh.host_delta_t_array_);

    deviceFloat delta_t_min_local = std::numeric_limits<deviceFloat>::infinity();
    for (int i = 0; i < mesh.elements_numBlocks_; ++i) {
        delta_t_min_local = min(delta_t_min_local, mesh.host_delta_t_array_[i]);
    }

    deviceFloat delta_t_min;
    constexpr MPI_Datatype data_type = (sizeof(deviceFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(&delta_t_min_local, &delta_t_min, 1, data_type, MPI_MIN, MPI_COMM_WORLD);
    return delta_t_min;
}

__global__
void SEM::Solvers::calculate_wave_fluxes(size_t N_faces, Face2D_t* faces, const Element2D_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < N_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];

        // Getting element solution
        const Element2D_t& element_L = elements[face.elements_[0]];
        // Conforming
        if ((face.N_ == element_L.N_) 
                && (face.nodes_[0] == element_L.nodes_[face.elements_side_[0]]) 
                && (face.nodes_[1] == element_L.nodes_[(face.elements_side_[0] + 1) * (!(face.elements_side_[0] == (element_L.nodes_.size() - 1)))])) {
            for (int i = 0; i <= face.N_; ++i) {
                face.p_[0][i] = element_L.p_extrapolated_[face.elements_side_[0]][i];
                face.u_[0][i] = element_L.u_extrapolated_[face.elements_side_[0]][i];
                face.v_[0][i] = element_L.v_extrapolated_[face.elements_side_[0]][i];
            }
        }
        else { // We need to interpolate
            printf("Warning, non-conforming surfaces are not implemented yet.\n");
        }


        const Element2D_t& element_R = elements[face.elements_[1]];
        // Conforming, but reversed
        if ((face.N_ == element_R.N_) 
                && (face.nodes_[1] == element_R.nodes_[face.elements_side_[1]]) 
                && (face.nodes_[0] == element_R.nodes_[(face.elements_side_[1] + 1) * (!(face.elements_side_[1] == (element_R.nodes_.size() - 1)))])) {
            for (int i = 0; i <= face.N_; ++i) {
                face.p_[1][face.N_ - i] = element_R.p_extrapolated_[face.elements_side_[1]][i];
                face.u_[1][face.N_ - i] = element_R.u_extrapolated_[face.elements_side_[1]][i];
                face.v_[1][face.N_ - i] = element_R.v_extrapolated_[face.elements_side_[1]][i];
            }
        }
        else { // We need to interpolate
            printf("Warning, non-conforming surfaces are not implemented yet.\n");
        }

        // Computing fluxes
        for (int i = 0; i <= face.N_; ++i) {
            const Vec2<deviceFloat> u_L {face.u_[0][i], face.v_[0][i]};
            const Vec2<deviceFloat> u_R {face.u_[1][i], face.v_[1][i]};
            const Vec2<deviceFloat> u_prime_L {u_L.dot(face.normal_), u_L.dot(face.tangent_)};
            const Vec2<deviceFloat> u_prime_R {u_R.dot(face.normal_), u_R.dot(face.tangent_)};

            const deviceFloat w_L = (face.p_[0][i] + SEM::Constants::c * u_prime_L.x()) / 2;
            const deviceFloat w_R = (face.p_[1][i] - SEM::Constants::c * u_prime_R.x()) / 2;

            const Vec2<deviceFloat> normal_inv {face.normal_[0], face.tangent_[0]};
            const Vec2<deviceFloat> tangent_inv {face.normal_[1], face.tangent_[1]};

            const Vec2<deviceFloat> velocity_flux {w_L + w_R, 0};

            face.p_flux_[i] = SEM::Constants::c * (w_L - w_R);
            face.u_flux_[i] = velocity_flux.dot(normal_inv);
            face.v_flux_[i] = velocity_flux.dot(tangent_inv);
        }
    }
}

// Algorithm 60 (not really anymore)
__global__
void SEM::Solvers::compute_dg_wave_derivative(size_t N_elements, Element2D_t* elements, const Face2D_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        /*const size_t offset_1D = elements[i].N_ * (elements[i].N_ + 1) /2; // CHECK cache?
        const size_t offset_2D = elements[i].N_ * (elements[i].N_ + 1) * (2 * elements[i].N_ + 1) /6;

        const deviceFloat flux_L = faces[elements[i].faces_[0]].flux_;
        const deviceFloat flux_R = faces[elements[i].faces_[1]].flux_;

        SEM::Meshes::matrix_vector_multiply(elements[i].N_, derivative_matrices_hat + offset_2D, elements[i].phi_, elements[i].q_);
        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].q_[j] = -elements[i].q_[j] - (flux_R * lagrange_interpolant_right[offset_1D + j]
                                                     - flux_L * lagrange_interpolant_left[offset_1D + j]) / weights[offset_1D + j];
            elements[i].q_[j] *= 2.0f/elements[i].delta_x_;
        }*/
    }
}
