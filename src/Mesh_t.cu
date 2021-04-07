#include "Mesh_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include "ProgressBar_t.h"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

SEM::Mesh_t::Mesh_t(size_t N_elements, int initial_N, deviceFloat delta_x_min, deviceFloat x_min, deviceFloat x_max, int adaptivity_interval, cudaStream_t &stream) : 
        N_elements_global_(N_elements), 
        delta_x_min_(delta_x_min), 
        adaptivity_interval_(adaptivity_interval),      
        stream_(stream) {

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    N_elements_per_process_ = (N_elements_global_ + global_size - 1)/global_size;
    N_elements_ = (global_rank == global_size - 1) ? N_elements_per_process_ + N_elements_global_ - N_elements_per_process_ * global_size : N_elements_per_process_;
    if (N_elements_ == N_elements_global_) {
        N_local_boundaries_ = 2;
        N_MPI_boundaries_ = 0;
    }
    else {
        N_local_boundaries_ = 0;
        N_MPI_boundaries_ = 2;
    }

    N_faces_ = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 1; 
    global_element_offset_ = global_rank * N_elements_per_process_;
    initial_N_ = initial_N;
    elements_numBlocks_ = (N_elements_ + elements_blockSize_ - 1) / elements_blockSize_;
    faces_numBlocks_ = (N_faces_ + faces_blockSize_ - 1) / faces_blockSize_;
    boundaries_numBlocks_ = (N_local_boundaries_ + N_MPI_boundaries_ + boundaries_blockSize_ - 1) / boundaries_blockSize_;

    host_delta_t_array_ = std::vector<deviceFloat>(elements_numBlocks_);
    host_refine_array_ = std::vector<unsigned long>(elements_numBlocks_);
    host_boundary_phi_L_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_boundary_phi_R_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_boundary_phi_prime_L_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_boundary_phi_prime_R_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_MPI_boundary_to_element_ = std::vector<size_t>(N_MPI_boundaries_);
    host_MPI_boundary_from_element_ = std::vector<size_t>(N_MPI_boundaries_);
    send_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    receive_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    requests_ = std::vector<MPI_Request>(N_MPI_boundaries_*2);
    statuses_ = std::vector<MPI_Status>(N_MPI_boundaries_*2);

    cudaMalloc(&elements_, (N_elements_ + N_local_boundaries_ + N_MPI_boundaries_) * sizeof(Element_t));
    cudaMalloc(&faces_, N_faces_ * sizeof(Face_t));
    cudaMalloc(&local_boundary_to_element_, N_local_boundaries_ * sizeof(size_t));
    cudaMalloc(&MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t));
    cudaMalloc(&MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t));
    cudaMalloc(&device_delta_t_array_, elements_numBlocks_ * sizeof(deviceFloat));
    cudaMalloc(&device_refine_array_, elements_numBlocks_ * sizeof(unsigned long));
    cudaMalloc(&device_boundary_phi_L_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_R_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_prime_L_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_prime_R_, N_MPI_boundaries_ * sizeof(deviceFloat));

    const deviceFloat delta_x = (x_max - x_min)/N_elements_global_;
    const deviceFloat x_min_local = x_min + delta_x * global_rank * N_elements_per_process_;
    const deviceFloat x_max_local = x_min_local + N_elements_ * delta_x;

    SEM::build_elements<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, initial_N_, elements_, x_min_local, x_max_local);
    SEM::build_boundaries<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_, N_elements_global_, N_local_boundaries_, N_MPI_boundaries_, elements_, global_element_offset_, local_boundary_to_element_, MPI_boundary_to_element_, MPI_boundary_from_element_);
    SEM::build_faces<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_);

    cudaMemcpy(host_MPI_boundary_to_element_.data(), MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_from_element_.data(), MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
}

SEM::Mesh_t::~Mesh_t() {
    SEM::free_elements<<<elements_numBlocks_, elements_blockSize_>>>(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_, elements_);
    cudaFree(elements_);
    cudaFree(faces_);
    cudaFree(local_boundary_to_element_);
    cudaFree(MPI_boundary_to_element_);
    cudaFree(MPI_boundary_from_element_);
    cudaFree(device_delta_t_array_);
    cudaFree(device_refine_array_);
    cudaFree(device_boundary_phi_L_);
    cudaFree(device_boundary_phi_R_);
    cudaFree(device_boundary_phi_prime_L_);
    cudaFree(device_boundary_phi_prime_R_);
}

void SEM::Mesh_t::set_initial_conditions(const deviceFloat* nodes) {
    SEM::initial_conditions<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, nodes);
}

void SEM::Mesh_t::print() {
    std::vector<Face_t> host_faces(N_faces_);
    std::vector<Element_t> host_elements(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_);
    std::vector<size_t> host_local_boundary_to_element(N_local_boundaries_);

    cudaMemcpy(host_faces.data(), faces_, N_faces_ * sizeof(Face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_elements.data(), elements_, (N_elements_ + N_local_boundaries_ + N_MPI_boundaries_) * sizeof(Element_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_local_boundary_to_element.data(), local_boundary_to_element_, N_local_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_to_element_.data(), MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_from_element_.data(), MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);

    // Invalidate GPU pointers, or else they will be deleted on the CPU, where they point to random stuff
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        host_elements[i].phi_ = nullptr;
        host_elements[i].q_ = nullptr;
        host_elements[i].ux_ = nullptr;
        host_elements[i].phi_prime_ = nullptr;
        host_elements[i].intermediate_ = nullptr;
    }

    std::cout << "N elements global: " << N_elements_global_ << std::endl;
    std::cout << "N elements local: " << N_elements_ << std::endl;
    std::cout << "N faces: " << N_faces_ << std::endl;
    std::cout << "N local boundaries: " << N_local_boundaries_ << std::endl;
    std::cout << "N MPI boundaries: " << N_MPI_boundaries_ << std::endl;
    std::cout << "Global element offset: " << global_element_offset_ << std::endl;
    std::cout << "Number of elements per process: " << N_elements_per_process_ << std::endl;
    std::cout << "Initial N: " << initial_N_ << std::endl;

    std::cout << std::endl << "Phi interpolated: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].phi_L_ << " ";
        std::cout << host_elements[i].phi_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime interpolated: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].phi_prime_L_ << " ";
        std::cout << host_elements[i].phi_prime_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "x: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].x_[0] << " ";
        std::cout << host_elements[i].x_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "Neighbouring faces: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].faces_[0] << " ";
        std::cout << host_elements[i].faces_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "N: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].N_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "delta x: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].delta_x_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "Fluxes: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].flux_ << std::endl;
    }

    std::cout << std::endl << "Derivative fluxes: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].derivative_flux_ << std::endl;
    }

    std::cout << std::endl << "Non linear fluxes: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].nl_flux_ << std::endl;
    }

    std::cout << std::endl << "Elements: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].elements_[0] << " ";
        std::cout << host_faces[i].elements_[1] << std::endl;
    }

    std::cout << std::endl << "Local boundaries elements: " << std::endl;
    for (size_t i = 0; i < N_local_boundaries_; ++i) {
        std::cout << '\t' << "Local boundary " << i << ": ";
        std::cout << '\t';
        std::cout << host_local_boundary_to_element[i] << std::endl;
    }

    std::cout << std::endl << "MPI boundaries to elements: " << std::endl;
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "MPI boundary " << i << ": ";
        std::cout << '\t';
        std::cout << host_MPI_boundary_to_element_[N_local_boundaries_ + i] << std::endl;
    }

    std::cout << std::endl << "MPI boundaries from elements: " << std::endl;
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "MPI boundary " << i << ": ";
        std::cout << '\t';
        std::cout << host_MPI_boundary_from_element_[N_local_boundaries_ + i] << std::endl;
    }
    std::cout << std::endl;
}

void SEM::Mesh_t::write_file_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& coordinates, const std::vector<deviceFloat>& velocity, const std::vector<deviceFloat>& du_dx, const std::vector<deviceFloat>& intermediate, const std::vector<deviceFloat>& x_L, const std::vector<deviceFloat>& x_R, const std::vector<int>& N, const std::vector<deviceFloat>& sigma, const bool* refine, const bool* coarsen, const std::vector<deviceFloat>& error, const std::vector<deviceFloat>& delta_x) {
    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    std::stringstream ss;
    std::ofstream file;
    ss << "output_t" << std::setprecision(9) << std::fixed << time << "_proc" << std::setfill('0') << std::setw(6) << rank << ".dat";
    file.open(save_dir / ss.str());

    file << "TITLE = \"Velocity at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\", \"U_x_prime\", \"intermediate\"" << std::endl;

    for (size_t i = 0; i < N_elements; ++i) {
        file << "ZONE T= \"Zone " << i + 1 << "\",  I= " << N_interpolation_points << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

        for (size_t j = 0; j < N_interpolation_points; ++j) {
            file       << std::setw(12) << coordinates[i*N_interpolation_points + j] 
                << " " << std::setw(12) << velocity[i*N_interpolation_points + j]
                << " " << std::setw(12) << du_dx[i*N_interpolation_points + j]
                << " " << std::setw(12) << intermediate[i*N_interpolation_points + j] << std::endl;
        }
    }

    file.close();

    std::stringstream ss_element;
    std::ofstream file_element;
    ss_element << "output_element_t" << std::setprecision(9) << std::fixed << time << "_proc" << std::setfill('0') << std::setw(6) << rank << ".dat";
    file_element.open(save_dir / ss_element.str());

    file_element << "TITLE = \"Element values at t= " << time << "\"" << std::endl
                 << "VARIABLES = \"X\", \"X_L\", \"X_R\", \"N\", \"sigma\", \"refine\", \"coarsen\", \"error\", \"delta_x\"" << std::endl
                 << "ZONE T= \"Zone     1\",  I= " << N_elements << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (size_t j = 0; j < N_elements; ++j) {
        file_element << std::setw(12) << (x_L[j] + x_R[j]) * 0.5
              << " " << std::setw(12) << x_L[j]
              << " " << std::setw(12) << x_R[j]
              << " " << std::setw(12) << N[j]
              << " " << std::setw(12) << sigma[j]
              << " " << std::setw(12) << refine[j]
              << " " << std::setw(12) << coarsen[j]
              << " " << std::setw(12) << error[j] 
              << " " << std::setw(12) << delta_x[j] << std::endl;
    }

    file_element.close();
}

void SEM::Mesh_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices) {
    deviceFloat* x;
    deviceFloat* phi;
    deviceFloat* phi_prime;
    deviceFloat* intermediate;
    deviceFloat* x_L;
    deviceFloat* x_R;
    int* N;
    deviceFloat* sigma;
    bool* refine;
    bool* coarsen;
    deviceFloat* error;
    deviceFloat* delta_x;
    std::vector<deviceFloat> host_x(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_phi(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_phi_prime(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_intermediate(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_x_L(N_elements_);
    std::vector<deviceFloat> host_x_R(N_elements_);
    std::vector<int> host_N(N_elements_);
    std::vector<deviceFloat> host_sigma(N_elements_);
    bool* host_refine = new bool[N_elements_]; // Vectors of bools can be messed-up by some implementations
    bool* host_coarsen = new bool[N_elements_]; // Like they won't be an array of bools but packed in integers, in which case getting them from Cuda will fail.
    std::vector<deviceFloat> host_error(N_elements_);
    std::vector<deviceFloat> host_delta_x(N_elements_);
    cudaMalloc(&x, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi_prime, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&intermediate, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&x_L, N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&x_R, N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&N, N_elements_ * sizeof(int));
    cudaMalloc(&sigma, N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&refine, N_elements_ * sizeof(bool));
    cudaMalloc(&coarsen, N_elements_ * sizeof(bool));
    cudaMalloc(&error, N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&delta_x, N_elements_ * sizeof(deviceFloat));

    SEM::get_solution<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, x, phi, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error, delta_x);
    
    cudaMemcpy(host_x.data(), x , N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi.data(), phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime.data(), phi_prime, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_intermediate.data(), intermediate, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_L.data(), x_L, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_R.data(), x_R, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_N.data(), N, N_elements_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sigma.data(), sigma, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_refine, refine, N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_coarsen, coarsen, N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_error.data(), error, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_delta_x.data(), delta_x, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    write_file_data(N_interpolation_points, N_elements_, time, global_rank, host_x, host_phi, host_phi_prime, host_intermediate, host_x_L, host_x_R, host_N, host_sigma, host_refine, host_coarsen, host_error, host_delta_x);

    delete[] host_refine;
    delete[] host_coarsen;
    cudaFree(x);
    cudaFree(phi);
    cudaFree(phi_prime);
    cudaFree(intermediate);
    cudaFree(x_L);
    cudaFree(x_R);
    cudaFree(N);
    cudaFree(sigma);
    cudaFree(refine);
    cudaFree(coarsen);
    cudaFree(error);
    cudaFree(delta_x);
}

template void SEM::Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<ChebyshevPolynomial_t> &NDG, deviceFloat viscosity); // Get with the times c++, it's crazy I have to do this
template void SEM::Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<LegendrePolynomial_t> &NDG, deviceFloat viscosity);

template<typename Polynomial>
void SEM::Mesh_t::solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG, deviceFloat viscosity) {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    deviceFloat time = 0.0;
    const deviceFloat t_end = output_times.back();
    ProgressBar_t bar;
    size_t timestep = 0;

    deviceFloat delta_t = get_delta_t(CFL);
    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
    if (global_rank == 0) {
        bar.update(0.0);
        bar.set_status_text("Iteration 0");
    }
    
    while (time < t_end) {
        ++timestep;
        delta_t = get_delta_t(CFL);
        if (time + delta_t > t_end) {
            delta_t = t_end - time;
        }

        // Kinda algorithm 62
        deviceFloat t = time;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::interpolate_q_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        SEM::calculate_q_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative2<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(viscosity, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_first_step<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, delta_t, 1.0/3.0);

        t = time + 0.33333333333f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::interpolate_q_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        SEM::calculate_q_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative2<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(viscosity, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, delta_t, -5.0/9.0, 15.0/16.0);

        t = time + 0.75f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::interpolate_q_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        SEM::calculate_q_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative2<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(viscosity, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, delta_t, -153.0/128.0, 8.0/15.0);

        time += delta_t;
        if (global_rank == 0) {
            std::stringstream ss;
            bar.update(time/t_end);
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
        }
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                SEM::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.nodes_, NDG.weights_);
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
                break;
            }
        }

        if (timestep % adaptivity_interval_ == 0) {
            SEM::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.nodes_, NDG.weights_);
            adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
        }
    }
    if (global_rank == 0) {
        std::cout << std::endl;
    }

    bool did_write = false;
    for (auto const& e : std::as_const(output_times)) {
        if ((time >= e) && (time < e + delta_t)) {
            did_write = true;
            break;
        }
    }

    if (!did_write) {
        SEM::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, NDG.nodes_, NDG.weights_);
        write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
    }
}

deviceFloat SEM::Mesh_t::get_delta_t(const deviceFloat CFL) {   
    SEM::reduce_delta_t<elements_blockSize_/2><<<elements_numBlocks_, elements_blockSize_/2, 0, stream_>>>(CFL, N_elements_, elements_, device_delta_t_array_);
    cudaMemcpy(host_delta_t_array_.data(), device_delta_t_array_, elements_numBlocks_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    double delta_t_min_local = std::numeric_limits<double>::infinity();
    for (int i = 0; i < elements_numBlocks_; ++i) {
        delta_t_min_local = min(delta_t_min_local, host_delta_t_array_[i]);
    }

    double delta_t_min;
    MPI_Allreduce(&delta_t_min_local, &delta_t_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return delta_t_min;
}

void SEM::Mesh_t::adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) {
    // CHECK needs to rebuild boundaries
    SEM::reduce_refine<elements_blockSize_/2><<<elements_numBlocks_, elements_blockSize_/2, 0, stream_>>>(N_elements_, delta_x_min_, elements_, device_refine_array_);
    cudaMemcpy(host_refine_array_.data(), device_refine_array_, elements_numBlocks_ * sizeof(unsigned long), cudaMemcpyDeviceToHost);

    unsigned long long additional_elements = 0;
    for (int i = 0; i < elements_numBlocks_; ++i) {
        additional_elements += host_refine_array_[i];
        host_refine_array_[i] = additional_elements - host_refine_array_[i]; // Current block offset
    }

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    std::vector<unsigned long long> additional_elements_global(global_size);
    MPI_Allgather(&additional_elements, 1, MPI_UNSIGNED_LONG_LONG, additional_elements_global.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    size_t N_additional_elements_previous = 0;
    for (int i = 0; i < global_rank; ++i) {
        N_additional_elements_previous += additional_elements_global[i];
    }
    const size_t global_element_offset_current = global_element_offset_ + N_additional_elements_previous;
    size_t N_additional_elements_global = 0;
    for (int i = 0; i < global_size; ++i) {
        N_additional_elements_global += additional_elements_global[i];
    }
    N_elements_global_ += N_additional_elements_global;
    const size_t global_element_offset_end_current = global_element_offset_current + N_elements_ + additional_elements - 1;

    const size_t N_elements_per_process_old = N_elements_per_process_;
    N_elements_per_process_ = (N_elements_global_ + global_size - 1)/global_size;
    global_element_offset_ = global_rank * N_elements_per_process_;
    const size_t global_element_offset_end = min(global_element_offset_ + N_elements_per_process_ - 1, N_elements_global_ - 1);

    if ((additional_elements == 0) && (global_element_offset_ == global_element_offset_current) && (global_element_offset_end == global_element_offset_end_current)) {
        SEM::p_adapt<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, N_max, nodes, barycentric_weights);

        if (N_additional_elements_previous > 0 || ((global_element_offset_ == 0) && (N_additional_elements_global > 0))) {
            SEM::adjust_boundaries<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_, N_elements_global_, N_MPI_boundaries_, global_element_offset_, MPI_boundary_to_element_, MPI_boundary_from_element_);
            cudaMemcpy(host_MPI_boundary_to_element_.data(), MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_MPI_boundary_from_element_.data(), MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
        }

        return;
    }

    cudaMemcpy(device_refine_array_, host_refine_array_.data(), elements_numBlocks_ * sizeof(unsigned long), cudaMemcpyHostToDevice);

    Element_t* new_elements;

    cudaMalloc(&new_elements, (N_elements_ + additional_elements) * sizeof(Element_t));

    SEM::hp_adapt<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, new_elements, device_refine_array_, delta_x_min_, N_max, nodes, barycentric_weights);

    SEM::free_elements<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_, elements_);
    cudaFree(elements_);
    
    const size_t N_elements_old = N_elements_;
    N_elements_ = (global_rank == global_size - 1) ? N_elements_per_process_ + N_elements_global_ - N_elements_per_process_ * global_size : N_elements_per_process_;
    N_faces_ = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 1; 
    elements_numBlocks_ = (N_elements_ + elements_blockSize_ - 1) / elements_blockSize_;
    faces_numBlocks_ = (N_faces_ + faces_blockSize_ - 1) / faces_blockSize_;
    boundaries_numBlocks_ = (N_local_boundaries_ + N_MPI_boundaries_ + boundaries_blockSize_ - 1) / boundaries_blockSize_;

    cudaFree(faces_);
    cudaMalloc(&faces_, N_faces_ * sizeof(Face_t));
    SEM::build_faces<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(N_faces_, faces_);

    cudaMalloc(&elements_, (N_elements_ + N_local_boundaries_ + N_MPI_boundaries_) * sizeof(Element_t));

    const size_t N_elements_send_left = (global_element_offset_ > global_element_offset_current) ? global_element_offset_ - global_element_offset_current : 0;
    const size_t N_elements_recv_left = (global_element_offset_current > global_element_offset_) ? global_element_offset_current - global_element_offset_ : 0;
    const size_t N_elements_send_right = (global_element_offset_end_current > global_element_offset_end) ? global_element_offset_end_current - global_element_offset_end : 0;
    const size_t N_elements_recv_right = (global_element_offset_end > global_element_offset_end_current) ? global_element_offset_end - global_element_offset_end_current : 0;

    if (N_elements_send_left + N_elements_recv_left + N_elements_send_right + N_elements_recv_right > 0) {
        std::vector<Element_t> elements_send_left(N_elements_send_left);
        std::vector<Element_t> elements_recv_left(N_elements_recv_left);
        std::vector<Element_t> elements_send_right(N_elements_send_right);
        std::vector<Element_t> elements_recv_right(N_elements_recv_right);

        cudaMemcpy(elements_send_left.data(), new_elements, N_elements_send_left * sizeof(Element_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(elements_send_right.data(), new_elements + N_elements_ - N_elements_send_right, N_elements_send_right * sizeof(Element_t), cudaMemcpyDeviceToHost);

        for (auto& element: elements_send_left) {
            element.phi_ = nullptr; // Those are GPU pointers, deleting them would delete random memory
            element.q_ = nullptr;
            element.ux_ = nullptr;
            element.phi_prime_ = nullptr;
            element.intermediate_ = nullptr;
        }

        for (auto& element: elements_send_right) {
            element.phi_ = nullptr; // Those are GPU pointers, deleting them would delete random memory
            element.q_ = nullptr;
            element.ux_ = nullptr;
            element.phi_prime_ = nullptr;
            element.intermediate_ = nullptr;
        }

        std::vector<std::vector<deviceFloat>> phi_arrays_send_left(N_elements_send_left);
        std::vector<std::vector<deviceFloat>> phi_arrays_recv_left(N_elements_recv_left);
        std::vector<deviceFloat*> phi_arrays_send_left_host(N_elements_send_left);
        std::vector<deviceFloat*> phi_arrays_recv_left_host(N_elements_recv_left);
        for (int i = 0; i < N_elements_send_left; ++i) {
            phi_arrays_send_left[i] = std::vector<deviceFloat>(elements_send_left[i].N_ + 1);
            cudaMalloc(&phi_arrays_send_left_host[i], (elements_send_left[i].N_ + 1) * sizeof(deviceFloat));
        }
        deviceFloat** phi_arrays_send_left_device;
        deviceFloat** phi_arrays_recv_left_device;
        cudaMalloc(&phi_arrays_send_left_device, N_elements_send_left * sizeof(deviceFloat*)); 
        cudaMalloc(&phi_arrays_recv_left_device, N_elements_recv_left * sizeof(deviceFloat*)); 
        cudaMemcpy(phi_arrays_send_left_device, phi_arrays_send_left_host.data(), N_elements_send_left * sizeof(deviceFloat*), cudaMemcpyHostToDevice);

        std::vector<std::vector<deviceFloat>> phi_arrays_send_right(N_elements_send_right);
        std::vector<std::vector<deviceFloat>> phi_arrays_recv_right(N_elements_recv_right);
        std::vector<deviceFloat*> phi_arrays_send_right_host(N_elements_send_right);
        std::vector<deviceFloat*> phi_arrays_recv_right_host(N_elements_recv_right);
        for (int i = 0; i < N_elements_send_right; ++i) {
            phi_arrays_send_right[i] = std::vector<deviceFloat>(elements_send_right[i].N_ + 1);
            cudaMalloc(&phi_arrays_send_right_host[i], (elements_send_right[i].N_ + 1) * sizeof(deviceFloat));
        }
        deviceFloat** phi_arrays_send_right_device;
        deviceFloat** phi_arrays_recv_right_device;
        cudaMalloc(&phi_arrays_send_right_device, N_elements_send_right * sizeof(deviceFloat*)); 
        cudaMalloc(&phi_arrays_recv_right_device, N_elements_recv_right * sizeof(deviceFloat*)); 
        cudaMemcpy(phi_arrays_send_right_device, phi_arrays_send_right_host.data(), N_elements_send_right * sizeof(deviceFloat*), cudaMemcpyHostToDevice);

        SEM::get_phi<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_send_left, new_elements, phi_arrays_send_left_device);
        SEM::get_phi<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_send_right, new_elements + N_elements_ - N_elements_send_right, phi_arrays_send_right_device);

        for (int i = 0; i < N_elements_send_left; ++i) {
            cudaMemcpy(phi_arrays_send_left[i].data(), phi_arrays_send_left_host[i], (elements_send_left[i].N_ + 1) * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
        }

        for (int i = 0; i < N_elements_send_right; ++i) {
            cudaMemcpy(phi_arrays_send_right[i].data(), phi_arrays_send_right_host[i], (elements_send_right[i].N_ + 1) * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
        }

        for (int i = 0; i < N_elements_send_left; ++i) {
            cudaFree(phi_arrays_send_left_host[i]);
        }
        cudaFree(phi_arrays_send_left_device);

        for (int i = 0; i < N_elements_send_right; ++i) {
            cudaFree(phi_arrays_send_right_host[i]);
        }
        cudaFree(phi_arrays_send_right_device);

        std::vector<int> left_origins(N_elements_recv_left);
        std::vector<int> right_origins(N_elements_recv_right);
        for (int i = 0; i < N_elements_recv_left; ++i) {
            const int index = global_element_offset_ + i;
            int process_end_index = -1;
            for (int rank = 0; rank < global_rank; ++rank) { 
                process_end_index += N_elements_per_process_old + additional_elements_global[rank];
                if (process_end_index >= index) {
                    left_origins[i] = rank;
                    break;
                }
            }
        }
        for (int i = 0; i < N_elements_recv_right; ++i) {
            const int index = global_element_offset_end_current + i + 1;
            int process_start_index = 0;
            for (int rank = 1; rank < global_size; ++rank) { 
                process_start_index += N_elements_per_process_old + additional_elements_global[rank - 1];
                if (process_start_index >= index) {
                    right_origins[i] = rank;
                    break;
                }
            }
        }
        
        std::vector<MPI_Request> adaptivity_requests(3 * (N_elements_send_left + N_elements_recv_left + N_elements_send_right + N_elements_recv_right));
        std::vector<MPI_Status> adaptivity_statuses(3 * (N_elements_send_left + N_elements_recv_left + N_elements_send_right + N_elements_recv_right));
        constexpr MPI_Datatype data_type = (sizeof(deviceFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

        for (int i = 0; i < N_elements_send_left; ++i) {
            const int index = global_element_offset_current + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&elements_send_left[i].N_, 1, MPI_INT, destination, 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right]);
        }

        for (int i = 0; i < N_elements_send_right; ++i) {
            const int index = global_element_offset_end + 1 + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&elements_send_right[i].N_, 1, MPI_INT, destination, 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + N_elements_send_left]);
        }

        for (int i = 0; i < N_elements_recv_left; ++i) {
            const int index = global_element_offset_ + i;

            MPI_Irecv(&elements_recv_left[i].N_, 1, MPI_INT, left_origins[i], 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i]);
        }

        for (int i = 0; i < N_elements_recv_right; ++i) {
            const int index = global_element_offset_end_current + i + 1;

            MPI_Irecv(&elements_recv_right[i].N_, 1, MPI_INT, right_origins[i], 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i + N_elements_recv_left]);
        }

        MPI_Waitall(N_elements_recv_left + N_elements_recv_right, adaptivity_requests.data(), adaptivity_statuses.data());

        for (int i = 0; i < N_elements_recv_left; ++i) {
            cudaMalloc(&phi_arrays_recv_left_host[i], (elements_recv_left[i].N_ + 1) * sizeof(deviceFloat));
            phi_arrays_recv_left[i] = std::vector<deviceFloat>(elements_recv_left[i].N_ + 1);
        }
        cudaMemcpy(phi_arrays_recv_left_device, phi_arrays_recv_left_host.data(), N_elements_recv_left * sizeof(deviceFloat*), cudaMemcpyHostToDevice);

        for (int i = 0; i < N_elements_recv_right; ++i) {
            cudaMalloc(&phi_arrays_recv_right_host[i], (elements_recv_right[i].N_ + 1) * sizeof(deviceFloat));
            phi_arrays_recv_right[i] = std::vector<deviceFloat>(elements_recv_right[i].N_ + 1);
        }
        cudaMemcpy(phi_arrays_recv_right_device, phi_arrays_recv_right_host.data(), N_elements_recv_right * sizeof(deviceFloat*), cudaMemcpyHostToDevice);

        for (int i = 0; i < N_elements_send_left; ++i) {
            const int index = global_element_offset_current + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&elements_send_left[i].x_[0], 2, data_type, destination, 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + N_elements_send_left + N_elements_send_right]);
            MPI_Isend(phi_arrays_send_left[i].data(), elements_send_left[i].N_ + 1, data_type, destination, 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + 2 * N_elements_send_left + 2 * N_elements_send_right]);
        }

        for (int i = 0; i < N_elements_send_right; ++i) {
            const int index = global_element_offset_end + 1 + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&elements_send_right[i].x_[0], 2, data_type, destination, 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + 2 * N_elements_send_left + N_elements_send_right]);
            MPI_Isend(phi_arrays_send_right[i].data(), elements_send_right[i].N_ + 1, data_type, destination, 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + 3 * N_elements_send_left + 2 * N_elements_send_right]);
        }

        for (int i = 0; i < N_elements_recv_left; ++i) {
            const int index = global_element_offset_ + i;

            MPI_Irecv(&elements_recv_left[i].x_[0], 2, data_type, left_origins[i], 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + N_elements_recv_left + N_elements_recv_right]);
            MPI_Irecv(phi_arrays_recv_left[i].data(), elements_recv_left[i].N_ + 1, data_type, left_origins[i], 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 2 * N_elements_recv_left + 2 * N_elements_recv_right]);
        }

        for (int i = 0; i < N_elements_recv_right; ++i) {
            const int index = global_element_offset_end_current + i + 1;

            MPI_Irecv(&elements_recv_right[i].x_[0], 2, data_type, right_origins[i], 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + 2 * N_elements_recv_left + N_elements_recv_right]);
            MPI_Irecv(phi_arrays_recv_right[i].data(), elements_recv_right[i].N_ + 1, data_type, right_origins[i], 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 2 * N_elements_recv_right]);
        }

        MPI_Waitall(2 * N_elements_recv_left + 2 * N_elements_recv_right, adaptivity_requests.data() + N_elements_recv_left + N_elements_recv_right, adaptivity_statuses.data() + N_elements_recv_left + N_elements_recv_right);
        
        for (int i = 0; i < N_elements_recv_left; ++i) {
            elements_recv_left[i].delta_x_ = elements_recv_left[i].x_[1] - elements_recv_left[i].x_[0];
            elements_recv_left[i].faces_ = {static_cast<size_t>(i), static_cast<size_t>(i) + 1};
            cudaMemcpy(phi_arrays_recv_left_host[i], phi_arrays_recv_left[i].data(), (elements_recv_left[i].N_ + 1) * sizeof(deviceFloat), cudaMemcpyHostToDevice);
        }

        for (int i = 0; i < N_elements_recv_right; ++i) {
            elements_recv_right[i].delta_x_ = elements_recv_right[i].x_[1] - elements_recv_right[i].x_[0];
            elements_recv_right[i].faces_ = {N_elements_ - N_elements_recv_right + static_cast<size_t>(i), N_elements_ - N_elements_recv_right + static_cast<size_t>(i) + 1};
            cudaMemcpy(phi_arrays_recv_right_host[i], phi_arrays_recv_right[i].data(), (elements_recv_right[i].N_ + 1) * sizeof(deviceFloat), cudaMemcpyHostToDevice);
        }

        cudaMemcpy(elements_, elements_recv_left.data(), N_elements_recv_left * sizeof(Element_t), cudaMemcpyHostToDevice);
        cudaMemcpy(elements_ + N_elements_ - N_elements_recv_right, elements_recv_right.data(), N_elements_recv_right * sizeof(Element_t), cudaMemcpyHostToDevice);
        SEM::put_phi<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_recv_left, elements_, phi_arrays_recv_left_device);
        SEM::put_phi<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_recv_right, elements_ + N_elements_ - N_elements_recv_right, phi_arrays_recv_right_device);

        for (int i = 0; i < N_elements_recv_left; ++i) {
            cudaFree(phi_arrays_recv_left_host[i]);
        }
        cudaFree(phi_arrays_recv_left_device);

        for (int i = 0; i < N_elements_recv_right; ++i) {
            cudaFree(phi_arrays_recv_right_host[i]);
        }
        cudaFree(phi_arrays_recv_right_device);

        // We also wait for the send requests
        MPI_Waitall(3 * N_elements_send_left + 3 * N_elements_send_right, adaptivity_requests.data() + 3 * N_elements_recv_left + 3 * N_elements_recv_right, adaptivity_statuses.data() + 3 * N_elements_recv_left + 3 * N_elements_recv_right);
    }

    SEM::move_elements<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_ - N_elements_recv_left - N_elements_recv_right, new_elements, elements_, N_elements_send_left, N_elements_recv_left);

    SEM::free_elements<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_old + additional_elements, new_elements);
    cudaFree(new_elements);

    host_delta_t_array_ = std::vector<deviceFloat>(elements_numBlocks_);
    host_refine_array_ = std::vector<unsigned long>(elements_numBlocks_);
    host_boundary_phi_L_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_boundary_phi_R_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_boundary_phi_prime_L_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_boundary_phi_prime_R_ = std::vector<deviceFloat>(N_MPI_boundaries_);
    host_MPI_boundary_to_element_ = std::vector<size_t>(N_MPI_boundaries_);
    host_MPI_boundary_from_element_ = std::vector<size_t>(N_MPI_boundaries_);
    send_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    receive_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    requests_ = std::vector<MPI_Request>(N_MPI_boundaries_*2);
    statuses_ = std::vector<MPI_Status>(N_MPI_boundaries_*2);

    cudaFree(local_boundary_to_element_);
    cudaFree(MPI_boundary_to_element_);
    cudaFree(MPI_boundary_from_element_);
    cudaFree(device_delta_t_array_);
    cudaFree(device_refine_array_);
    cudaFree(device_boundary_phi_L_);
    cudaFree(device_boundary_phi_R_);
    cudaFree(device_boundary_phi_prime_L_);
    cudaFree(device_boundary_phi_prime_R_);
    cudaMalloc(&local_boundary_to_element_, N_local_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t));
    cudaMalloc(&MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t));
    cudaMalloc(&device_delta_t_array_, elements_numBlocks_ * sizeof(deviceFloat));
    cudaMalloc(&device_refine_array_, elements_numBlocks_ * sizeof(unsigned long));
    cudaMalloc(&device_boundary_phi_L_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_R_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_prime_L_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_prime_R_, N_MPI_boundaries_ * sizeof(deviceFloat));

    SEM::build_boundaries<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_, N_elements_global_, N_local_boundaries_, N_MPI_boundaries_, elements_, global_element_offset_, local_boundary_to_element_, MPI_boundary_to_element_, MPI_boundary_from_element_);

    cudaMemcpy(host_MPI_boundary_to_element_.data(), MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_from_element_.data(), MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
}

void SEM::Mesh_t::boundary_conditions() {
    SEM::local_boundaries<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_, N_local_boundaries_, elements_, local_boundary_to_element_);
    SEM::get_MPI_boundaries<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_, N_local_boundaries_, N_MPI_boundaries_, elements_, faces_, device_boundary_phi_L_, device_boundary_phi_R_, device_boundary_phi_prime_L_, device_boundary_phi_prime_R_);
    
    cudaMemcpy(host_boundary_phi_L_.data(), device_boundary_phi_L_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_boundary_phi_R_.data(), device_boundary_phi_R_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_boundary_phi_prime_L_.data(), device_boundary_phi_prime_L_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_boundary_phi_prime_R_.data(), device_boundary_phi_prime_R_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        send_buffers_[i] = {host_boundary_phi_L_[i],
                            host_boundary_phi_R_[i],
                            host_boundary_phi_prime_L_[i],
                            host_boundary_phi_prime_R_[i]};
        const int destination = host_MPI_boundary_to_element_[i]/N_elements_per_process_;

        MPI_Irecv(&receive_buffers_[i][0], 4, MPI_DOUBLE, destination, host_MPI_boundary_from_element_[i], MPI_COMM_WORLD, &requests_[i]);
        MPI_Isend(&send_buffers_[i][0], 4, MPI_DOUBLE, destination, host_MPI_boundary_to_element_[i], MPI_COMM_WORLD, &requests_[i + N_MPI_boundaries_]);
    }

    MPI_Waitall(N_MPI_boundaries_, requests_.data(), statuses_.data()); // CHECK maybe MPI barrier?

    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        host_boundary_phi_L_[i] = receive_buffers_[i][0];
        host_boundary_phi_R_[i] = receive_buffers_[i][1];
        host_boundary_phi_prime_L_[i] = receive_buffers_[i][2];
        host_boundary_phi_prime_R_[i] = receive_buffers_[i][3];
    }

    cudaMemcpy(device_boundary_phi_L_, host_boundary_phi_L_.data(), N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(device_boundary_phi_R_, host_boundary_phi_R_.data(), N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(device_boundary_phi_prime_L_, host_boundary_phi_prime_L_.data(), N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(device_boundary_phi_prime_R_, host_boundary_phi_prime_R_.data(), N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);

    SEM::put_MPI_boundaries<<<boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_, N_local_boundaries_, N_MPI_boundaries_, elements_, device_boundary_phi_L_, device_boundary_phi_R_, device_boundary_phi_prime_L_, device_boundary_phi_prime_R_);
    
    // We also wait for the send requests
    MPI_Waitall(N_MPI_boundaries_, requests_.data() + N_MPI_boundaries_, statuses_.data() + N_MPI_boundaries_);
}

__global__
void SEM::rk3_first_step(size_t N_elements, Element_t* elements, deviceFloat delta_t, deviceFloat g) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j){
            elements[i].intermediate_[j] = elements[i].phi_prime_[j];
            elements[i].phi_[j] += g * delta_t * elements[i].intermediate_[j];
        }
    }
}

__global__
void SEM::rk3_step(size_t N_elements, Element_t* elements, deviceFloat delta_t, deviceFloat a, deviceFloat g) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j){
            elements[i].intermediate_[j] = a * elements[i].intermediate_[j] + elements[i].phi_prime_[j];
            elements[i].phi_[j] += g * delta_t * elements[i].intermediate_[j];
        }
    }
}

__global__
void SEM::calculate_fluxes(size_t N_faces, Face_t* faces, const Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_faces; i += stride) {
        deviceFloat u;
        const deviceFloat u_left = elements[faces[i].elements_[0]].phi_R_;
        const deviceFloat u_right = elements[faces[i].elements_[1]].phi_L_;

        if (u_left < 0.0f && u_right > 0.0f) { // In expansion fan
            u = 0.5f * (u_left + u_right);
        }
        else if (u_left >= u_right) { // Shock
            if (u_left > 0.0f) {
                u = u_left;
            }
            else {
                u = u_right;
            }
        }
        else { // Expansion fan
            if (u_left > 0.0f) {
                u = u_left;
            }
            else  {
                u = u_right;
            }
        }
    
        faces[i].flux_ = u_right;
        faces[i].nl_flux_ = 0.5f * u * u;
    }
}

__global__
void SEM::calculate_q_fluxes(size_t N_faces, Face_t* faces, const Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_faces; i += stride) {
        const deviceFloat u_prime_left = elements[faces[i].elements_[0]].phi_prime_R_;

        faces[i].derivative_flux_ = u_prime_left;
    }
}

__device__
void SEM::matrix_vector_multiply(int N, const deviceFloat* matrix, const deviceFloat* vector, deviceFloat* result) {
    for (int i = 0; i <= N; ++i) {
        result[i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            result[i] +=  matrix[i * (N + 1) + j] * vector[j];
        }
    }
}

// Algorithm 19
__device__
void SEM::matrix_vector_derivative(int N, const deviceFloat* derivative_matrices_hat, const deviceFloat* phi, deviceFloat* phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    for (int i = 0; i <= N; ++i) {
        phi_prime[i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            phi_prime[i] += derivative_matrices_hat[i * (N + 1) + j] * phi[j] * phi[j]/2;
        }
    }
}

// Algorithm 60 (not really anymore)
__global__
void SEM::compute_dg_derivative(size_t N_elements, Element_t* elements, const Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t offset_1D = elements[i].N_ * (elements[i].N_ + 1) /2; // CHECK cache?
        const size_t offset_2D = elements[i].N_ * (elements[i].N_ + 1) * (2 * elements[i].N_ + 1) /6;

        const deviceFloat flux_L = faces[elements[i].faces_[0]].flux_;
        const deviceFloat flux_R = faces[elements[i].faces_[1]].flux_;

        SEM::matrix_vector_multiply(elements[i].N_, derivative_matrices_hat + offset_2D, elements[i].phi_, elements[i].q_);
        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].q_[j] = -elements[i].q_[j] - (flux_R * lagrange_interpolant_right[offset_1D + j]
                                                     - flux_L * lagrange_interpolant_left[offset_1D + j]) / weights[offset_1D + j];
            elements[i].q_[j] *= 2.0f/elements[i].delta_x_;
        }
    }
}

// Algorithm 60 (not really anymore)
__global__
void SEM::compute_dg_derivative2(deviceFloat viscosity, size_t N_elements, Element_t* elements, const Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t offset_1D = elements[i].N_ * (elements[i].N_ + 1) /2; // CHECK cache?
        const size_t offset_2D = elements[i].N_ * (elements[i].N_ + 1) * (2 * elements[i].N_ + 1) /6;

        const deviceFloat derivative_flux_L = faces[elements[i].faces_[0]].derivative_flux_;
        const deviceFloat derivative_flux_R = faces[elements[i].faces_[1]].derivative_flux_;
        const deviceFloat nl_flux_L = faces[elements[i].faces_[0]].nl_flux_;
        const deviceFloat nl_flux_R = faces[elements[i].faces_[1]].nl_flux_;
        
        SEM::matrix_vector_derivative(elements[i].N_, derivative_matrices_hat + offset_2D, elements[i].phi_, elements[i].ux_);
        SEM::matrix_vector_multiply(elements[i].N_, derivative_matrices_hat + offset_2D, elements[i].q_, elements[i].phi_prime_);
        
        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_prime_[j] = -elements[i].phi_prime_[j] * viscosity
                                        - (derivative_flux_R * lagrange_interpolant_right[offset_1D + j]
                                           - derivative_flux_L * lagrange_interpolant_left[offset_1D + j]) * viscosity /weights[offset_1D + j]
                                        - elements[i].ux_[j]
                                        + (nl_flux_L * lagrange_interpolant_left[offset_1D + j] 
                                            - nl_flux_R * lagrange_interpolant_right[offset_1D + j]) / weights[offset_1D + j];

            elements[i].phi_prime_[j] *= 2.0f/elements[i].delta_x_;
        }
    }
}
