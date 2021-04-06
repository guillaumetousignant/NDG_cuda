#include "Mesh2D_t.cuh"
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

SEM::Mesh2D_t::Mesh2D_t(size_t N_elements, int initial_N, deviceFloat delta_x_min, deviceFloat x_min, deviceFloat x_max, int adaptivity_interval, cudaStream_t &stream) : 
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

SEM::Mesh2D_t::~Mesh2D_t() {
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

void SEM::Mesh2D_t::set_initial_conditions(const deviceFloat* nodes) {
    SEM::initial_conditions<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, nodes);
}

void SEM::Mesh2D_t::print() {
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

void SEM::Mesh2D_t::write_file_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& coordinates, const std::vector<deviceFloat>& velocity, const std::vector<deviceFloat>& du_dx, const std::vector<deviceFloat>& intermediate, const std::vector<deviceFloat>& x_L, const std::vector<deviceFloat>& x_R, const std::vector<int>& N, const std::vector<deviceFloat>& sigma, const bool* refine, const bool* coarsen, const std::vector<deviceFloat>& error) {
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
                 << "VARIABLES = \"X\", \"X_L\", \"X_R\", \"N\", \"sigma\", \"refine\", \"coarsen\", \"error\"" << std::endl
                 << "ZONE T= \"Zone     1\",  I= " << N_elements << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (size_t j = 0; j < N_elements; ++j) {
        file_element << std::setw(12) << (x_L[j] + x_R[j]) * 0.5
              << " " << std::setw(12) << x_L[j]
              << " " << std::setw(12) << x_R[j]
              << " " << std::setw(12) << N[j]
              << " " << std::setw(12) << sigma[j]
              << " " << std::setw(12) << refine[j]
              << " " << std::setw(12) << coarsen[j]
              << " " << std::setw(12) << error[j] << std::endl;
    }

    file_element.close();
}

void SEM::Mesh2D_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices) {
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

    SEM::get_solution<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, x, phi, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);
    
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

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    write_file_data(N_interpolation_points, N_elements_, time, global_rank, host_x, host_phi, host_phi_prime, host_intermediate, host_x_L, host_x_R, host_N, host_sigma, host_refine, host_coarsen, host_error);

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
}

template void SEM::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<ChebyshevPolynomial_t> &NDG, deviceFloat viscosity); // Get with the times c++, it's crazy I have to do this
template void SEM::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<LegendrePolynomial_t> &NDG, deviceFloat viscosity);

template<typename Polynomial>
void SEM::Mesh2D_t::solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG, deviceFloat viscosity) {
    
}

deviceFloat SEM::Mesh2D_t::get_delta_t(const deviceFloat CFL) {   
    return 0.0;
}

void SEM::Mesh2D_t::adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) {
    
}

void SEM::Mesh2D_t::boundary_conditions() {
    
}
