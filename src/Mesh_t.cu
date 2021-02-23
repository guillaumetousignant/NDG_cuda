#include "Mesh_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

constexpr int elements_blockSize = 32;
constexpr int faces_blockSize = 32; // Same number of faces as elements for periodic BC
constexpr int boundaries_blockSize = 32;

SEM::Mesh_t::Mesh_t(size_t N_elements, int initial_N, deviceFloat x_min, deviceFloat x_max, cudaStream_t &stream) : 
        N_elements_global_(N_elements),        
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
    elements_numBlocks_ = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    faces_numBlocks_ = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
    boundaries_numBlocks_ = (N_local_boundaries_ + N_MPI_boundaries_ + boundaries_blockSize - 1) / boundaries_blockSize;

    host_delta_t_array_ = new deviceFloat[elements_numBlocks_];
    host_refine_array_ = new unsigned long[elements_numBlocks_];
    host_boundary_phi_L_ = new deviceFloat[N_MPI_boundaries_];
    host_boundary_phi_R_ = new deviceFloat[N_MPI_boundaries_];
    host_boundary_phi_prime_L_ = new deviceFloat[N_MPI_boundaries_];
    host_boundary_phi_prime_R_ = new deviceFloat[N_MPI_boundaries_];
    host_MPI_boundary_to_element_ = new size_t[N_MPI_boundaries_];
    host_MPI_boundary_from_element_ = new size_t[N_MPI_boundaries_];
    send_buffers_ = new std::array<double, 4>[N_MPI_boundaries_];
    receive_buffers_ = new std::array<double, 4>[N_MPI_boundaries_];
    requests_ = new MPI_Request[N_MPI_boundaries_*2];
    statuses_ = new MPI_Status[N_MPI_boundaries_*2];

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

    SEM::build_elements<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, initial_N_, elements_, x_min_local, x_max_local);
    SEM::build_boundaries<<<boundaries_numBlocks_, boundaries_blockSize, 0, stream_>>>(N_elements_, N_elements_global_, N_local_boundaries_, N_MPI_boundaries_, initial_N_, elements_, x_min_local, x_max_local, global_element_offset_, local_boundary_to_element_, MPI_boundary_to_element_, MPI_boundary_from_element_);
    SEM::build_faces<<<faces_numBlocks_, faces_blockSize, 0, stream_>>>(N_faces_, faces_);

    cudaMemcpy(host_MPI_boundary_to_element_, MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_from_element_, MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
}

SEM::Mesh_t::~Mesh_t() {
    SEM::free_elements<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_, elements_);
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
    
    delete[] host_delta_t_array_;
    delete[] host_refine_array_;
    delete[] host_boundary_phi_L_;
    delete[] host_boundary_phi_R_;
    delete[] host_boundary_phi_prime_L_;
    delete[] host_boundary_phi_prime_R_;
    delete[] host_MPI_boundary_to_element_;
    delete[] host_MPI_boundary_from_element_;
    delete[] send_buffers_;
    delete[] receive_buffers_;
    delete[] requests_;
    delete[] statuses_;
}

void SEM::Mesh_t::set_initial_conditions(const deviceFloat* nodes) {
    SEM::initial_conditions<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, nodes);
}

void SEM::Mesh_t::print() {
    Face_t* host_faces = new Face_t[N_faces_];
    Element_t* host_elements = new Element_t[N_elements_ + N_local_boundaries_ + N_MPI_boundaries_];
    size_t* host_local_boundary_to_element = new size_t[N_local_boundaries_];

    cudaMemcpy(host_faces, faces_, N_faces_ * sizeof(Face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_elements, elements_, (N_elements_ + N_local_boundaries_ + N_MPI_boundaries_) * sizeof(Element_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_local_boundary_to_element, local_boundary_to_element_, N_local_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_to_element_, MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_from_element_, MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);

    // Invalidate GPU pointers, or else they will be deleted on the CPU, where they point to random stuff
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        host_elements[i].phi_ = nullptr;
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

    delete[] host_faces;
    delete[] host_elements;
    delete[] host_local_boundary_to_element;
}

void SEM::Mesh_t::write_file_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const deviceFloat* coordinates, const deviceFloat* velocity, const deviceFloat* du_dx, const deviceFloat* intermediate, const deviceFloat* x_L, const deviceFloat* x_R, const int* N, const deviceFloat* sigma, const bool* refine, const bool* coarsen, const deviceFloat* error) {
    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    std::stringstream ss;
    std::ofstream file;
    ss << "output_t" << std::setprecision(4) << std::fixed << time << "_proc" << std::setfill('0') << std::setw(6) << rank << ".dat";
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
    ss_element << "output_element_t" << std::setprecision(4) << std::fixed << time << "_proc" << std::setfill('0') << std::setw(6) << rank << ".dat";
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

void SEM::Mesh_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices) {
    // CHECK find better solution for multiple elements
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
    deviceFloat* host_x = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_phi = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_phi_prime = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_intermediate = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_x_L = new deviceFloat[N_elements_];
    deviceFloat* host_x_R = new deviceFloat[N_elements_];
    int* host_N = new int[N_elements_];
    deviceFloat* host_sigma = new deviceFloat[N_elements_];
    bool* host_refine = new bool[N_elements_];
    bool* host_coarsen = new bool[N_elements_];
    deviceFloat* host_error = new deviceFloat[N_elements_ ];
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

    SEM::get_solution<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, x, phi, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);
    
    cudaMemcpy(host_x, x , N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi, phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime, phi_prime, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_intermediate, intermediate, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_L, x_L, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_R, x_R, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_N, N, N_elements_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sigma, sigma, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_refine, refine, N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_coarsen, coarsen, N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_error, error, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    write_file_data(N_interpolation_points, N_elements_, time, global_rank, host_x, host_phi, host_phi_prime, host_intermediate, host_x_L, host_x_R, host_N, host_sigma, host_refine, host_coarsen, host_error);

    delete[] host_x;
    delete[] host_phi;
    delete[] host_phi_prime;
    delete[] host_intermediate;
    delete[] host_x_L;
    delete[] host_x_R;
    delete[] host_N;
    delete[] host_sigma;
    delete[] host_refine;
    delete[] host_coarsen;
    delete[] host_error;
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

template void SEM::Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<ChebyshevPolynomial_t> &NDG, deviceFloat viscosity); // Get with the times c++, it's crazy I have to do this
template void SEM::Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<LegendrePolynomial_t> &NDG, deviceFloat viscosity);

template<typename Polynomial>
void SEM::Mesh_t::solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG, deviceFloat viscosity) {
    deviceFloat time = 0.0;
    const deviceFloat t_end = output_times.back();

    deviceFloat delta_t = get_delta_t(CFL);
    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

    while (time < t_end) {
        // Kinda algorithm 62
        deviceFloat t = time;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_, NDG.lagrange_interpolant_derivative_left_, NDG.lagrange_interpolant_derivative_right_);
        boundary_conditions();
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(viscosity, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.g_hat_derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_first_step<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, delta_t, 1.0f/3.0f);

        t = time + 0.33333333333f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_, NDG.lagrange_interpolant_derivative_left_, NDG.lagrange_interpolant_derivative_right_);
        boundary_conditions();
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(viscosity, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.g_hat_derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, delta_t, -5.0f/9.0f, 15.0f/16.0f);

        t = time + 0.75f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_, NDG.lagrange_interpolant_derivative_left_, NDG.lagrange_interpolant_derivative_right_);
        boundary_conditions();
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize, 0, stream_>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(viscosity, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.g_hat_derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, delta_t, -153.0f/128.0f, 8.0f/15.0f);
              
        time += delta_t;
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                SEM::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, NDG.nodes_, NDG.weights_);
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
                adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
                delta_t = get_delta_t(CFL);
                break;
            }
        }
    }

    bool did_write = false;
    for (auto const& e : std::as_const(output_times)) {
        if ((time >= e) && (time < e + delta_t)) {
            did_write = true;
            break;
        }
    }

    if (!did_write) {
        SEM::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, NDG.nodes_, NDG.weights_);
        write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
    }
}

deviceFloat SEM::Mesh_t::get_delta_t(const deviceFloat CFL) {   
    SEM::reduce_delta_t<elements_blockSize/2><<<elements_numBlocks_, elements_blockSize/2, 0, stream_>>>(CFL, N_elements_, elements_, device_delta_t_array_);
    cudaMemcpy(host_delta_t_array_, device_delta_t_array_, elements_numBlocks_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

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
    SEM::reduce_refine<elements_blockSize/2><<<elements_numBlocks_, elements_blockSize/2, 0, stream_>>>(N_elements_, elements_, device_refine_array_);
    cudaMemcpy(host_refine_array_, device_refine_array_, elements_numBlocks_ * sizeof(unsigned long), cudaMemcpyDeviceToHost);

    unsigned long additional_elements = 0;
    for (int i = 0; i < elements_numBlocks_; ++i) {
        additional_elements += host_refine_array_[i];
        host_refine_array_[i] = additional_elements - host_refine_array_[i]; // Current block offset
    }

    if (additional_elements == 0) {
        SEM::p_adapt<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, N_max, nodes, barycentric_weights);
        return;
    }

    cudaMemcpy(device_refine_array_, host_refine_array_, elements_numBlocks_ * sizeof(unsigned long), cudaMemcpyHostToDevice);

    Element_t* new_elements;
    Face_t* new_faces;

    // CHECK N_faces = N_elements only for periodic BC.
    cudaMalloc(&new_elements, (N_elements_ + additional_elements) * sizeof(Element_t));
    cudaMalloc(&new_faces, (N_faces_ + additional_elements) * sizeof(Face_t));

    SEM::copy_faces<<<faces_numBlocks_, faces_blockSize, 0, stream_>>>(N_faces_, faces_, new_faces);
    SEM::adapt<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_, elements_, new_elements, new_faces, device_refine_array_, N_max, nodes, barycentric_weights);

    SEM::free_elements<<<elements_numBlocks_, elements_blockSize, 0, stream_>>>(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_, elements_);
    cudaFree(elements_);
    cudaFree(faces_);
    elements_ = new_elements;
    faces_ = new_faces;
    
    N_elements_ += additional_elements;
    N_faces_ += additional_elements;
    elements_numBlocks_ = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    faces_numBlocks_ = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
    boundaries_numBlocks_ = (N_local_boundaries_ + N_MPI_boundaries_ + boundaries_blockSize - 1) / boundaries_blockSize;
    N_elements_per_process_ = N_elements_per_process_; // CHECK change

    delete[] host_delta_t_array_;
    delete[] host_refine_array_;
    delete[] host_boundary_phi_L_;
    delete[] host_boundary_phi_R_;
    delete[] host_boundary_phi_prime_L_;
    delete[] host_boundary_phi_prime_R_;
    delete[] host_MPI_boundary_to_element_;
    delete[] host_MPI_boundary_from_element_;
    delete[] send_buffers_;
    delete[] receive_buffers_;
    delete[] requests_;
    delete[] statuses_;
    host_delta_t_array_ = new deviceFloat[elements_numBlocks_];
    host_refine_array_ = new unsigned long[elements_numBlocks_];
    host_boundary_phi_L_ = new deviceFloat[N_MPI_boundaries_];
    host_boundary_phi_R_ = new deviceFloat[N_MPI_boundaries_];
    host_boundary_phi_prime_L_ = new deviceFloat[N_MPI_boundaries_];
    host_boundary_phi_prime_R_ = new deviceFloat[N_MPI_boundaries_];
    host_MPI_boundary_to_element_ = new size_t[N_MPI_boundaries_];
    host_MPI_boundary_from_element_ = new size_t[N_MPI_boundaries_];
    send_buffers_ = new std::array<double, 4>[N_MPI_boundaries_];
    receive_buffers_ = new std::array<double, 4>[N_MPI_boundaries_];
    requests_ = new MPI_Request[N_MPI_boundaries_*2];
    statuses_ = new MPI_Status[N_MPI_boundaries_*2];

    cudaFree(local_boundary_to_element_);
    cudaFree(MPI_boundary_to_element_);
    cudaFree(MPI_boundary_from_element_);
    cudaFree(device_delta_t_array_);
    cudaFree(device_refine_array_);
    cudaFree(device_boundary_phi_L_);
    cudaFree(device_boundary_phi_R_);
    cudaFree(device_boundary_phi_prime_L_);
    cudaFree(device_boundary_phi_prime_R_);
    cudaMalloc(&local_boundary_to_element_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t));
    cudaMalloc(&MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t));
    cudaMalloc(&device_delta_t_array_, elements_numBlocks_ * sizeof(deviceFloat));
    cudaMalloc(&device_refine_array_, elements_numBlocks_ * sizeof(unsigned long));
    cudaMalloc(&device_boundary_phi_L_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_R_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_prime_L_, N_MPI_boundaries_ * sizeof(deviceFloat));
    cudaMalloc(&device_boundary_phi_prime_R_, N_MPI_boundaries_ * sizeof(deviceFloat));

    // CHECK create boundaries here.

    cudaMemcpy(host_MPI_boundary_to_element_, MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_from_element_, MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
}

void SEM::Mesh_t::boundary_conditions() {
    SEM::local_boundaries<<<boundaries_numBlocks_, boundaries_blockSize, 0, stream_>>>(N_elements_, N_local_boundaries_, elements_, local_boundary_to_element_);
    SEM::get_MPI_boundaries<<<boundaries_numBlocks_, boundaries_blockSize, 0, stream_>>>(N_elements_, N_local_boundaries_, N_MPI_boundaries_, elements_, faces_, device_boundary_phi_L_, device_boundary_phi_R_, device_boundary_phi_prime_L_, device_boundary_phi_prime_R_);
    
    cudaMemcpy(host_boundary_phi_L_, device_boundary_phi_L_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_boundary_phi_R_, device_boundary_phi_R_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_boundary_phi_prime_L_, device_boundary_phi_prime_L_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_boundary_phi_prime_R_, device_boundary_phi_prime_R_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        send_buffers_[i] = {host_boundary_phi_L_[i],
                            host_boundary_phi_R_[i],
                            host_boundary_phi_prime_L_[i],
                            host_boundary_phi_prime_R_[i]};
        const int destination = host_MPI_boundary_to_element_[i]/N_elements_per_process_;

        MPI_Irecv(&receive_buffers_[i][0], 4, MPI_DOUBLE, destination, host_MPI_boundary_from_element_[i], MPI_COMM_WORLD, &requests_[i]);
        MPI_Isend(&send_buffers_[i][0], 4, MPI_DOUBLE, destination, host_MPI_boundary_to_element_[i], MPI_COMM_WORLD, &requests_[i + N_MPI_boundaries_]);
    }

    MPI_Waitall(N_MPI_boundaries_, requests_, statuses_);

    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        host_boundary_phi_L_[i] = receive_buffers_[i][0];
        host_boundary_phi_R_[i] = receive_buffers_[i][1];
        host_boundary_phi_prime_L_[i] = receive_buffers_[i][2];
        host_boundary_phi_prime_R_[i] = receive_buffers_[i][3];
    }

    cudaMemcpy(device_boundary_phi_L_, host_boundary_phi_L_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(device_boundary_phi_R_, host_boundary_phi_R_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(device_boundary_phi_prime_L_, host_boundary_phi_prime_L_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);
    cudaMemcpy(device_boundary_phi_prime_R_, host_boundary_phi_prime_R_, N_MPI_boundaries_ * sizeof(deviceFloat), cudaMemcpyHostToDevice);

    SEM::put_MPI_boundaries<<<boundaries_numBlocks_, boundaries_blockSize, 0, stream_>>>(N_elements_, N_local_boundaries_, N_MPI_boundaries_, elements_, device_boundary_phi_L_, device_boundary_phi_R_, device_boundary_phi_prime_L_, device_boundary_phi_prime_R_);
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
        deviceFloat u_prime;
        const deviceFloat u_left = elements[faces[i].elements_[0]].phi_R_;
        const deviceFloat u_right = elements[faces[i].elements_[1]].phi_L_;
        const deviceFloat u_prime_left = elements[faces[i].elements_[0]].phi_prime_R_;
        const deviceFloat u_prime_right = elements[faces[i].elements_[1]].phi_prime_L_;

        if (u_left < 0.0f && u_right > 0.0f) { // In expansion fan
            u = 0.5f * (u_left + u_right);
            u_prime = 0.5f * (u_prime_left + u_prime_right);
        }
        else if (u_left >= u_right) { // Shock
            if (u_left > 0.0f) {
                u = u_left;
                u_prime = u_prime_left;
            }
            else {
                u = u_right;
                u_prime = u_prime_right;
            }
        }
        else { // Expansion fan
            if (u_left > 0.0f) {
                u = u_left;
                u_prime = u_prime_left;
            }
            else  {
                u = u_right;
                u_prime = u_prime_right;
            }
        }

        faces[i].flux_ = 0.5f * u * u;
        faces[i].derivative_flux_ = u_prime;
    }
}

// Algorithm 19
__device__
void SEM::matrix_vector_derivative(deviceFloat viscosity, int N, const deviceFloat* derivative_matrices_hat, const deviceFloat* g_hat_derivative_matrices, const deviceFloat* phi, deviceFloat* phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = 0; i <= N; ++i) {
        phi_prime[i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            phi_prime[i] += derivative_matrices_hat[offset_2D + i * (N + 1) + j] * phi[j] * phi[j] * 0.5f - viscosity * g_hat_derivative_matrices[offset_2D + i * (N + 1) + j] * phi[j]; // phi not squared in textbook, squared for Burger's
        }
    }
}

// Algorithm 60 (not really anymore)
__global__
void SEM::compute_dg_derivative(deviceFloat viscosity, size_t N_elements, Element_t* elements, const Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* g_hat_derivative_matrices, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t offset_1D = elements[i].N_ * (elements[i].N_ + 1) /2; // CHECK cache?

        const deviceFloat flux_L = faces[elements[i].faces_[0]].flux_;
        const deviceFloat flux_R = faces[elements[i].faces_[1]].flux_;
        const deviceFloat derivative_flux_L = faces[elements[i].faces_[0]].derivative_flux_;
        const deviceFloat derivative_flux_R = faces[elements[i].faces_[1]].derivative_flux_;

        SEM::matrix_vector_derivative(viscosity, elements[i].N_, derivative_matrices_hat, g_hat_derivative_matrices, elements[i].phi_, elements[i].phi_prime_);

        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_prime_[j] += (flux_L * lagrange_interpolant_left[offset_1D + j] 
                                        - flux_R * lagrange_interpolant_right[offset_1D + j] 
                                        + viscosity * derivative_flux_R * lagrange_interpolant_right[offset_1D + j]
                                        - viscosity * derivative_flux_L * lagrange_interpolant_left[offset_1D + j]) / weights[offset_1D + j];
            elements[i].phi_prime_[j] *= 2.0f/elements[i].delta_x_;
        }
    }
}
