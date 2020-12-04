#include "Mesh_t.cuh"
#include "Element_t.cuh"
#include "Face_t.cuh"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

constexpr int elements_blockSize = 32; // For when we'll have multiple elements
constexpr int faces_blockSize = 32; // Same number of faces as elements for periodic BC

Mesh_t::Mesh_t(int N_elements, int initial_N, float x_min, float x_max) : N_elements_(N_elements), N_faces_(N_elements), initial_N_(initial_N) {
    // CHECK N_faces = N_elements only for periodic BC.
    cudaMalloc(&elements_, N_elements_ * sizeof(Element_t));
    cudaMalloc(&faces_, N_faces_ * sizeof(Face_t));

    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
    build_elements<<<elements_numBlocks, elements_blockSize>>>(N_elements_, initial_N_, elements_, x_min, x_max);
    build_faces<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_); // CHECK
}

Mesh_t::~Mesh_t() {
    if (elements_ != nullptr){
        cudaFree(elements_);
    }

    if (faces_ != nullptr){
        cudaFree(faces_);
    }
}

void Mesh_t::set_initial_conditions(const float* nodes) {
    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    initial_conditions<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, nodes);
}

void Mesh_t::print() {
    // CHECK find better solution for multiple elements. This only works if all elements have the same N.
    float* phi;
    float* phi_prime;
    float* host_phi = new float[(initial_N_ + 1) * N_elements_];
    float* host_phi_prime = new float[(initial_N_ + 1) * N_elements_];
    Face_t* host_faces = new Face_t[N_faces_];
    Element_t* host_elements = new Element_t[N_elements_];
    cudaMalloc(&phi, (initial_N_ + 1) * N_elements_ * sizeof(float));
    cudaMalloc(&phi_prime, (initial_N_ + 1) * N_elements_ * sizeof(float));

    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    get_elements_data<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, phi, phi_prime);
    
    cudaDeviceSynchronize();
    cudaMemcpy(host_phi, phi, (initial_N_ + 1) * N_elements_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime, phi_prime, (initial_N_ + 1) * N_elements_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_faces, faces_, N_faces_ * sizeof(Face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_elements, elements_, N_elements_ * sizeof(Element_t), cudaMemcpyDeviceToHost);

    // Invalidate GPU pointers, or else they will be deleted on the CPU, where they point to random stuff
    for (int i = 0; i < N_elements_; ++i) {
        host_elements[i].phi_ = nullptr;
        host_elements[i].phi_prime_ = nullptr;
        host_elements[i].intermediate_ = nullptr;
    }

    std::cout << std::endl << "Phi: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        const int element_offset = i * (initial_N_ + 1);
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N_; ++j) {
            std::cout << host_phi[element_offset + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        const int element_offset = i * (initial_N_ + 1);
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N_; ++j) {
            std::cout << host_phi_prime[element_offset + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi interpolated: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].phi_L_ << " ";
        std::cout << host_elements[i].phi_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "x: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].x_[0] << " ";
        std::cout << host_elements[i].x_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "Neighbouring elements: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].neighbours_[0] << " ";
        std::cout << host_elements[i].neighbours_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "Neighbouring faces: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].faces_[0] << " ";
        std::cout << host_elements[i].faces_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "N: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].N_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "delta x: " << std::endl;
    for (int i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].delta_x_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "Fluxes: " << std::endl;
    for (int i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].flux_ << std::endl;
    }

    std::cout << std::endl << "Elements: " << std::endl;
    for (int i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].elements_[0] << " ";
        std::cout << host_faces[i].elements_[1] << std::endl;
    }

    delete[] host_phi;
    delete[] host_phi_prime;
    delete[] host_faces;
    delete[] host_elements;

    cudaFree(phi);
    cudaFree(phi_prime);
}

void Mesh_t::write_file_data(int N_points, float time, const float* velocity, const float* coordinates) {
    std::stringstream ss;
    std::ofstream file;

    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    ss << "output_t" << std::setprecision(4) << std::fixed << time << ".dat";
    file.open(save_dir / ss.str());

    file << "TITLE = \"Velocity  at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\"" << std::endl;
    file << "ZONE T= \"Zone     1\",  I= " << N_points << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (int i = 0; i < N_points; ++i) {
        file << std::setw(12) << coordinates[i] << " " << std::setw(12) << velocity[i] << std::endl;
    }

    file.close();
}

void Mesh_t::write_data(float time, int N_interpolation_points, const float* interpolation_matrices) {
    // CHECK find better solution for multiple elements
    float* phi;
    float* x;
    float* host_phi = new float[N_elements_ * N_interpolation_points];
    float* host_x = new float[N_elements_ * N_interpolation_points];
    cudaMalloc(&phi, N_elements_ * N_interpolation_points * sizeof(float));
    cudaMalloc(&x, N_elements_ * N_interpolation_points * sizeof(float));

    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    get_solution<<<elements_numBlocks, elements_blockSize>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, phi, x);
    
    cudaDeviceSynchronize();
    cudaMemcpy(host_phi, phi, N_elements_ * N_interpolation_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x, x , N_elements_ * N_interpolation_points * sizeof(float), cudaMemcpyDeviceToHost);

    write_file_data(N_elements_ * N_interpolation_points, time, host_phi, host_x);

    delete[] host_phi;
    delete[] host_x;
    cudaFree(phi);
    cudaFree(x);
}

void Mesh_t::solve(const float delta_t, const std::vector<float> output_times, const NDG_t &NDG) {
    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
    float time = 0.0;
    const float t_end = output_times.back();

    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

    while (time < t_end) {
        // Kinda algorithm 62
        float t = time;
        interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
        compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, 0.0f, 1.0f/3.0f);

        t = time + 0.33333333333f * delta_t;
        interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
        compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, -5.0f/9.0f, 15.0f/16.0f);

        t = time + 0.75f * delta_t;
        interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
        compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, -153.0f/128.0f, 8.0f/15.0f);
              
        time += delta_t;
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
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
        write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
    }
}