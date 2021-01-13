#include "Mesh_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
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
    SEM::build_elements<<<elements_numBlocks, elements_blockSize>>>(N_elements_, initial_N_, elements_, x_min, x_max);
    SEM::build_faces<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_); // CHECK
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
    SEM::initial_conditions<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, nodes);
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
    SEM::get_elements_data<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, phi, phi_prime);
    
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
    SEM::get_solution<<<elements_numBlocks, elements_blockSize>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, phi, x);
    
    cudaDeviceSynchronize();
    cudaMemcpy(host_phi, phi, N_elements_ * N_interpolation_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x, x , N_elements_ * N_interpolation_points * sizeof(float), cudaMemcpyDeviceToHost);

    write_file_data(N_elements_ * N_interpolation_points, time, host_phi, host_x);

    delete[] host_phi;
    delete[] host_x;
    cudaFree(phi);
    cudaFree(x);
}

template void Mesh_t::solve(const float delta_t, const std::vector<float> output_times, const NDG_t<ChebyshevPolynomial_t> &NDG); // Get with the times c++, it's crazy I have to do this
template void Mesh_t::solve(const float delta_t, const std::vector<float> output_times, const NDG_t<LegendrePolynomial_t> &NDG);

template<typename Polynomial>
void Mesh_t::solve(const float delta_t, const std::vector<float> output_times, const NDG_t<Polynomial> &NDG) {
    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
    float time = 0.0;
    const float t_end = output_times.back();

    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

    while (time < t_end) {
        // Kinda algorithm 62
        float t = time;
        SEM::interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, 0.0f, 1.0f/3.0f);

        t = time + 0.33333333333f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, -5.0f/9.0f, 15.0f/16.0f);

        t = time + 0.75f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, -153.0f/128.0f, 8.0f/15.0f);
              
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

__global__
void SEM::rk3_step(int N_elements, Element_t* elements, float delta_t, float a, float g) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j){
            elements[i].intermediate_[j] = a * elements[i].intermediate_[j] + elements[i].phi_prime_[j];
            elements[i].phi_[j] += g * delta_t * elements[i].intermediate_[j];
        }
    }
}

__global__
void SEM::calculate_fluxes(int N_faces, Face_t* faces, const Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_faces; i += stride) {
        float u;
        const float u_left = elements[faces[i].elements_[0]].phi_R_;
        const float u_right = elements[faces[i].elements_[1]].phi_L_;

        if (u_left < 0.0f && u_right > 0.0f) { // In expansion fan
            u = 0.5f * (u_left + u_right);
        }
        else if (u_left > u_right) { // Shock
            if (u_left > 0.0f) {
                u = u_left;
            }
            else if (u_left < u_right) {
                u = u_right;
            }
            else { // ADDED
                u = 0.5f * (u_left + u_right);
            }
        }
        else if (u_left < u_right) { // Expansion fan
            if (u_left > 0.0f) {
                u = u_left;
            }
            else if (u_left < 0.0f) {
                u = u_right;
            }
            else { // ADDED
                u = 0.5f * (u_left + u_right);
            }
        }
        else { // ADDED
            u = 0.5f * (u_left + u_right);
        }

        faces[i].flux_ = 0.5f * u * u;
    }
}

// Algorithm 19
__device__
void SEM::matrix_vector_derivative(int N, const float* derivative_matrices_hat, const float* phi, float* phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = 0; i <= N; ++i) {
        phi_prime[i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            phi_prime[i] += derivative_matrices_hat[offset_2D + i * (N + 1) + j] * phi[j] * phi[j] * 0.5f; // phi not squared in textbook, squared for Burger's
        }
    }
}

// Algorithm 60 (not really anymore)
__global__
void SEM::compute_dg_derivative(int N_elements, Element_t* elements, const Face_t* faces, const float* weights, const float* derivative_matrices_hat, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int offset_1D = elements[i].N_ * (elements[i].N_ + 1) /2; // CHECK cache?

        const float flux_L = faces[elements[i].faces_[0]].flux_;
        const float flux_R = faces[elements[i].faces_[1]].flux_;

        SEM::matrix_vector_derivative(elements[i].N_, derivative_matrices_hat, elements[i].phi_, elements[i].phi_prime_);

        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_prime_[j] += (flux_L * lagrange_interpolant_left[offset_1D + j] - flux_R * lagrange_interpolant_right[offset_1D + j]) / weights[offset_1D + j];
            elements[i].phi_prime_[j] *= 2.0f/elements[i].delta_x_;
        }
    }
}