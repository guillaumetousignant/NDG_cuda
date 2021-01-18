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

Mesh_t::Mesh_t(size_t N_elements, int initial_N, deviceFloat x_min, deviceFloat x_max) : N_elements_(N_elements), N_faces_(N_elements), initial_N_(initial_N) {
    // CHECK N_faces = N_elements only for periodic BC.
    cudaMalloc(&elements_, N_elements_ * sizeof(Element_t));
    cudaMalloc(&faces_, N_faces_ * sizeof(Face_t));

    cudaMalloc(&phi_arrays_, N_elements_ * sizeof(deviceFloat*));
    cudaMalloc(&phi_prime_arrays_, N_elements_ * sizeof(deviceFloat*));
    cudaMalloc(&intermediate_arrays_, N_elements_ * sizeof(deviceFloat*));
    phi_arrays_ = new deviceFloat*[N_elements_];
    phi_prime_arrays_ = new deviceFloat*[N_elements_];
    intermediate_arrays_ = new deviceFloat*[N_elements_];

    for (size_t i = 0; i < N_elements_; ++i) {
        cudaMalloc(&(phi_arrays_[i]), initial_N_ * sizeof(deviceFloat));
        cudaMalloc(&(phi_prime_arrays_[i]), initial_N_ * sizeof(deviceFloat));
        cudaMalloc(&(intermediate_arrays_[i]), initial_N_ * sizeof(deviceFloat));
    }

    deviceFloat** phi_arrays_device;
    deviceFloat** phi_prime_arrays_device;
    deviceFloat** intermediate_arrays_device;
    cudaMalloc(&phi_arrays_device, N_elements_ * sizeof(deviceFloat*));
    cudaMalloc(&phi_prime_arrays_device, N_elements_ * sizeof(deviceFloat*));
    cudaMalloc(&intermediate_arrays_device, N_elements_ * sizeof(deviceFloat*));
    cudaMemcpy(phi_arrays_device ,phi_arrays_, N_elements_ * sizeof(deviceFloat*), cudaMemcpyHostToDevice);
    cudaMemcpy(phi_prime_arrays_device ,phi_prime_arrays_, N_elements_ * sizeof(deviceFloat*), cudaMemcpyHostToDevice);
    cudaMemcpy(intermediate_arrays_device ,intermediate_arrays_, N_elements_ * sizeof(deviceFloat*), cudaMemcpyHostToDevice);



    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
    SEM::build_elements<<<elements_numBlocks, elements_blockSize>>>(N_elements_, initial_N_, elements_, x_min, x_max, phi_arrays_device, phi_prime_arrays_device, intermediate_arrays_device);
    SEM::build_faces<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_); // CHECK

    cudaFree(phi_arrays_device);
    cudaFree(phi_prime_arrays_device);
    cudaFree(intermediate_arrays_device);
}

Mesh_t::~Mesh_t() {
    if (elements_ != nullptr){
        cudaFree(elements_);
    }

    if (faces_ != nullptr){
        cudaFree(faces_);
    }

    if (phi_arrays_ != nullptr) {
        for (size_t i = 0; i < N_elements_; ++i) {
            if (phi_arrays_[i] != nullptr) {
                cudaFree(phi_arrays_[i]);
            }
        }
        delete [] phi_arrays_;
    }

    if (phi_prime_arrays_ != nullptr) {
        for (size_t i = 0; i < N_elements_; ++i) {
            if (phi_prime_arrays_[i] != nullptr) {
                cudaFree(phi_prime_arrays_[i]);
            }
        }
        delete [] phi_prime_arrays_;
    }

    if (intermediate_arrays_ != nullptr) {
        for (size_t i = 0; i < N_elements_; ++i) {
            if (intermediate_arrays_[i] != nullptr) {
                cudaFree(intermediate_arrays_[i]);
            }
        }
        delete [] intermediate_arrays_;
    }
}

void Mesh_t::set_initial_conditions(const deviceFloat* nodes) {
    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    SEM::initial_conditions<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, nodes);
}

void Mesh_t::print() {
    // CHECK find better solution for multiple elements. This only works if all elements have the same N.
    deviceFloat* phi;
    deviceFloat* phi_prime;
    deviceFloat* host_phi = new deviceFloat[(initial_N_ + 1) * N_elements_];
    deviceFloat* host_phi_prime = new deviceFloat[(initial_N_ + 1) * N_elements_];
    Face_t* host_faces = new Face_t[N_faces_];
    Element_t* host_elements = new Element_t[N_elements_];
    cudaMalloc(&phi, (initial_N_ + 1) * N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&phi_prime, (initial_N_ + 1) * N_elements_ * sizeof(deviceFloat));

    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    SEM::get_elements_data<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, phi, phi_prime);
    
    cudaMemcpy(host_phi, phi, (initial_N_ + 1) * N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime, phi_prime, (initial_N_ + 1) * N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_faces, faces_, N_faces_ * sizeof(Face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_elements, elements_, N_elements_ * sizeof(Element_t), cudaMemcpyDeviceToHost);

    // Invalidate GPU pointers, or else they will be deleted on the CPU, where they point to random stuff
    for (size_t i = 0; i < N_elements_; ++i) {
        host_elements[i].phi_ = nullptr;
        host_elements[i].phi_prime_ = nullptr;
        host_elements[i].intermediate_ = nullptr;
    }

    std::cout << std::endl << "Phi: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
        const size_t element_offset = i * (initial_N_ + 1);
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N_; ++j) {
            std::cout << host_phi[element_offset + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
        const size_t element_offset = i * (initial_N_ + 1);
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N_; ++j) {
            std::cout << host_phi_prime[element_offset + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi interpolated: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].phi_L_ << " ";
        std::cout << host_elements[i].phi_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "x: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].x_[0] << " ";
        std::cout << host_elements[i].x_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "Neighbouring elements: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].neighbours_[0] << " ";
        std::cout << host_elements[i].neighbours_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "Neighbouring faces: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].faces_[0] << " ";
        std::cout << host_elements[i].faces_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "N: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].N_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "delta x: " << std::endl;
    for (size_t i = 0; i < N_elements_; ++i) {
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

    std::cout << std::endl << "Elements: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
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

void Mesh_t::write_file_data(size_t N_points, deviceFloat time, const deviceFloat* velocity, const deviceFloat* coordinates) {
    std::stringstream ss;
    std::ofstream file;

    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    ss << "output_t" << std::setprecision(4) << std::fixed << time << ".dat";
    file.open(save_dir / ss.str());

    file << "TITLE = \"Velocity  at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\"" << std::endl;
    file << "ZONE T= \"Zone     1\",  I= " << N_points << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (size_t i = 0; i < N_points; ++i) {
        file << std::setw(12) << coordinates[i] << " " << std::setw(12) << (std::isnan(velocity[i]) ? 0.0 : velocity[i]) << std::endl;
    }

    file.close();
}

void Mesh_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices) {
    // CHECK find better solution for multiple elements
    deviceFloat* phi;
    deviceFloat* x;
    deviceFloat* host_phi = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_x = new deviceFloat[N_elements_ * N_interpolation_points];
    cudaMalloc(&phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&x, N_elements_ * N_interpolation_points * sizeof(deviceFloat));

    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    SEM::get_solution<<<elements_numBlocks, elements_blockSize>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, phi, x);
    
    cudaMemcpy(host_phi, phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x, x , N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    write_file_data(N_elements_ * N_interpolation_points, time, host_phi, host_x);

    delete[] host_phi;
    delete[] host_x;
    cudaFree(phi);
    cudaFree(x);
}

template void Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<ChebyshevPolynomial_t> &NDG); // Get with the times c++, it's crazy I have to do this
template void Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<LegendrePolynomial_t> &NDG);

template<typename Polynomial>
void Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG) {
    const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
    const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
    deviceFloat time = 0.0;
    const deviceFloat t_end = output_times.back();

    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

    while (time < t_end) {
        // Kinda algorithm 62
        deviceFloat t = time;
        SEM::interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_first_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, 1.0f/3.0f);

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
void SEM::matrix_vector_derivative(int N, const deviceFloat* derivative_matrices_hat, const deviceFloat* phi, deviceFloat* phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = 0; i <= N; ++i) {
        phi_prime[i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            phi_prime[i] += derivative_matrices_hat[offset_2D + i * (N + 1) + j] * phi[j] * phi[j] * 0.5f; // phi not squared in textbook, squared for Burger's
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

        const deviceFloat flux_L = faces[elements[i].faces_[0]].flux_;
        const deviceFloat flux_R = faces[elements[i].faces_[1]].flux_;

        SEM::matrix_vector_derivative(elements[i].N_, derivative_matrices_hat, elements[i].phi_, elements[i].phi_prime_);

        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_prime_[j] += (flux_L * lagrange_interpolant_left[offset_1D + j] - flux_R * lagrange_interpolant_right[offset_1D + j]) / weights[offset_1D + j];
            elements[i].phi_prime_[j] *= 2.0f/elements[i].delta_x_;
        }
    }
}