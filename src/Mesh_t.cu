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

Mesh_t::Mesh_t(size_t N_elements, int initial_N, deviceFloat x_min, deviceFloat x_max) : 
        N_elements_(N_elements), 
        N_faces_(N_elements), 
        initial_N_(initial_N),
        elements_numBlocks_((N_elements_ + elements_blockSize - 1) / elements_blockSize),
        faces_numBlocks_((N_faces_ + faces_blockSize - 1) / faces_blockSize),
        host_delta_t_array_(new deviceFloat[elements_numBlocks_]) {

    // CHECK N_faces = N_elements only for periodic BC.
    cudaMalloc(&elements_, N_elements_ * sizeof(Element_t));
    cudaMalloc(&faces_, N_faces_ * sizeof(Face_t));
    cudaMalloc(&device_delta_t_array_, elements_numBlocks_ * sizeof(deviceFloat));

    SEM::build_elements<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, initial_N_, elements_, x_min, x_max);
    SEM::build_faces<<<faces_numBlocks_, faces_blockSize>>>(N_faces_, faces_); // CHECK
}

Mesh_t::~Mesh_t() {
    cudaFree(elements_);
    cudaFree(faces_);
    cudaFree(device_delta_t_array_);
    
    delete[] host_delta_t_array_;
}

void Mesh_t::set_initial_conditions(const deviceFloat* nodes) {
    SEM::initial_conditions<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, nodes);
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

    SEM::get_elements_data<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, phi, phi_prime);
    
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

void Mesh_t::write_file_data(size_t N_points, deviceFloat time, const deviceFloat* coordinates, const deviceFloat* velocity, const deviceFloat* du_dx, const deviceFloat* intermediate, const deviceFloat* sigma, const deviceFloat* refine, const deviceFloat* coarsen, const deviceFloat* error) {
    std::stringstream ss;
    std::ofstream file;

    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    ss << "output_t" << std::setprecision(4) << std::fixed << time << ".dat";
    file.open(save_dir / ss.str());

    file << "TITLE = \"Velocity  at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\", \"U_x_prime\", \"intermediate\", \"sigma\", \"refine\", \"coarsen\"" << std::endl;
    file << "ZONE T= \"Zone     1\",  I= " << N_points << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (size_t i = 0; i < N_points; ++i) {
        file << std::setw(12) << coordinates[i] 
            << " " << std::setw(12) << (std::isnan(velocity[i]) ? velocity[i] : velocity[i]) 
            << " " << std::setw(12) << (std::isnan(du_dx[i]) ? du_dx[i] : du_dx[i])
            << " " << std::setw(12) << (std::isnan(intermediate[i]) ? intermediate[i] : intermediate[i])
            << " " << std::setw(12) << (std::isnan(sigma[i]) ? sigma[i] : sigma[i])
            << " " << std::setw(12) << (std::isnan(refine[i]) ? refine[i] : refine[i])
            << " " << std::setw(12) << (std::isnan(coarsen[i]) ? coarsen[i] : coarsen[i])
            << " " << std::setw(12) << (std::isnan(error[i]) ? error[i] : error[i]) << std::endl;
    }

    file.close();
}

void Mesh_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices) {
    // CHECK find better solution for multiple elements
    deviceFloat* x;
    deviceFloat* phi;
    deviceFloat* phi_prime;
    deviceFloat* intermediate;
    deviceFloat* sigma;
    deviceFloat* refine;
    deviceFloat* coarsen;
    deviceFloat* error;
    deviceFloat* host_x = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_phi = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_phi_prime = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_intermediate = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_sigma = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_refine = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_coarsen = new deviceFloat[N_elements_ * N_interpolation_points];
    deviceFloat* host_error = new deviceFloat[N_elements_ * N_interpolation_points];
    cudaMalloc(&x, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi_prime, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&intermediate, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&sigma, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&refine, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&coarsen, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&error, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    

    SEM::get_solution<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, x, phi, phi_prime, intermediate, sigma, refine, coarsen, error);
    
    cudaMemcpy(host_x, x , N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi, phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime, phi_prime, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_intermediate, intermediate, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sigma, sigma, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_refine, refine, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_coarsen, coarsen, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_error, error, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    
    write_file_data(N_elements_ * N_interpolation_points, time, host_x, host_phi, host_phi_prime, host_intermediate, host_sigma, host_refine, host_coarsen, host_error);

    delete[] host_x;
    delete[] host_phi;
    delete[] host_phi_prime;
    delete[] host_intermediate;
    delete[] host_sigma;
    delete[] host_refine;
    delete[] host_coarsen;
    delete[] host_error;
    cudaFree(x);
    cudaFree(phi);
    cudaFree(phi_prime);
    cudaFree(intermediate);
    cudaFree(sigma);
    cudaFree(refine);
    cudaFree(coarsen);
    cudaFree(error);
}

template void Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<ChebyshevPolynomial_t> &NDG); // Get with the times c++, it's crazy I have to do this
template void Mesh_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<LegendrePolynomial_t> &NDG);

template<typename Polynomial>
void Mesh_t::solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG) {
    deviceFloat time = 0.0;
    const deviceFloat t_end = output_times.back();

    deviceFloat delta_t = get_delta_t(CFL);
    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

    while (time < t_end) {
        // Kinda algorithm 62
        deviceFloat t = time;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_first_step<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, delta_t, 1.0f/3.0f);

        t = time + 0.33333333333f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, delta_t, -5.0f/9.0f, 15.0f/16.0f);

        t = time + 0.75f * delta_t;
        SEM::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::calculate_fluxes<<<faces_numBlocks_, faces_blockSize>>>(N_faces_, faces_, elements_);
        SEM::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::rk3_step<<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, delta_t, -153.0f/128.0f, 8.0f/15.0f);
              
        time += delta_t;
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                SEM::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, NDG.nodes_, NDG.weights_);
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
                adapt();
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
        SEM::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize>>>(N_elements_, elements_, NDG.nodes_, NDG.weights_);
        write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
    }
}

deviceFloat Mesh_t::get_delta_t(const deviceFloat CFL) {   
    SEM::reduce_delta_t<elements_blockSize><<<elements_numBlocks_, elements_blockSize>>>(CFL, N_elements_, elements_, device_delta_t_array_);
    cudaMemcpy(host_delta_t_array_, device_delta_t_array_, elements_numBlocks_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    deviceFloat delta_t_min = std::numeric_limits<deviceFloat>::infinity();
    for (int i = 0; i < elements_numBlocks_; ++i) {
        delta_t_min = min(delta_t_min, host_delta_t_array_[i]);
    }
    
    return delta_t_min;
}

void Mesh_t::adapt() {
    // Update elements_numBlocks_ and faces_numBlocks_
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
