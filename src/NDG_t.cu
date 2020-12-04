#include "NDG_t.h"
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>

constexpr float pi = 3.14159265358979323846f;
constexpr int poly_blockSize = 16; // Small number of threads per block because N will never be huge
constexpr int interpolation_blockSize = 32;
const dim3 matrix_blockSize(16, 16); // Small number of threads per block because N will never be huge

// Algorithm 26
__global__
void SEM::chebyshev_gauss_nodes_and_weights(int N, float* nodes, float* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        nodes[offset + i] = -std::cos(pi * (2 * i + 1) / (2 * N + 2));
        weights[offset + i] = pi / (N + 1);
    }
}

// Algorithm 22
__device__
void SEM::legendre_polynomial_and_derivative(int N, float x, float &L_N, float &L_N_prime) {
    if (N == 0) {
        L_N = 1.0f;
        L_N_prime = 0.0f;
    }
    else if (N == 1) {
        L_N = x;
        L_N_prime = 1.0f;
    }
    else {
        float L_N_2 = 1.0f;
        float L_N_1 = x;
        float L_N_2_prime = 0.0f;
        float L_N_1_prime = 1.0f;

        for (int k = 2; k <= N; ++k) {
            L_N = (2 * k - 1) * x * L_N_1/k - (k - 1) * L_N_2/k; // L_N_1(x) ??
            L_N_prime = L_N_2_prime + (2 * k - 1) * L_N_1;
            L_N_2 = L_N_1;
            L_N_1 = L_N;
            L_N_2_prime = L_N_1_prime;
            L_N_1_prime = L_N_prime;
        }
    }
}

// Algorithm 23
__global__
void SEM::legendre_gauss_nodes_and_weights(int N, float* nodes, float* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = N * (N + 1) /2;

    if (index == 0) {
        if (N == 0) {
            nodes[offset] = 0.0f;
            weights[offset] = 2.0f;
        }
        else if (N == 1) {
            nodes[offset] = -std::sqrt(1.0f/3.0f);
            weights[offset] = 1.0f;
            nodes[offset + 1] = -nodes[offset];
            weights[offset + 1] = weights[offset];
        }
        else {
            for (int j = 0; j < (N + 1)/2; ++j) {
                nodes[offset + j] = -std::cos(pi * (2 * j + 1)/(2 * N + 2));
                
                for (int k = 0; k < 1000; ++k) {
                    float L_N_plus1, L_N_plus1_prime;
                    legendre_polynomial_and_derivative(N + 1, nodes[offset + j], L_N_plus1, L_N_plus1_prime);
                    float delta = -L_N_plus1/L_N_plus1_prime;
                    nodes[offset + j] += delta;
                    if (std::abs(delta) <= 0.00000001f * std::abs(nodes[offset + j])) {
                        break;
                    }

                }

                float dummy, L_N_plus1_prime_final;
                legendre_polynomial_and_derivative(N + 1, nodes[offset + j], dummy, L_N_plus1_prime_final);
                nodes[offset + N - j] = -nodes[offset + j];
                weights[offset + j] = 2.0f/((1.0f - std::pow(nodes[offset + j], 2)) * std::pow(L_N_plus1_prime_final, 2));
                weights[offset + N - j] = weights[offset + j];
            }
        }

        if (N % 2 == 0) {
            float dummy, L_N_plus1_prime_final;
            legendre_polynomial_and_derivative(N + 1, 0.0f, dummy, L_N_plus1_prime_final);
            nodes[offset + N/2] = 0.0f;
            weights[offset + N/2] = 2/std::pow(L_N_plus1_prime_final, 2);
        }
    }
}

// Algorithm 30
__global__
void SEM::calculate_barycentric_weights(int N, const float* nodes, float* barycentric_weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int j = index; j <= N; j += stride) {
        float xjxi = 1.0f;
        for (int i = 0; i < j; ++i) {
            xjxi *= nodes[offset + j] - nodes[offset + i];
        }
        for (int i = j + 1; i <= N; ++i) {
            xjxi *= nodes[offset + j] - nodes[offset + i];
        }

        barycentric_weights[offset + j] = 1.0f/xjxi;
    }
}

/*__device__
bool almost_equal(float a, float b) {
    return (std::abs(a) > std::numeric_limits<float>::min()) * (std::abs(b) > std::numeric_limits<float>::min()) * ((std::abs(a - b) <= std::numeric_limits<float>::epsilon() * a) * (std::abs(a - b) <= std::numeric_limits<float>::epsilon() * b)) 
    + (1 - (std::abs(a) > std::numeric_limits<float>::min()) * (std::abs(b) > std::numeric_limits<float>::min())) * (std::abs(a - b) <= std::numeric_limits<float>::epsilon() * 2);
}*/

// From cppreference.com
__device__
bool SEM::almost_equal(float x, float y) {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= FLT_EPSILON * std::abs(x+y) * ulp // CHECK change this to double equivalent if using double instead of float
        // unless the result is subnormal
        || std::abs(x-y) < FLT_MIN; // CHECK change this to 64F if using double instead of float
}

// This will not work if we are on a node, or at least be pretty inefficient
// Algorithm 34
__global__
void SEM::lagrange_interpolating_polynomials(float x, int N, const float* nodes, const float* barycentric_weights, float* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_interpolant[offset + i] = barycentric_weights[offset + i] / (x - nodes[offset + i]);
    }
}

// Algorithm 34
__global__
void SEM::normalize_lagrange_interpolating_polynomials(int N_max, float* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int N = index; N <= N_max; N += stride) {
        const int offset = N * (N + 1) /2;
        float sum = 0.0f;
        for (int i = 0; i <= N; ++i) {
            sum += lagrange_interpolant[offset + i];
        }
        for (int i = 0; i <= N; ++i) {
            lagrange_interpolant[offset + i] /= sum;
        }
    }
}

// Be sure to compute the diagonal afterwards
// Algorithm 37
__global__
void SEM::polynomial_derivative_matrices(int N, const float* nodes, const float* barycentric_weights, float* derivative_matrices) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const int offset_1D = N * (N + 1) /2;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            if (i != j) { // CHECK remove for branchless, i == j will be overwritten anyway
                derivative_matrices[offset_2D + i * (N + 1) + j] = barycentric_weights[offset_1D + j] / (barycentric_weights[offset_1D + i] * (nodes[offset_1D + i] - nodes[offset_1D + j]));
            }
        }
    }
}

// Algorithm 37
__global__
void SEM::polynomial_derivative_matrices_diagonal(int N, float* derivative_matrices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index; i <= N; i += stride) {
        derivative_matrices[offset_2D + i * (N + 2)] = 0.0f;
        for (int j = 0; j < i; ++j) {
            derivative_matrices[offset_2D + i * (N + 2)] -= derivative_matrices[offset_2D + i * (N + 1) + j];
        }
        for (int j = i + 1; j <= N; ++j) {
            derivative_matrices[offset_2D + i * (N + 2)] -= derivative_matrices[offset_2D + i * (N + 1) + j];
        }
    }
}

__global__
void SEM::polynomial_derivative_matrices_hat(int N, const float* weights, const float* derivative_matrices, float* derivative_matrices_hat) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const int offset_1D = N * (N + 1) /2;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            derivative_matrices_hat[offset_2D + i * (N + 1) + j] = derivative_matrices[offset_2D + j * (N + 1) + i] * weights[offset_1D + j] / weights[offset_1D + i];
        }
    }
}

// Will interpolate N_interpolation_points between -1 and 1
__global__
void SEM::create_interpolation_matrices(int N, int N_interpolation_points, const float* nodes, const float* barycentric_weights, float* interpolation_matrices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset_1D = N * (N + 1) /2;
    const int offset_interp = N * (N + 1) * N_interpolation_points/2;

    for (int j = index; j < N_interpolation_points; j += stride) {
        bool row_has_match = false;
        const float x_coord = 2.0f * j / (N_interpolation_points - 1) - 1.0f;

        for (int k = 0; k <= N; ++k) {
            interpolation_matrices[offset_interp + j * (N + 1) + k] = 0.0f;
            if (almost_equal(x_coord, nodes[offset_1D + k])) {
                interpolation_matrices[offset_interp + j * (N + 1) + k] = 1.0f;
                row_has_match = true;
            }
        }

        if (!row_has_match) {
            float total = 0.0f;
            for (int k = 0; k <= N; ++k) {
                interpolation_matrices[offset_interp + j * (N + 1) + k] = barycentric_weights[offset_1D + k] / (x_coord - nodes[offset_1D + k]);
                total += interpolation_matrices[offset_interp + j * (N + 1) + k];
            }
            for (int k = 0; k <= N; ++k) {
                interpolation_matrices[offset_interp + j * (N + 1) + k] /= total;
            }
        }
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

// Algorithm 61
__device__
float SEM::interpolate_to_boundary(int N, const float* phi, const float* lagrange_interpolant) {
    const int offset_1D = N * (N + 1) /2;
    float result = 0.0f;

    for (int j = 0; j <= N; ++j) {
        result += lagrange_interpolant[offset_1D + j] * phi[j];
    }

    return result;
}


NDG_t::NDG_t(int N_max, int N_interpolation_points) : 
        N_max_(N_max), 
        N_interpolation_points_(N_interpolation_points),
        vector_length_((N_max_ + 1) * (N_max_ + 2)/2), 
        matrix_length_((N_max_ + 1) * (N_max_ + 2) * (2 * N_max_ + 3)/6),
        interpolation_length_((N_max_ + 1) * (N_max_ + 2) * N_interpolation_points_/2) {

    cudaMalloc(&nodes_, vector_length_ * sizeof(float));
    cudaMalloc(&weights_, vector_length_ * sizeof(float));
    cudaMalloc(&barycentric_weights_, vector_length_ * sizeof(float));
    cudaMalloc(&lagrange_interpolant_left_, vector_length_ * sizeof(float));
    cudaMalloc(&lagrange_interpolant_right_, vector_length_ * sizeof(float));
    cudaMalloc(&derivative_matrices_, matrix_length_ * sizeof(float));
    cudaMalloc(&derivative_matrices_hat_, matrix_length_ * sizeof(float));
    cudaMalloc(&interpolation_matrices_, interpolation_length_ * sizeof(float));

    for (int N = 0; N <= N_max_; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        legendre_gauss_nodes_and_weights<<<vector_numBlocks, poly_blockSize>>>(N, nodes_, weights_);
    }

    // Nodes are needed to compute barycentric weights
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max_; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        calculate_barycentric_weights<<<vector_numBlocks, poly_blockSize>>>(N, nodes_, barycentric_weights_);
    }

    // We need the barycentric weights for derivative matrix, interpolation matrices and Lagrange interpolants
    cudaDeviceSynchronize();
    const int interpolation_numBlocks = (N_interpolation_points_ + interpolation_blockSize) / interpolation_blockSize;
    for (int N = 0; N <= N_max_; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices<<<matrix_numBlocks, matrix_blockSize>>>(N, nodes_, barycentric_weights_, derivative_matrices_);
        create_interpolation_matrices<<<interpolation_numBlocks, interpolation_blockSize>>>(N, N_interpolation_points_, nodes_, barycentric_weights_, interpolation_matrices_);
        lagrange_interpolating_polynomials<<<vector_numBlocks, poly_blockSize>>>(-1.0f, N, nodes_, barycentric_weights_, lagrange_interpolant_left_);
        lagrange_interpolating_polynomials<<<vector_numBlocks, poly_blockSize>>>(1.0f, N, nodes_, barycentric_weights_, lagrange_interpolant_right_);
    }

    // Then we calculate the derivative matrix diagonal and normalize the Lagrange interpolants
    cudaDeviceSynchronize();
    const int poly_numBlocks = (N_max_ + poly_blockSize) / poly_blockSize;
    normalize_lagrange_interpolating_polynomials<<<poly_numBlocks, poly_blockSize>>>(N_max_, lagrange_interpolant_left_);
    normalize_lagrange_interpolating_polynomials<<<poly_numBlocks, poly_blockSize>>>(N_max_, lagrange_interpolant_right_);
    for (int N = 0; N <= N_max_; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices_diagonal<<<vector_numBlocks, poly_blockSize>>>(N, derivative_matrices_);
    }

    // All the derivative matrix has to be computed before D^
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max_; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices_hat<<<matrix_numBlocks, matrix_blockSize>>>(N, weights_, derivative_matrices_, derivative_matrices_hat_);
    }
}

NDG_t::~NDG_t() {
    // Not sure if null checks are needed
    if (nodes_ != nullptr){
        cudaFree(nodes_);
    }
    if (weights_ != nullptr){
        cudaFree(weights_);
    }
    if (barycentric_weights_ != nullptr){
        cudaFree(barycentric_weights_);
    }
    if (lagrange_interpolant_left_ != nullptr){
        cudaFree(lagrange_interpolant_left_);
    }
    if (lagrange_interpolant_right_ != nullptr){
        cudaFree(lagrange_interpolant_right_);
    }
    if (derivative_matrices_ != nullptr){
        cudaFree(derivative_matrices_);
    }
    if (derivative_matrices_hat_ != nullptr){
        cudaFree(derivative_matrices_hat_);
    }
    if (interpolation_matrices_ != nullptr){
        cudaFree(interpolation_matrices_);
    }
}

   

void NDG_t::print() {
    // Copy vectors from device memory to host memory
    float* host_nodes = new float[vector_length_];
    float* host_weights = new float[vector_length_];
    float* host_barycentric_weights = new float[vector_length_];
    float* host_lagrange_interpolant_left = new float[vector_length_];
    float* host_lagrange_interpolant_right = new float[vector_length_];
    float* host_derivative_matrices = new float[matrix_length_];
    float* host_derivative_matrices_hat = new float[matrix_length_];
    float* host_interpolation_matrices = new float[interpolation_length_];

    cudaDeviceSynchronize();

    cudaMemcpy(host_nodes, nodes_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights, weights_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights, barycentric_weights_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left, lagrange_interpolant_left_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right, lagrange_interpolant_right_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices, derivative_matrices_, matrix_length_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices_hat, derivative_matrices_hat_, matrix_length_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_interpolation_matrices, interpolation_matrices_, interpolation_length_ * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Nodes: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_nodes[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Weights: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl << "Barycentric weights: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_barycentric_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants -1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_left[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants +1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_right[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Derivative matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << host_derivative_matrices[offset_2D + i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "Derivative matrices hat: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << host_derivative_matrices_hat[offset_2D + i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    /*std::cout << std::endl << "Interpolation matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const int offset_interp = N * (N + 1) * N_interpolation_points_/2;

        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i < N_interpolation_points_; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << host_interpolation_matrices[offset_interp + i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }*/

    delete[] host_nodes;
    delete[] host_weights;
    delete[] host_barycentric_weights;
    delete[] host_lagrange_interpolant_left;
    delete[] host_lagrange_interpolant_right;
    delete[] host_derivative_matrices;
    delete[] host_derivative_matrices_hat;
    delete[] host_interpolation_matrices;
}