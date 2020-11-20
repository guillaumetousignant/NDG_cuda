#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>
#include <cfloat>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

constexpr float pi = 3.14159265358979323846f;
constexpr float c = 5.0f;
constexpr int poly_blockSize = 16; // Small number of threads per block because N will never be huge
constexpr int elements_blockSize = 32; // For when we'll have multiple elements
constexpr int faces_blockSize = 32; // Same number of faces as elements for periodic BC
constexpr int interpolation_blockSize = 32;
const dim3 matrix_blockSize(16, 16); // Small number of threads per block because N will never be huge

// Algorithm 26
__global__
void chebyshev_gauss_nodes_and_weights(int N, float* nodes, float* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        nodes[offset + i] = -cos(pi * (2 * i + 1) / (2 * N + 2));
        weights[offset + i] = pi / (N + 1);
    }
}

// Algorithm 22
__device__
void legendre_polynomial_and_derivative(int N, float x, float &L_N, float &L_N_prime) {
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
void legendre_gauss_nodes_and_weights(int N, float* nodes, float* weights) {
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
    }

    if (N % 2 == 0) {
        float dummy, L_N_plus1_prime_final;
        legendre_polynomial_and_derivative(N + 1, 0.0f, dummy, L_N_plus1_prime_final);
        nodes[offset + N/2] = 0.0f;
        weights[offset + N/2] = 2/std::pow(L_N_plus1_prime_final, 2);
    }
}

// Algorithm 30
__global__
void calculate_barycentric_weights(int N, const float* nodes, float* barycentric_weights) {
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
bool almost_equal(float x, float y) {
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
void lagrange_interpolating_polynomials(float x, int N, const float* nodes, const float* barycentric_weights, float* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_interpolant[offset + i] = barycentric_weights[offset + i] / (x - nodes[offset + i]);
    }
}

// Algorithm 34
__global__
void normalize_lagrange_interpolating_polynomials(int N_max, float* lagrange_interpolant) {
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
void polynomial_derivative_matrices(int N, const float* nodes, const float* barycentric_weights, float* derivative_matrices) {
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
void polynomial_derivative_matrices_diagonal(int N, float* derivative_matrices) {
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
void polynomial_derivative_matrices_hat(int N, const float* weights, const float* derivative_matrices, float* derivative_matrices_hat) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const int offset_1D = N * (N + 1) /2;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            derivative_matrices_hat[offset_2D + i * (N + 1) + j] = -derivative_matrices[offset_2D + j * (N + 1) + i] * weights[offset_1D + j] / weights[offset_1D + i];
        }
    }
}

// Will interpolate N_interpolation_points between -1 and 1
__global__
void create_interpolation_matrices(int N, int N_interpolation_points, const float* nodes, const float* barycentric_weights, float* interpolation_matrices) {
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
void matrix_vector_derivative(int N, const float* derivative_matrices_hat, const float* phi, float* phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = 0; i <= N; ++i) {
        phi_prime[i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            phi_prime[i] += derivative_matrices_hat[offset_2D + i * (N + 1) + j] * phi[j]; // phi not squared in textbook, squared for Burger's
        }
    }
}

class NDG_t { 
public: 
    NDG_t(int N_max, int N_interpolation_points) : 
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

    ~NDG_t() {
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

    int N_max_;
    int N_interpolation_points_;
    int vector_length_; // Flattened length of all N one after the other
    int matrix_length_; // Flattened length of all N² one after the other
    int interpolation_length_;
    float* nodes_;
    float* weights_;
    float* barycentric_weights_;
    float* lagrange_interpolant_left_;
    float* lagrange_interpolant_right_;
    float* derivative_matrices_;
    float* derivative_matrices_hat_;
    float* interpolation_matrices_;

    void print() {
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

        const int N = 8;
        const int offset = N * (N + 1) /2;
        float* u = new float[N + 1];
        float* u_prime = new float[N + 1];
        float* u_prime_expected = new float[N + 1];
        float u_L = 0.0f;
        float u_R = 0.0f;

        for (int i = 0; i <= N; ++i) {
            u[i] = -sin(pi * host_nodes[offset + i]);
        }

        const int offset_2D = N * (N + 1) * (2 * N + 1) /6;
        for (int i = 0; i <= N; ++i) {
            u_prime[i] = 0.0f;
            for (int j = 0; j <= N; ++j) {
                u_prime[i] += host_derivative_matrices[offset_2D + i * (N + 1) + j] * u[j];
            }
        }

        for (int i = 0; i <= N; ++i) {
            u_prime_expected[i] = -pi * cos(pi * host_nodes[offset + i]);
        }
        
        for (int i = 0; i <= N; ++i) {
            u_L += host_lagrange_interpolant_left[offset + i] * u[i];
        }

        for (int i = 0; i <= N; ++i) {
            u_R += host_lagrange_interpolant_right[offset + i] * u[i];
        }

        std::cout << "x:" << std::endl;
        std::cout << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << std::setw(12) << host_nodes[offset + i] << "    ";
        }
        std::cout << std::endl;

        std::cout << "u:" << std::endl;
        std::cout << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << std::setw(12) << u[i] << "    ";
        }
        std::cout << std::endl;

        std::cout << "u prime:" << std::endl;
        std::cout << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << std::setw(12) << u_prime[i] << "    ";
        }
        std::cout << std::endl;

        std::cout << "Expected u prime:" << std::endl;
        std::cout << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << std::setw(12) << u_prime_expected[i] << "    ";
        }
        std::cout << std::endl;

        std::cout << "Interpolated u:" << std::endl;
        std::cout << '\t';
        std::cout << std::setw(12) << u_L << "    ";
        std::cout << std::setw(12) << u_R;
        std::cout << std::endl;
        

        delete[] u;
        delete[] u_prime;
        delete[] u_prime_expected;

        delete[] host_nodes;
        delete[] host_weights;
        delete[] host_barycentric_weights;
        delete[] host_lagrange_interpolant_left;
        delete[] host_lagrange_interpolant_right;
        delete[] host_derivative_matrices;
        delete[] host_derivative_matrices_hat;
        delete[] host_interpolation_matrices;
    }
};

class Element_t { // Turn this into separate vectors, because cache exists
public:
    __device__ 
    Element_t(int N, int neighbour_L, int neighbour_R, int face_L, int face_R, float x_L, float x_R) : 
            N_(N),
            neighbours_{neighbour_L, neighbour_R},
            faces_{face_L, face_R},
            x_{x_L, x_R},
            delta_x_(x_R - x_L) {
        phi_ = new float[N_ + 1];
        phi_prime_ = new float[N_ + 1];
        intermediate_ = new float[N_ + 1];
        for (int i = 0; i <= N_; ++i) {
            intermediate_[i] = 0.0f;
        }
    }

    __host__ 
    Element_t() {};

    __host__ __device__
    ~Element_t() {
        if (phi_ != nullptr){
            delete[] phi_;
        }
        if (phi_prime_ != nullptr) {
            delete[] phi_prime_;
        }
        if (intermediate_ != nullptr) {
            delete[] intermediate_;
        }
    }

    int N_;
    int neighbours_[2]; // Could also be pointers
    int faces_[2]; // Could also be pointers. left, right
    float x_[2];
    float delta_x_;
    float phi_L_;
    float phi_R_;
    float* phi_; // Solution
    float* phi_prime_;
    float* intermediate_;
};

__global__
void build_elements(int N_elements, int N, Element_t* elements, float x_min, float x_max) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int neighbour_L = (i > 0) ? i - 1 : N_elements - 1; // First cell has last cell as left neighbour
        const int neighbour_R = (i < N_elements - 1) ? i + 1 : 0; // Last cell has first cell as right neighbour
        const int face_L = (i > 0) ? i - 1 : N_elements - 1;
        const int face_R = i;
        const float delta_x = (x_max - x_min)/N_elements;
        const float element_x_min = x_min + i * delta_x;
        const float element_y_min = x_min + (i + 1) * delta_x;
        elements[i] = Element_t(N, neighbour_L, neighbour_R, face_L, face_R, element_x_min, element_y_min);
    }
}

__device__
float g(float t, float x) {
    constexpr float sigma = 0.2;
    return std::exp(-std::log(2.0f) * std::pow(x - t * c, 2) / std::pow(sigma, 2));
}


__global__
void initial_conditions(int N_elements, Element_t* elements, const float* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int offset = elements[i].N_ * (elements[i].N_ + 1) /2;
        for (int j = 0; j <= elements[i].N_; ++j) {
            const float x = (0.5 + nodes[offset + j]/2.0f) * (elements[i].x_[1] - elements[i].x_[0]) + elements[i].x_[0];
            //elements[i].phi_[j] = -sin(pi * x);
            elements[i].phi_[j] = g(0.0f, x);
        }
    }
}

// Basically useless, find better solution when multiple elements.
__global__
void get_elements_data(int N_elements, const Element_t* elements, float* phi, float* phi_prime) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int element_offset = i * (elements[i].N_ + 1);
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[element_offset + j] = elements[i].phi_[j];
            phi_prime[element_offset + j] = elements[i].phi_prime_[j];
        }
    }
}

// Basically useless, find better solution when multiple elements.
__global__
void get_phi(int N_elements, const Element_t* elements, float* phi) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
        }
    }
}

__global__
void get_solution(int N_elements, int N_interpolation_points, const Element_t* elements, const float* interpolation_matrices, float* phi, float* x) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int offset_interp_1D = i * N_interpolation_points;
        const int offset_interp = elements[i].N_ * (elements[i].N_ + 1) * N_interpolation_points/2;

        for (int j = 0; j < N_interpolation_points; ++j) {
            phi[offset_interp_1D + j] = 0.0f;
            for (int k = 0; k <= elements[i].N_; ++k) {
                phi[offset_interp_1D + j] += interpolation_matrices[offset_interp + j * (elements[i].N_ + 1) + k] * elements[i].phi_[k];
            }
            x[offset_interp_1D + j] = j * (elements[i].x_[1] - elements[i].x_[0]) / (N_interpolation_points - 1) + elements[i].x_[0];
        }
    }
}

// Algorithm 61
__device__
float interpolate_to_boundary(int N, const float* phi, const float* lagrange_interpolant) {
    const int offset_1D = N * (N + 1) /2;
    float result = 0.0f;

    for (int j = 0; j <= N; ++j) {
        result += lagrange_interpolant[offset_1D + j] * phi[j];
    }

    return result;
}

__global__
void rk3_step(int N_elements, Element_t* elements, float delta_t, float a, float g) {
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
void interpolate_to_boundaries(int N_elements, Element_t* elements, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        elements[i].phi_L_ = interpolate_to_boundary(elements[i].N_, elements[i].phi_, lagrange_interpolant_left);
        elements[i].phi_R_ = interpolate_to_boundary(elements[i].N_, elements[i].phi_, lagrange_interpolant_right);
    }
}

class Face_t {
public:
    __device__ 
    Face_t(int element_L, int element_R) : elements_{element_L, element_R} {}

    __host__
    Face_t() {}

    __host__ __device__
    ~Face_t() {}

    int elements_[2]; // left, right
    float flux_;
};

__global__
void build_faces(int N_faces, Face_t* faces) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_faces; i += stride) {
        const int neighbour_L = i;
        const int neighbour_R = (i < N_faces - 1) ? i + 1 : 0; // Last face links last element to first element
        faces[i] = Face_t(neighbour_L, neighbour_R);
    }
}

__global__
void calculate_fluxes(int N_faces, Face_t* faces, const Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_faces; i += stride) {
        const float u_left = elements[faces[i].elements_[0]].phi_R_;
        const float u_right = elements[faces[i].elements_[1]].phi_L_;

        if (c >= 0.0f) {
            faces[i].flux_ = u_left;
        }
        else  {
            faces[i].flux_ = u_right;
        }
    }
}

// Algorithm 60 (not really anymore)
__global__
void compute_dg_derivative(float t, int N_elements, Element_t* elements, const Face_t* faces, const float* weights, const float* derivative_matrices_hat, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int offset_1D = elements[i].N_ * (elements[i].N_ + 1) /2; // CHECK cache?

        float flux_L = faces[elements[i].faces_[0]].flux_;
        float flux_R = faces[elements[i].faces_[1]].flux_;

        if (c > 0.0f) {
            flux_L = g(t, -1.0f);
            flux_R = elements[i].phi_R_;
        } else {
            flux_L = elements[i].phi_L_;
            flux_R =  g(t, 1.0f);
        }

        matrix_vector_derivative(elements[i].N_, derivative_matrices_hat, elements[i].phi_, elements[i].phi_prime_);

        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_prime_[j] += (flux_R * lagrange_interpolant_right[offset_1D + j] - flux_L * lagrange_interpolant_left[offset_1D + j]) / weights[offset_1D + j];
            elements[i].phi_prime_[j] *= -c * 2.0f/elements[i].delta_x_;
        }
    }
}

class Mesh_t {
public:
    Mesh_t(int N_elements, int initial_N, float x_min, float x_max) : N_elements_(N_elements), N_faces_(N_elements), initial_N_(initial_N) {
        // CHECK N_faces = N_elements only for periodic BC.
        cudaMalloc(&elements_, N_elements_ * sizeof(Element_t));
        cudaMalloc(&faces_, N_faces_ * sizeof(Face_t));

        const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
        const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
        build_elements<<<elements_numBlocks, elements_blockSize>>>(N_elements_, initial_N_, elements_, x_min, x_max);
        build_faces<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_); // CHECK
    }

    ~Mesh_t() {
        if (elements_ != nullptr){
            cudaFree(elements_);
        }

        if (faces_ != nullptr){
            cudaFree(faces_);
        }
    }

    int N_elements_;
    int N_faces_;
    int initial_N_;
    Element_t* elements_;
    Face_t* faces_;

    void set_initial_conditions(const float* nodes) {
        const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
        initial_conditions<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, nodes);
    }

    void print() {
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

    void write_file_data(int N_points, float time, const float* velocity, const float* coordinates) {
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

    void write_data(float time, int N_interpolation_points, const float* interpolation_matrices) {
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

    void solve(const NDG_t &NDG) {
        const float delta_t = 0.00015f;
        const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
        const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
        float time = 0.0;
        std::vector<float> output_times{0.5f, 1.0f, 1.5f};
        const float t_end = output_times.back();

        write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

        while (time < (t_end + delta_t)) {
            // Kinda algorithm 62
            float t = time;
            interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
            compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(t, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, 0.0f, 1.0f/3.0f);

            t = time + 0.33333333333f * delta_t;
            interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
            compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(t, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, -5.0f/9.0f, 15.0f/16.0f);

            t = time + 0.75f * delta_t;
            interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
            compute_dg_derivative<<<elements_numBlocks, elements_blockSize>>>(t, N_elements_, elements_, faces_, NDG.weights_, NDG.derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
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
};

int main(void) {
    const int N_elements = 1;
    const int N_max = 8;
    const int initial_N = N_max;
    const int N_interpolation_points = 100;
    
    NDG_t NDG(N_max, N_interpolation_points);
    Mesh_t Mesh(N_elements, initial_N, -1.0f, 1.0f);
    Mesh.set_initial_conditions(NDG.nodes_);

    // Starting actual computation
    cudaDeviceSynchronize();
    auto t_start = std::chrono::high_resolution_clock::now();
    Mesh.solve(NDG);
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 
            << "s." << std::endl;

    NDG.print();
    Mesh.print();
    
    return 0;
}