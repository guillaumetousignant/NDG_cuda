#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>
#include <cfloat>
#include <filesystem>

namespace fs = std::filesystem;

constexpr float pi = 3.14159265358979323846f;
constexpr int poly_blockSize = 16; // Small number of threads per block because N will never be huge
constexpr int elements_blockSize = 32; // For when we'll have multiple elements
constexpr int faces_blockSize = 32; // Same number of faces as elements for periodic BC
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
void lagrange_integrating_polynomials(float x, int N, const float* nodes, const float* weights, float* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_interpolant[offset + i] = weights[offset + i] / (x - nodes[offset + i]);
    }
}

// Algorithm 34
__global__
void normalize_lagrange_integrating_polynomials(int N_max, float* lagrange_interpolant) {
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
            derivative_matrices_hat[offset_2D + i * (N + 1) + j] = derivative_matrices[offset_2D + j * (N + 1) + i] * weights[offset_1D + j] / weights[offset_1D + i];
        }
    }
}

// Algorithm 19
__device__
void matrix_vector_derivative(int N, const float* derivative_matrices, const float* phi, float* phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = 0; i <= N; ++i) {
        phi_prime[i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            phi_prime[i] += derivative_matrices[offset_2D + i * (N + 1) + j] * phi[j] * phi[j] * 0.5f; // phi not squared in textbook, squared for Burger's
        }
    }
}

class NDG_t { 
public: 
    NDG_t(int N_max) : 
            N_max_(N_max), 
            vector_length_((N_max_ + 1) * (N_max_ + 2)/2), 
            matrix_length_((N_max_ + 1) * (N_max_ + 2) * (2 * N_max_ + 3)/6) {

        cudaMalloc(&nodes_, vector_length_ * sizeof(float));
        cudaMalloc(&weights_, vector_length_ * sizeof(float));
        cudaMalloc(&barycentric_weights_, vector_length_ * sizeof(float));
        cudaMalloc(&lagrange_interpolant_left_, vector_length_ * sizeof(float));
        cudaMalloc(&lagrange_interpolant_right_, vector_length_ * sizeof(float));
        cudaMalloc(&derivative_matrices_, matrix_length_ * sizeof(float));
        cudaMalloc(&derivative_matrices_hat_, matrix_length_ * sizeof(float));

        for (int N = 0; N <= N_max_; ++N) {
            const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
            chebyshev_gauss_nodes_and_weights<<<vector_numBlocks, poly_blockSize>>>(N, nodes_, weights_);
        }

        // Nodes are needed to compute barycentric weights
        cudaDeviceSynchronize();
        for (int N = 0; N <= N_max_; ++N) {
            const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
            calculate_barycentric_weights<<<vector_numBlocks, poly_blockSize>>>(N, nodes_, barycentric_weights_);
            lagrange_integrating_polynomials<<<vector_numBlocks, poly_blockSize>>>(-1.0f, N, nodes_, weights_, lagrange_interpolant_left_);
            lagrange_integrating_polynomials<<<vector_numBlocks, poly_blockSize>>>(1.0f, N, nodes_, weights_, lagrange_interpolant_right_);
        }

        // We need to divide lagrange_integrating_polynomials by sum, and barycentric weights for derivative matrix
        cudaDeviceSynchronize();
        const int numBlocks = (N_max_ + poly_blockSize) / poly_blockSize;
        normalize_lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(N_max_, lagrange_interpolant_left_);
        normalize_lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(N_max_, lagrange_interpolant_right_);
        for (int N = 0; N <= N_max_; ++N) {
            const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
            polynomial_derivative_matrices<<<matrix_numBlocks, matrix_blockSize>>>(N, nodes_, barycentric_weights_, derivative_matrices_);
        }

        // Then we calculate the derivative matrix diagonal
        cudaDeviceSynchronize();
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
    }

    int N_max_;
    int vector_length_; // Flattened length of all N one after the other
    int matrix_length_; // Flattened length of all N² one after the other
    float* nodes_;
    float* weights_;
    float* barycentric_weights_;
    float* lagrange_interpolant_left_;
    float* lagrange_interpolant_right_;
    float* derivative_matrices_;
    float* derivative_matrices_hat_;

    void print() {
        // Copy vectors from device memory to host memory
        float* host_nodes = new float[vector_length_];
        float* host_weights = new float[vector_length_];
        float* host_barycentric_weights = new float[vector_length_];
        float* host_lagrange_interpolant_left = new float[vector_length_];
        float* host_lagrange_interpolant_right = new float[vector_length_];
        float* host_derivative_matrices = new float[matrix_length_];
        float* host_derivative_matrices_hat = new float[matrix_length_];

        cudaDeviceSynchronize();

        cudaMemcpy(host_nodes, nodes_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_weights, weights_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_barycentric_weights, barycentric_weights_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_lagrange_interpolant_left, lagrange_interpolant_left_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_lagrange_interpolant_right, lagrange_interpolant_right_, vector_length_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_derivative_matrices, derivative_matrices_, matrix_length_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_derivative_matrices_hat, derivative_matrices_hat_, matrix_length_ * sizeof(float), cudaMemcpyDeviceToHost);

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

        delete host_nodes;
        delete host_weights;
        delete host_barycentric_weights;
        delete host_lagrange_interpolant_left;
        delete host_lagrange_interpolant_right;
        delete host_derivative_matrices;
        delete host_derivative_matrices_hat;
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
    }

    __device__
    ~Element_t() {
        if (phi_ != nullptr){
            delete [] phi_;
        }
        if (phi_prime_ != nullptr) {
            delete [] phi_prime_;
        }
        if (intermediate_ != nullptr) {
            delete [] intermediate_;
        }
    }

    int N_;
    int neighbours_[2]; // Could also be pointers
    int faces_[2]; // Could also be pointers
    float x_[2];
    float delta_x_;
    float phi_L_;
    float phi_R_;
    float* phi_; // Solution
    float* phi_prime_;
    float* intermediate_;
};

__global__
void build_elements(int N_elements, int N, Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int neighbour_L = (i > 0) ? i - 1 : N_elements - 1; // First cell has last cell as left neighbour
        const int neighbour_R = (i < N_elements - 1) ? i + 1 : 0; // Last cell has first cell as right neighbour
        const int face_L = (i > 0) ? i - 1 : N_elements - 1;
        const int face_R = i;
        elements[i] = Element_t(N, neighbour_L, neighbour_R, face_L, face_R, -1.0f, 1.0f);
    }
}

__global__
void initial_conditions(int N_elements, Element_t* elements, const float* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int offset = elements[i].N_ * (elements[i].N_ + 1) /2;
        elements[i].phi_L_ = -sin(-pi);
        elements[i].phi_R_ = -sin(pi);
        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_[j] = -sin(pi * nodes[offset + j]);
        }
    }
}

// Basically useless, find better solution when multiple elements.
__global__
void get_phi_phi_prime(int N_elements, const Element_t* elements, float* phi, float* phi_prime) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
            phi_prime[j] = elements[i].phi_prime_[j];
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

// Algorithm 60
__device__
void compute_dg_derivative(Element_t &element, const float* weights, const float* derivative_matrices, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right) {
    const int offset_1D = element.N_ * (element.N_ + 1) /2;
    
    matrix_vector_derivative(element.N_, derivative_matrices, element.phi_, element.phi_prime_);

    for (int j = 0; j <= element.N_; ++j) {
        element.phi_prime_[j] += (element.phi_L_ * element.phi_L_ * lagrange_interpolant_left[offset_1D + j] - element.phi_R_ * element.phi_R_ * lagrange_interpolant_right[offset_1D + j]) / (2 * weights[offset_1D + j]);
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

// Algorithm 61 (not really anymore)
__device__
void compute_dg_time_derivative(Element_t &element, const float* weights, const float* derivative_matrices, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right) {
    element.phi_L_ = interpolate_to_boundary(element.N_, element.phi_, lagrange_interpolant_left);
    element.phi_R_ = interpolate_to_boundary(element.N_, element.phi_, lagrange_interpolant_right);
    
    compute_dg_derivative(element, weights, derivative_matrices, lagrange_interpolant_left, lagrange_interpolant_right); // Multiplied by -c in textbook, here multiplied by phi in matrix_vector_derivative
}

// Algorithm 62
// Not used anymore, needs to calculate fluxes between elements between steps. Split between the following methods
__global__
void gd_step_by_rk3(int N_elements, Element_t* elements, float delta_t, const float* weights, const float* derivative_matrices, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        // Unrolled loop from m = 1 to 3
        // a_m and b_ at textbook p.99
        compute_dg_time_derivative(elements[i], weights, derivative_matrices, lagrange_interpolant_left, lagrange_interpolant_right);
        for (int j = 0; j <= elements[i].N_; ++j){
            elements[i].intermediate_[j] = elements[i].phi_prime_[j];
            elements[i].phi_[j] += delta_t * elements[i].intermediate_[j] /3.0f;
        }

        compute_dg_time_derivative(elements[i], weights, derivative_matrices, lagrange_interpolant_left, lagrange_interpolant_right);
        for (int j = 0; j <= elements[i].N_; ++j){
            elements[i].intermediate_[j] = -5.0f * elements[i].intermediate_[j] / 9.0f + elements[i].phi_prime_[j];
            elements[i].phi_[j] += 15.0f * delta_t * elements[i].intermediate_[j] /16.0f;
        }

        compute_dg_time_derivative(elements[i], weights, derivative_matrices, lagrange_interpolant_left, lagrange_interpolant_right);
        for (int j = 0; j <= elements[i].N_; ++j){
            elements[i].intermediate_[j] = -153.0f * elements[i].intermediate_[j] / 128.0f + elements[i].phi_prime_[j];
            elements[i].phi_[j] += 8.0f * delta_t * elements[i].intermediate_[j] /15.0f;
        }
    }
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
        elements[i].phi_R_ = interpolate_to_boundary(elements[i.]N_, elements[i].phi_, lagrange_interpolant_right);
    }
}

__global__
void calculate_fluxes(int N_faces, const Face_t* faces, Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_faces; i += stride) {
        float u;
        float u_left = elements[faces[i].elements_[0]].phi_R_;
        float u_right = elements[faces[i].elements_[1]].phi_L_;

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
            else {
                u = u_right;
            }
        }

        faces[i].flux_ = 0.5f * u * u;
    }
}

class Face_t {
public:
    __device__ 
    Face_t(int element_L, int element_R) : elements_{element_L, element_R} {}

    __device__
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

class Mesh_t {
public:
    Mesh_t(int N_elements, int initial_N) : N_elements_(N_elements), N_faces_(N_elements), initial_N_(initial_N) {
        // CHECK N_faces = N_elements only for periodic BC.
        cudaMalloc(&elements_, N_elements_ * sizeof(Element_t));
        cudaMalloc(&faces_, N_faces_ * sizeof(Face_t));

        const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
        const int faces_numBlocks = (N_faces_ + faces_blockSize - 1) / faces_blockSize;
        build_elements<<<elements_numBlocks, elements_blockSize>>>(N_elements_, initial_N_, elements_);
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

    void solve(const NDG_t &NDG) {
        const int N_steps = 600;
        const float delta_t = 0.001f;
        const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
        float time = 0.0;

        write_data(time, NDG.nodes_);

        for (int step = 0; step < N_steps; ++step) {
            time += delta_t;
            gd_step_by_rk3<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            
            interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
            rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, 0.0f, 1.0f/3.0f);

            interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
            rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, -5.0f/9.0f, 15.0f/16.0f);

            interpolate_to_boundaries<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
            calculate_fluxes<<<faces_numBlocks, faces_blockSize>>>(N_faces_, faces_, elements_);
            rk3_step<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, delta_t, -153.0f/128.0f, 8.0f/15.0f);
                  
            if (step % 100 == 0) {
                write_data(time, NDG.nodes_);
            }
        }

        write_data(time, NDG.nodes_);
    }

    void print() {
        // CHECK find better solution for multiple elements
        float* phi;
        float* phi_prime;
        float* host_phi = new float[initial_N_ + 1];
        float* host_phi_prime = new float[initial_N_ + 1];
        cudaMalloc(&phi, (initial_N_ + 1) * sizeof(float));
        cudaMalloc(&phi_prime, (initial_N_ + 1) * sizeof(float));

        const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
        get_phi_phi_prime<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, phi, phi_prime);
        
        cudaDeviceSynchronize();
        cudaMemcpy(host_phi, phi, (initial_N_ + 1) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_phi_prime, phi_prime, (initial_N_ + 1) * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << std::endl << "Phi: " << std::endl;
        for (int i = 0; i < N_elements_; ++i) {
            std::cout << '\t' << "Element " << i << ": ";
            std::cout << '\t' << '\t';
            for (int j = 0; j <= initial_N_; ++j) {
                std::cout << host_phi[j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Phi prime: " << std::endl;
        for (int i = 0; i < N_elements_; ++i) {
            std::cout << '\t' << "Element " << i << ": ";
            std::cout << '\t' << '\t';
            for (int j = 0; j <= initial_N_; ++j) {
                std::cout << host_phi_prime[j] << " ";
            }
            std::cout << std::endl;
        }

        delete host_phi;
        delete host_phi_prime;

        cudaFree(phi);
        cudaFree(phi_prime);
    }

    void write_file_data(int N, float time, const float* velocity, const float* coordinates) {
        std::stringstream ss;
        std::ofstream file;
    
        fs::path save_dir = fs::current_path() / "data";
        fs::create_directory(save_dir);
    
        ss << "output_t" << std::setprecision(4) << std::fixed << time << ".dat";
        file.open(save_dir / ss.str());
    
        file << "TITLE = \"Velocity  at t= " << time << "\"" << std::endl;
        file << "VARIABLES = \"X\", \"U_x\"" << std::endl;
        file << "ZONE T= \"Zone     1\",  I= " << N + 1 << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;
    
        for (int i = 0; i <= N; ++i) {
            file << std::setw(12) << coordinates[i] << " " << std::setw(12) << velocity[i] << std::endl;
        }
    
        file.close();
    }

    void write_data(float time, const float* nodes) {
        // CHECK find better solution for multiple elements
        float* phi;
        float* host_phi = new float[initial_N_ + 1];
        float* host_nodes = new float[initial_N_ + 1];
        cudaMalloc(&phi, (initial_N_ + 1) * sizeof(float));
        const int offset_1D = initial_N_ * (initial_N_ + 1) /2;

        const int elements_numBlocks = (N_elements_ + elements_blockSize - 1) / elements_blockSize;
        get_phi<<<elements_numBlocks, elements_blockSize>>>(N_elements_, elements_, phi);
        
        cudaDeviceSynchronize();
        cudaMemcpy(host_phi, phi, (initial_N_ + 1) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_nodes, nodes + offset_1D, (initial_N_ + 1) * sizeof(float), cudaMemcpyDeviceToHost);

        write_file_data(initial_N_, time, host_phi, host_nodes);

        delete host_phi;
        delete host_nodes;
        cudaFree(phi);
    }
};

int main(void) {
    const int N_elements = 1;
    const int initial_N = 64;
    const int N_max = 128;
    
    NDG_t NDG(N_max);
    Mesh_t Mesh(N_elements, initial_N);
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

    //NDG.print();
    //Mesh.print();
    
    return 0;
}