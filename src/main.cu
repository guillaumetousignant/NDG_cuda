#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>
#include <cfloat>

constexpr float pi = 3.14159265358979323846;

class Element_t { // Turn this into separate vectors, because cache exists
public:
    __device__ 
    Element_t(int N) : N_(N) {
        phi_ = new float[N_ + 1];
        phi_prime_ = new float[N_ + 1];
    }

    __device__
    ~Element_t() {
        delete [] phi_;
        delete [] phi_prime_;
    }

    int N_;
    float phi_L_;
    float phi_R_;
    float boundary_L_;
    float boundary_R_;
    float* phi_; // Solution
    float* phi_prime_;
};

__global__
void build_elements(int N_elements, int N, Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        elements[i] = Element_t(N);
    }
}

__global__
void initial_conditions(int N_elements, Element_t* elements, const float* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int offset = elements[i].N_ * (elements[i].N_ + 1) /2;
        elements[i].boundary_L_ = -sin(-1.0f);
        elements[i].boundary_R_ = -sin(1.0f);
        elements[i].phi_L_ = elements[i].boundary_L_;
        elements[i].phi_R_ = elements[i].boundary_R_;
        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_[j] = -sin(nodes[offset + j]);
        }
    }
}

// Basically useless, find better solution when multiple elements.
__global__
void get_solution(int N_elements, const Element_t* elements, float* phi, float* phi_prime) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
            phi_prime[j] = elements[i].phi_prime_[j];
        }
    }
}

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
            derivative_matrices_hat[offset_2D + i * (N + 1) + j] = -derivative_matrices[offset_2D + j * (N + 1) + i] * weights[offset_1D + j] / weights[offset_1D + i];
        }
    }
}

// Algorithm 19
__device__
void matrix_vector_derivative(int N, const float* derivative_matrices, const float* phi, float* phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    const int offset_1D = N * (N + 1) /2;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = 0; i <= N; ++i) {
        phi_prime[offset_1D + i] = 0.0f;
        for (int j = 0; j <= N; ++j) {
            phi_prime[offset_1D + i] += derivative_matrices[offset_2D + i * (N + 1) + j] * phi[offset_1D + j] * phi[offset_1D + j]; // phi not squared in textbook, squared for Burger's
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
        result += lagrange_interpolant[offset_1D + j] * phi[offset_1D + j];
    }

    return result;
}

// Algorithm 61
__global__
void compute_dg_time_derivative(int N_elements, Element_t* elements, const float* weights, const float* derivative_matrices, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        elements[i].phi_L_ = interpolate_to_boundary(elements[i].N_, elements[i].phi_, lagrange_interpolant_left);
        elements[i].phi_R_ = interpolate_to_boundary(elements[i].N_, elements[i].phi_, lagrange_interpolant_right);

        if (elements[i].phi_L_ > 0.0f) {
            elements[i].phi_L_ = elements[i].boundary_L_;
        }
        if (elements[i].phi_R_ < 0.0f) {
            elements[i].phi_R_ = elements[i].boundary_R_;
        }
        
        compute_dg_derivative(elements[i], weights, derivative_matrices, lagrange_interpolant_left, lagrange_interpolant_right); // Multiplied by -c in textbook, here multiplied by phi in matrix_vector_derivative
    }
}

int main(void) {
    const int N_elements = 1;
    const int initial_N = 8;
    const int N_max = 16;
    const int vector_length = (N_max + 1) * (N_max + 2)/2; // Flattened length of all N one after the other
    const int matrix_length = (N_max + 1) * (N_max + 2) * (2 * N_max + 3)/6; // Flattened length of all N² one after the other

    Element_t* elements;
    float* nodes;
    float* weights;
    float* barycentric_weights;
    float* lagrange_interpolant_left;
    float* lagrange_interpolant_right;
    float* derivative_matrices;
    float* derivative_matrices_hat;

    // Allocate GPU Memory – accessible from GPU
    cudaMalloc(&elements, N_elements * sizeof(Element_t));
    cudaMalloc(&nodes, vector_length * sizeof(float));
    cudaMalloc(&weights, vector_length * sizeof(float));
    cudaMalloc(&barycentric_weights, vector_length * sizeof(float));
    cudaMalloc(&lagrange_interpolant_left, vector_length * sizeof(float));
    cudaMalloc(&lagrange_interpolant_right, vector_length * sizeof(float));
    cudaMalloc(&derivative_matrices, matrix_length * sizeof(float));
    cudaMalloc(&derivative_matrices_hat, matrix_length * sizeof(float));

    auto t_start = std::chrono::high_resolution_clock::now(); 
    const int poly_blockSize = 16; // Small number of threads per block because N will never be huge
    for (int N = 0; N <= N_max; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        chebyshev_gauss_nodes_and_weights<<<vector_numBlocks, poly_blockSize>>>(N, nodes, weights);
    }

    const int elements_blockSize = 32; // For when we'll have multiple elements
    const int elements_numBlocks = (N_elements + elements_blockSize - 1) / elements_blockSize;
    build_elements<<<elements_numBlocks, elements_blockSize>>>(N_elements, initial_N, elements);

    // Nodes are needed to compute barycentric weights
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        calculate_barycentric_weights<<<vector_numBlocks, poly_blockSize>>>(N, nodes, barycentric_weights);
        lagrange_integrating_polynomials<<<vector_numBlocks, poly_blockSize>>>(-1.0f, N, nodes, weights, lagrange_interpolant_left);
        lagrange_integrating_polynomials<<<vector_numBlocks, poly_blockSize>>>(1.0f, N, nodes, weights, lagrange_interpolant_right);
    }

    initial_conditions<<<elements_numBlocks, elements_blockSize>>>(N_elements, elements, nodes);

    // We need to divide lagrange_integrating_polynomials by sum, and barycentric weights for derivative matrix
    cudaDeviceSynchronize();
    const int numBlocks = (N_max + poly_blockSize) / poly_blockSize;
    normalize_lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(N_max, lagrange_interpolant_left);
    normalize_lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(N_max, lagrange_interpolant_right);
    const dim3 matrix_blockSize(16, 16); // Small number of threads per block because N will never be huge
    for (int N = 0; N <= N_max; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices<<<matrix_numBlocks, matrix_blockSize>>>(N, nodes, barycentric_weights, derivative_matrices);
    }

    // Then we calculate the derivative matrix diagonal
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices_diagonal<<<vector_numBlocks, poly_blockSize>>>(N, derivative_matrices);
    }

    // All the derivative matrix has to be computed before D^
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices_hat<<<matrix_numBlocks, matrix_blockSize>>>(N, weights, derivative_matrices, derivative_matrices_hat);
    }

    // Starting actual computation
    cudaDeviceSynchronize();
    // This one right here officer
    compute_dg_time_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements, elements, weights, derivative_matrices_hat, lagrange_interpolant_left, lagrange_interpolant_right);
    
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 
            << "s." << std::endl;

    // Copy vectors from device memory to host memory
    float* host_nodes = new float[vector_length];
    float* host_weights = new float[vector_length];
    float* host_barycentric_weights = new float[vector_length];
    float* host_lagrange_interpolant_left = new float[vector_length];
    float* host_lagrange_interpolant_right = new float[vector_length];
    float* host_derivative_matrices = new float[matrix_length];
    float* host_derivative_matrices_hat = new float[matrix_length];

    cudaMemcpy(host_nodes, nodes, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights, weights, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights, barycentric_weights, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left, lagrange_interpolant_left, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right, lagrange_interpolant_right, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices, derivative_matrices, matrix_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices_hat, derivative_matrices_hat, matrix_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Can't do that!
    //cudaMemcpy(host_phi, elements[0].phi_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_phi_prime, elements[0].phi_prime_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);

    // CHECK find better solution for multiple elements
    float* phi;
    float* phi_prime;
    float* host_phi = new float[initial_N + 1];
    float* host_phi_prime = new float[initial_N + 1];
    cudaMalloc(&phi, (initial_N + 1) * sizeof(float));
    cudaMalloc(&phi_prime, (initial_N + 1) * sizeof(float));
    get_solution<<<elements_numBlocks, elements_blockSize>>>(N_elements, elements, phi, phi_prime);
    cudaDeviceSynchronize();
    cudaMemcpy(host_phi, phi, (initial_N + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime, phi_prime, (initial_N + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Nodes: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_nodes[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Weights: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl << "Barycentric weights: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_barycentric_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants -1: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_left[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants +1: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_right[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Derivative matrices: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
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
    for (int N = 0; N <= N_max; ++N) {
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

    std::cout << std::endl << "Phi: " << std::endl;
    for (int i = 0; i < N_elements; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N; ++j) {
            std::cout << host_phi[j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime: " << std::endl;
    for (int i = 0; i < N_elements; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N; ++j) {
            std::cout << host_phi_prime[j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(elements);
    cudaFree(nodes);
    cudaFree(weights);
    cudaFree(barycentric_weights);
    cudaFree(lagrange_interpolant_left);
    cudaFree(lagrange_interpolant_right);
    cudaFree(derivative_matrices);
    cudaFree(derivative_matrices_hat);
    cudaFree(phi);
    cudaFree(phi_prime);

    delete host_nodes;
    delete host_weights;
    delete host_barycentric_weights;
    delete host_lagrange_interpolant_left;
    delete host_lagrange_interpolant_right;
    delete host_derivative_matrices;
    delete host_derivative_matrices_hat;
    delete host_phi;
    delete host_phi_prime;
    
    return 0;
}