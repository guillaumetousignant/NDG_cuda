#include "NDG_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>

constexpr int poly_blockSize = 16; // Small number of threads per block because N will never be huge
constexpr int interpolation_blockSize = 32;
const dim3 matrix_blockSize(16, 16); // Small number of threads per block because N will never be huge

template class NDG_t<ChebyshevPolynomial_t>; // Like, I understand why I need this, but man is it crap.
template class NDG_t<LegendrePolynomial_t>;

template<typename Polynomial>
NDG_t<Polynomial>::NDG_t(int N_max, size_t N_interpolation_points) : 
        N_max_(N_max), 
        N_interpolation_points_(N_interpolation_points),
        vector_length_((N_max_ + 1) * (N_max_ + 2)/2), 
        matrix_length_((N_max_ + 1) * (N_max_ + 2) * (2 * N_max_ + 3)/6),
        interpolation_length_((N_max_ + 1) * (N_max_ + 2) * N_interpolation_points_/2) {

    cudaMalloc(&nodes_, vector_length_ * sizeof(deviceFloat));
    cudaMalloc(&weights_, vector_length_ * sizeof(deviceFloat));
    cudaMalloc(&barycentric_weights_, vector_length_ * sizeof(deviceFloat));
    cudaMalloc(&lagrange_interpolant_left_, vector_length_ * sizeof(deviceFloat));
    cudaMalloc(&lagrange_interpolant_right_, vector_length_ * sizeof(deviceFloat));
    cudaMalloc(&lagrange_interpolant_derivative_left_, vector_length_ * sizeof(deviceFloat));
    cudaMalloc(&lagrange_interpolant_derivative_right_, vector_length_ * sizeof(deviceFloat));
    cudaMalloc(&derivative_matrices_, matrix_length_ * sizeof(deviceFloat));
    cudaMalloc(&g_hat_derivative_matrices_, matrix_length_ * sizeof(deviceFloat));
    cudaMalloc(&derivative_matrices_hat_, matrix_length_ * sizeof(deviceFloat));
    cudaMalloc(&interpolation_matrices_, interpolation_length_ * sizeof(deviceFloat));

    Polynomial::nodes_and_weights(N_max_, poly_blockSize, nodes_, weights_);

    // Nodes are needed to compute barycentric weights
    for (int N = 0; N <= N_max_; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::calculate_barycentric_weights<<<vector_numBlocks, poly_blockSize>>>(N, nodes_, barycentric_weights_);
    }

    // We need the barycentric weights for derivative matrix, interpolation matrices and Lagrange interpolants
    const int interpolation_numBlocks = (N_interpolation_points_ + interpolation_blockSize) / interpolation_blockSize;
    for (int N = 0; N <= N_max_; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::polynomial_derivative_matrices<<<matrix_numBlocks, matrix_blockSize>>>(N, nodes_, barycentric_weights_, derivative_matrices_);
        SEM::create_interpolation_matrices<<<interpolation_numBlocks, interpolation_blockSize>>>(N, N_interpolation_points_, nodes_, barycentric_weights_, interpolation_matrices_);
        SEM::lagrange_interpolating_polynomials<<<vector_numBlocks, poly_blockSize>>>(-1.0f, N, nodes_, barycentric_weights_, lagrange_interpolant_left_);
        SEM::lagrange_interpolating_polynomials<<<vector_numBlocks, poly_blockSize>>>(1.0f, N, nodes_, barycentric_weights_, lagrange_interpolant_right_);
        SEM::lagrange_interpolating_derivative_polynomials<<<vector_numBlocks, poly_blockSize>>>(-1.0f, N, nodes_, barycentric_weights_, lagrange_interpolant_derivative_left_);
        SEM::lagrange_interpolating_derivative_polynomials<<<vector_numBlocks, poly_blockSize>>>(1.0f, N, nodes_, barycentric_weights_, lagrange_interpolant_derivative_right_);
    }

    // Then we calculate the derivative matrix diagonal and normalize the Lagrange interpolants
    const int poly_numBlocks = (N_max_ + poly_blockSize) / poly_blockSize;
    SEM::normalize_lagrange_interpolating_polynomials<<<poly_numBlocks, poly_blockSize>>>(N_max_, lagrange_interpolant_left_);
    SEM::normalize_lagrange_interpolating_polynomials<<<poly_numBlocks, poly_blockSize>>>(N_max_, lagrange_interpolant_right_);
    SEM::normalize_lagrange_interpolating_derivative_polynomials<<<poly_numBlocks, poly_blockSize>>>(-1.0f, N_max_, nodes_, barycentric_weights_, lagrange_interpolant_derivative_left_);
    SEM::normalize_lagrange_interpolating_derivative_polynomials<<<poly_numBlocks, poly_blockSize>>>(1.0f, N_max_, nodes_, barycentric_weights_, lagrange_interpolant_derivative_right_);
    for (int N = 0; N <= N_max_; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::polynomial_derivative_matrices_diagonal<<<vector_numBlocks, poly_blockSize>>>(N, derivative_matrices_);
    }

    // All the derivative matrix has to be computed before D^
    for (int N = 0; N <= N_max_; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::polynomial_cg_derivative_matrices<<<matrix_numBlocks, matrix_blockSize>>>(N, weights_, derivative_matrices_, g_hat_derivative_matrices_);
        SEM::polynomial_derivative_matrices_hat<<<matrix_numBlocks, matrix_blockSize>>>(N, weights_, derivative_matrices_, derivative_matrices_hat_);
    }
}

template<typename Polynomial>
NDG_t<Polynomial>::~NDG_t() {
    cudaFree(nodes_);
    cudaFree(weights_);
    cudaFree(barycentric_weights_);
    cudaFree(lagrange_interpolant_left_);
    cudaFree(lagrange_interpolant_right_);
    cudaFree(lagrange_interpolant_derivative_left_);
    cudaFree(lagrange_interpolant_derivative_right_);
    cudaFree(derivative_matrices_);
    cudaFree(g_hat_derivative_matrices_);
    cudaFree(derivative_matrices_hat_);
    cudaFree(interpolation_matrices_);
}
   
template<typename Polynomial>
void NDG_t<Polynomial>::print() {
    // Copy vectors from device memory to host memory
    deviceFloat* host_nodes = new deviceFloat[vector_length_];
    deviceFloat* host_weights = new deviceFloat[vector_length_];
    deviceFloat* host_barycentric_weights = new deviceFloat[vector_length_];
    deviceFloat* host_lagrange_interpolant_left = new deviceFloat[vector_length_];
    deviceFloat* host_lagrange_interpolant_right = new deviceFloat[vector_length_];
    deviceFloat* host_lagrange_interpolant_derivative_left = new deviceFloat[vector_length_];
    deviceFloat* host_lagrange_interpolant_derivative_right = new deviceFloat[vector_length_];
    deviceFloat* host_derivative_matrices = new deviceFloat[matrix_length_];
    deviceFloat* host_g_hat_derivative_matrices = new deviceFloat[matrix_length_];
    deviceFloat* host_derivative_matrices_hat = new deviceFloat[matrix_length_];
    deviceFloat* host_interpolation_matrices = new deviceFloat[interpolation_length_];

    cudaMemcpy(host_nodes, nodes_, vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights, weights_, vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights, barycentric_weights_, vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left, lagrange_interpolant_left_, vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right, lagrange_interpolant_right_, vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_derivative_left, lagrange_interpolant_derivative_left_, vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_derivative_right, lagrange_interpolant_derivative_right_, vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices, derivative_matrices_, matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_g_hat_derivative_matrices, g_hat_derivative_matrices_, matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices_hat, derivative_matrices_hat_, matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_interpolation_matrices, interpolation_matrices_, interpolation_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    std::cout << "Nodes: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_nodes[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Weights: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl << "Barycentric weights: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_barycentric_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants -1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_left[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants +1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_right[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants derivatives -1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_derivative_left[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants derivatives +1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_derivative_right[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Derivative matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << host_derivative_matrices[offset_2D + i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "CG derivative matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << host_g_hat_derivative_matrices[offset_2D + i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "Derivative matrices hat: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << host_derivative_matrices_hat[offset_2D + i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "Interpolation matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        const size_t offset_interp = N * (N + 1) * N_interpolation_points_/2;

        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i < N_interpolation_points_; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << host_interpolation_matrices[offset_interp + i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    delete[] host_nodes;
    delete[] host_weights;
    delete[] host_barycentric_weights;
    delete[] host_lagrange_interpolant_left;
    delete[] host_lagrange_interpolant_right;
    delete[] host_lagrange_interpolant_derivative_left;
    delete[] host_lagrange_interpolant_derivative_right;
    delete[] host_derivative_matrices;
    delete[] host_g_hat_derivative_matrices;
    delete[] host_derivative_matrices_hat;
    delete[] host_interpolation_matrices;
}

// Algorithm 30
__global__
void SEM::calculate_barycentric_weights(int N, const deviceFloat* nodes, deviceFloat* barycentric_weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int j = index; j <= N; j += stride) {
        deviceFloat xjxi = 1.0;
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
bool SEM::almost_equal(deviceFloat x, deviceFloat y) {
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
void SEM::lagrange_interpolating_polynomials(deviceFloat x, int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_interpolant[offset + i] = barycentric_weights[offset + i] / (x - nodes[offset + i]);
    }
}

// Algorithm 34
__global__
void SEM::normalize_lagrange_interpolating_polynomials(int N_max, deviceFloat* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int N = index; N <= N_max; N += stride) {
        const size_t offset = N * (N + 1) /2;
        deviceFloat sum = 0.0;
        for (int i = 0; i <= N; ++i) {
            sum += lagrange_interpolant[offset + i];
        }
        for (int i = 0; i <= N; ++i) {
            lagrange_interpolant[offset + i] /= sum;
        }
    }
}

// This will not work if we are on a node, or at least be pretty inefficient
// Algorithm 36
__global__
void SEM::lagrange_interpolating_derivative_polynomials(deviceFloat x, int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_derivative_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_derivative_interpolant[offset + i] = barycentric_weights[offset + i] / ((x - nodes[offset + i]) * (x - nodes[offset + i]));
    }
}

// Algorithm 36
__global__
void SEM::normalize_lagrange_interpolating_derivative_polynomials(deviceFloat x, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_derivative_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int N = index; N <= N_max; N += stride) {
        const size_t offset = N * (N + 1) /2;
        deviceFloat sum = 0.0;
        for (int i = 0; i <= N; ++i) {
            sum += barycentric_weights[offset + i]/(x - nodes[offset + i]);
        }
        for (int i = 0; i <= N; ++i) {
            lagrange_derivative_interpolant[offset + i] /= sum;
        }
    }
}

// Be sure to compute the diagonal afterwards
// Algorithm 37
__global__
void SEM::polynomial_derivative_matrices(int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* derivative_matrices) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const size_t offset_1D = N * (N + 1) /2;
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

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
void SEM::polynomial_derivative_matrices_diagonal(int N, deviceFloat* derivative_matrices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

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

// Algorithm 57
__global__
void SEM::polynomial_cg_derivative_matrices(int N, const deviceFloat* weights, const deviceFloat* derivative_matrices, deviceFloat* g_hat_derivative_matrices) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const size_t offset_1D = N * (N + 1) /2;
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int j = index_x; j <= N; j += stride_x) {
        for (int n = index_y; n <= N; n += stride_y) {
            deviceFloat s = 0.0;
            for (int k = 0; k <= N; ++k) {
                s += derivative_matrices[offset_2D + k * (N + 1) + n] * derivative_matrices[offset_2D + k * (N + 1) + j] * weights[offset_1D + k];
            }
            g_hat_derivative_matrices[offset_2D + j * (N + 1) + n] = s/weights[offset_1D + j];
        }
    }
}

__global__
void SEM::polynomial_derivative_matrices_hat(int N, const deviceFloat* weights, const deviceFloat* derivative_matrices, deviceFloat* derivative_matrices_hat) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const size_t offset_1D = N * (N + 1) /2;
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            derivative_matrices_hat[offset_2D + i * (N + 1) + j] = derivative_matrices[offset_2D + j * (N + 1) + i] * weights[offset_1D + j] / weights[offset_1D + i];
        }
    }
}

// Will interpolate N_interpolation_points between -1 and 1
__global__
void SEM::create_interpolation_matrices(int N, size_t N_interpolation_points, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* interpolation_matrices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset_1D = N * (N + 1) /2;
    const size_t offset_interp = N * (N + 1) * N_interpolation_points/2;

    for (size_t j = index; j < N_interpolation_points; j += stride) {
        bool row_has_match = false;
        const deviceFloat x_coord = 2.0f * j / (N_interpolation_points - 1) - 1.0f;

        for (int k = 0; k <= N; ++k) {
            interpolation_matrices[offset_interp + j * (N + 1) + k] = 0.0f;
            if (SEM::almost_equal(x_coord, nodes[offset_1D + k])) {
                interpolation_matrices[offset_interp + j * (N + 1) + k] = 1.0f;
                row_has_match = true;
            }
        }

        if (!row_has_match) {
            deviceFloat total = 0.0f;
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