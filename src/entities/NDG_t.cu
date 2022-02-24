#include "entities/NDG_t.cuh"
#include <limits>
#include <cmath>

// Algorithm 30
__global__
void SEM::Device::Entities::calculate_barycentric_weights(int N, const deviceFloat* nodes, deviceFloat* barycentric_weights) {
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
bool SEM::Device::Entities::almost_equal(deviceFloat x, deviceFloat y) {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<deviceFloat>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<deviceFloat>::min(); 
}

// This will not work if we are on a node, or at least be pretty inefficient
// Algorithm 34
__global__
void SEM::Device::Entities::lagrange_interpolating_polynomials(deviceFloat x, int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_interpolant[offset + i] = barycentric_weights[offset + i] / (x - nodes[offset + i]);
    }
}

// Algorithm 34
__global__
void SEM::Device::Entities::normalize_lagrange_interpolating_polynomials(int N_max, deviceFloat* lagrange_interpolant) {
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
void SEM::Device::Entities::lagrange_interpolating_derivative_polynomials(deviceFloat x, int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_derivative_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_derivative_interpolant[offset + i] = barycentric_weights[offset + i] / ((x - nodes[offset + i]) * (x - nodes[offset + i]));
    }
}

// Algorithm 36
__global__
void SEM::Device::Entities::normalize_lagrange_interpolating_derivative_polynomials(deviceFloat x, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_derivative_interpolant) {
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
void SEM::Device::Entities::polynomial_derivative_matrices(int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* derivative_matrices) {
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
void SEM::Device::Entities::polynomial_derivative_matrices_diagonal(int N, deviceFloat* derivative_matrices) {
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
void SEM::Device::Entities::polynomial_cg_derivative_matrices(int N, const deviceFloat* weights, const deviceFloat* derivative_matrices, deviceFloat* g_hat_derivative_matrices) {
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
void SEM::Device::Entities::polynomial_derivative_matrices_hat(int N, const deviceFloat* weights, const deviceFloat* derivative_matrices, deviceFloat* derivative_matrices_hat) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const size_t offset_1D = N * (N + 1) /2;
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            derivative_matrices_hat[offset_2D + i * (N + 1) + j] = -derivative_matrices[offset_2D + j * (N + 1) + i] * weights[offset_1D + j] / weights[offset_1D + i];
        }
    }
}

// Will interpolate n_interpolation_points between -1 and 1
__global__
void SEM::Device::Entities::create_interpolation_matrices(int N, size_t n_interpolation_points, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* interpolation_matrices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset_1D = N * (N + 1) /2;
    const size_t offset_interp = N * (N + 1) * n_interpolation_points/2;

    for (size_t j = index; j < n_interpolation_points; j += stride) {
        bool row_has_match = false;
        const deviceFloat x_coord = 2.0f * j / (n_interpolation_points - 1) - 1.0f;

        for (int k = 0; k <= N; ++k) {
            interpolation_matrices[offset_interp + j * (N + 1) + k] = 0.0f;
            if (SEM::Device::Entities::almost_equal(x_coord, nodes[offset_1D + k])) {
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

__global__
void SEM::Device::Entities::calculate_polynomials(int N, const deviceFloat* nodes, const deviceFloat* polynomials) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const size_t offset_1D = N * (N + 1) /2;
    const size_t offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            polynomials[offset_2D + i * (N + 1) + j] = polynomial(i, polynomial_nodes[offset_1D + j]);
        }
    }
}
