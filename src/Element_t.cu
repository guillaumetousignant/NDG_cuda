#include "Element_t.cuh"
#include <cmath>

constexpr deviceFloat pi = 3.14159265358979323846;

__device__ 
Element_t::Element_t(int N, size_t neighbour_L, size_t neighbour_R, size_t face_L, size_t face_R, deviceFloat x_L, deviceFloat x_R, deviceFloat* phi_array, deviceFloat* phi_prime_array, deviceFloat* intermediate_array) : 
        N_(N),
        neighbours_{neighbour_L, neighbour_R},
        faces_{face_L, face_R},
        x_{x_L, x_R},
        delta_x_(x_R - x_L),
        phi_(phi_array),
        phi_prime_(phi_prime_array),
        intermediate_(intermediate_array) {}

__host__ 
Element_t::Element_t() {};

__host__ __device__
Element_t::~Element_t() {}

__global__
void SEM::build_elements(size_t N_elements, int N, Element_t* elements, deviceFloat x_min, deviceFloat x_max, deviceFloat** phi_arrays, deviceFloat** phi_prime_arrays, deviceFloat** intermediate_arrays) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const size_t neighbour_L = (i > 0) ? i - 1 : N_elements - 1; // First cell has last cell as left neighbour
        const size_t neighbour_R = (i < N_elements - 1) ? i + 1 : 0; // Last cell has first cell as right neighbour
        const size_t face_L = (i > 0) ? i - 1 : N_elements - 1;
        const size_t face_R = i;
        const deviceFloat delta_x = (x_max - x_min)/N_elements;
        const deviceFloat element_x_min = x_min + i * delta_x;
        const deviceFloat element_y_min = x_min + (i + 1) * delta_x;
        elements[i] = Element_t(N, neighbour_L, neighbour_R, face_L, face_R, element_x_min, element_y_min, phi_arrays[i], phi_prime_arrays[i], intermediate_arrays[i]);
    }
}

__device__
deviceFloat SEM::g(deviceFloat x) {
    //return (x < -0.2f || x > 0.2f) ? 0.2f : 0.8f;
    return -std::sin(pi * x);
}


__global__
void SEM::initial_conditions(size_t N_elements, Element_t* elements, const deviceFloat* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t offset = elements[i].N_ * (elements[i].N_ + 1) /2;
        for (int j = 0; j <= elements[i].N_; ++j) {
            const deviceFloat x = (0.5 + nodes[offset + j]/2.0f) * (elements[i].x_[1] - elements[i].x_[0]) + elements[i].x_[0];
            elements[i].phi_[j] = SEM::g(x);
        }
    }
}

// Basically useless, find better solution when multiple elements.
__global__
void SEM::get_elements_data(size_t N_elements, const Element_t* elements, deviceFloat* phi, deviceFloat* phi_prime) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t element_offset = i * (elements[i].N_ + 1);
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[element_offset + j] = elements[i].phi_[j];
            phi_prime[element_offset + j] = elements[i].phi_prime_[j];
        }
    }
}

// Basically useless, find better solution when multiple elements.
__global__
void SEM::get_phi(size_t N_elements, const Element_t* elements, deviceFloat* phi) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
        }
    }
}

__global__
void SEM::get_solution(size_t N_elements, size_t N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* phi, deviceFloat* x) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t offset_interp_1D = i * N_interpolation_points;
        const size_t offset_interp = elements[i].N_ * (elements[i].N_ + 1) * N_interpolation_points/2;

        for (size_t j = 0; j < N_interpolation_points; ++j) {
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
deviceFloat SEM::interpolate_to_boundary(int N, const deviceFloat* phi, const deviceFloat* lagrange_interpolant) {
    const int offset_1D = N * (N + 1) /2;
    deviceFloat result = 0.0;

    for (int j = 0; j <= N; ++j) {
        result += lagrange_interpolant[offset_1D + j] * phi[j];
    }

    return result;
}

__global__
void SEM::interpolate_to_boundaries(size_t N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].phi_L_ = SEM::interpolate_to_boundary(elements[i].N_, elements[i].phi_, lagrange_interpolant_left);
        elements[i].phi_R_ = SEM::interpolate_to_boundary(elements[i].N_, elements[i].phi_, lagrange_interpolant_right);
    }
}