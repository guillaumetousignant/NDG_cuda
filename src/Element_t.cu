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
        intermediate_(intermediate_array),
        sigma_(0),
        refine_(false),
        coarsen_(false) {}

__host__ 
Element_t::Element_t() {};

__host__ __device__
Element_t::~Element_t() {}

// Algorithm 61
__device__
void Element_t::interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) {
    const int offset_1D = N_ * (N_ + 1) /2;
    phi_L_ = 0.0;
    phi_R_ = 0.0;

    for (int j = 0; j <= N_; ++j) {
        phi_L_ += lagrange_interpolant_left[offset_1D + j] * phi_[j];
        phi_R_ += lagrange_interpolant_right[offset_1D + j] * phi_[j];
    }
}

// Algorithm 22
__device__
void SEM::polynomial_and_derivative(int N, deviceFloat x, deviceFloat &L_N, deviceFloat &L_N_prime) {
    if (N == 0) {
        L_N = 1.0f;
        L_N_prime = 0.0f;
    }
    else if (N == 1) {
        L_N = x;
        L_N_prime = 1.0f;
    }
    else {
        deviceFloat L_N_2 = 1.0f;
        deviceFloat L_N_1 = x;
        deviceFloat L_N_2_prime = 0.0f;
        deviceFloat L_N_1_prime = 1.0f;

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

__device__
void Element_t::estimate_error(const deviceFloat* nodes, const deviceFloat* weights) {
    const int offset_1D = N_ * (N_ + 1) /2;
    /*deviceFloat p = N_;
    
    for (int node = N_; node >= 0; --node) {
        intermediate_[node] = 0.0;

        for (int i = 0; i <= p; ++i) {
            deficeFloat ap = 0.0;

            for (int j = 0; j <= N_; ++j) {
                deviceFloat L_N, dummy;
                SEM::legendre_polynomial_and_derivative(i, nodes[offset_1D + j], L_N, dummy);
                
                ap += (2.0 * i + 1.0) * (2.0 * p + 1.0) * 0.25 * phi_[j] * L_N * weights[offset_1D + j];
            }

            intermediate_[node] += std::abs(ap);
        }

        --p;
    }*/

    for (int k = 0; k <= N_; ++k) {
        intermediate_[k] = 0.0;
        for (int i = 0; i <= N_; ++i) {
            deviceFloat L_N, dummy;
            SEM::polynomial_and_derivative(k, nodes[offset_1D + i], L_N, dummy);

            intermediate_[k] += (2 * k + 1) * 0.5 * phi_[i] * weights[offset_1D + i];
        }
    }

    constexpr deviceFloat tolerance_min = 1e-6;     // Refine above this
    constexpr deviceFloat tolerance_max = 1e-14;    // Coarsen below this

    const deviceFloat C = exponential_decay();

    // sum of error
    const deviceFloat error = std::sqrt(N_ * N_
        + C * C / (2 * sigma_) * std::exp(-2 * sigma_ * (N_ + 1)));

    if(error > tolerance_min){	// need refine
        refine_ = true;
        coarsen_ = false;
    }
    else if(error <= tolerance_max ){	// need coarsen
        refine_ = false;
        coarsen_ = true;
    }
    else{	// if error in between then do nothing
        refine_ = false;
        coarsen_ = false;
    }
}

__device__
deviceFloat Element_t::exponential_decay() {
    constexpr int n_points_least_squares = 4; // Number of points to use for thew least squares reduction

    deviceFloat x_avg = 0.0;
    deviceFloat y_avg = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        x_avg += N_ - i;
        y_avg += std::log(std::abs(intermediate_[N_ - i]));
    }

    x_avg /= n_points_least_squares;
    y_avg /= n_points_least_squares;

    deviceFloat numerator = 0.0;
    deviceFloat denominator = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        numerator += (N_ - i - x_avg) * (std::log(std::abs(intermediate_[N_ - i])) - y_avg);
        denominator += std::pow((N_ - i - x_avg), 2);
    }

    sigma_ = numerator/denominator;

    const deviceFloat C = std::exp(y_avg - sigma_ * x_avg);
    sigma_ = std::abs(sigma_);
    return C;
}

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

__global__
void SEM::estimate_error(size_t N_elements, Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].estimate_error(nodes, weights);
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
void SEM::get_solution(size_t N_elements, size_t N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* phi, deviceFloat* phi_prime, deviceFloat* intermediate, deviceFloat* sigma, deviceFloat* refine, deviceFloat* coarsen) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t offset_interp_1D = i * N_interpolation_points;
        const size_t offset_interp = elements[i].N_ * (elements[i].N_ + 1) * N_interpolation_points/2;

        for (size_t j = 0; j < N_interpolation_points; ++j) {
            phi[offset_interp_1D + j] = 0.0f;
            phi_prime[offset_interp_1D + j] = 0.0f;
            intermediate[offset_interp_1D + j] = 0.0f;
            for (int k = 0; k <= elements[i].N_; ++k) {
                phi[offset_interp_1D + j] += interpolation_matrices[offset_interp + j * (elements[i].N_ + 1) + k] * elements[i].phi_[k];
                phi_prime[offset_interp_1D + j] += interpolation_matrices[offset_interp + j * (elements[i].N_ + 1) + k] * elements[i].phi_prime_[k];
                intermediate[offset_interp_1D + j] += interpolation_matrices[offset_interp + j * (elements[i].N_ + 1) + k] * elements[i].intermediate_[k];
            }
            x[offset_interp_1D + j] = j * (elements[i].x_[1] - elements[i].x_[0]) / (N_interpolation_points - 1) + elements[i].x_[0];
            sigma[offset_interp_1D + j] = elements[i].sigma_;
            refine[offset_interp_1D + j] = elements[i].refine_;
            coarsen[offset_interp_1D + j] = elements[i].coarsen_;
        }
    }
}

__global__
void SEM::interpolate_to_boundaries(size_t N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].interpolate_to_boundaries(lagrange_interpolant_left, lagrange_interpolant_right);
    }
}