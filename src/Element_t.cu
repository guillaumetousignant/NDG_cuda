#include "Element_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include <cmath>
#include <thrust/swap.h>

constexpr deviceFloat pi = 3.14159265358979323846;

__device__ 
Element_t::Element_t(int N, size_t neighbour_L, size_t neighbour_R, size_t face_L, size_t face_R, deviceFloat x_L, deviceFloat x_R) : 
        N_(N),
        neighbours_{neighbour_L, neighbour_R},
        faces_{face_L, face_R},
        x_{x_L, x_R},
        delta_x_(x_[1] - x_[0]),
        phi_(new deviceFloat[N_ + 1]),
        phi_prime_(new deviceFloat[N_ + 1]),
        intermediate_(new deviceFloat[N_ + 1]),
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {}

__device__
Element_t::Element_t(const Element_t& other) :
        N_(other.N_),
        neighbours_{other.neighbours_[0], other.neighbours_[1]},
        faces_{other.faces_[0], other.faces_[1]},
        x_{other.x_[0], other.x_[1]},
        delta_x_(other.delta_x_),
        phi_(new deviceFloat[N_ + 1]),
        phi_prime_(new deviceFloat[N_ + 1]),
        intermediate_(new deviceFloat[N_ + 1]),
        sigma_(other.sigma_),
        refine_(other.refine_),
        coarsen_(other.coarsen_),
        error_(other.error_) {

    for (int i = 0; i <= N_; ++i) {
        phi_[i] = other.phi_[i];
        phi_prime_[i] = other.phi_prime_[i];
        intermediate_[i] = other.intermediate_[i];
    }
}

__device__
Element_t::Element_t(Element_t&& other) :
        N_(other.N_),
        neighbours_{other.neighbours_[0], other.neighbours_[1]},
        faces_{other.faces_[0], other.faces_[1]},
        x_{other.x_[0], other.x_[1]},
        delta_x_(other.delta_x_),
        phi_(other.phi_),
        phi_prime_(other.phi_prime_),
        intermediate_(other.intermediate_),
        sigma_(other.sigma_),
        refine_(other.refine_),
        coarsen_(other.coarsen_),
        error_(other.error_) {
    
    other.phi_ = nullptr;
    other.phi_prime_ = nullptr;
    other.intermediate_ = nullptr;
}

__device__
Element_t& Element_t::operator=(const Element_t& other) {
    if (N_ != other.N_) {
        delete [] phi_;
        delete [] phi_prime_;
        delete [] intermediate_;

        phi_ = new deviceFloat[other.N_];
        phi_prime_ = new deviceFloat[other.N_];
        intermediate_ = new deviceFloat[other.N_];
    }

    N_ = other.N_;
    neighbours_[0] = other.neighbours_[0];
    neighbours_[1] = other.neighbours_[1];
    faces_[0] = other.faces_[0];
    faces_[1] = other.faces_[1];
    x_[0] = other.x_[0];
    x_[1] = other.x_[1];
    delta_x_ = other.delta_x_;
    sigma_ = other.sigma_;
    refine_ = other.refine_;
    coarsen_ = other.coarsen_;
    error_ = other.error_;

    for (int i = 0; i <= N_; ++i) {
        phi_[i] = other.phi_[i];
        phi_prime_[i] = other.phi_prime_[i];
        intermediate_[i] = other.intermediate_[i];
    }

    return *this;
}

__device__
Element_t& Element_t::operator=(Element_t&& other) {
    N_ = other.N_;
    neighbours_[0] = other.neighbours_[0];
    neighbours_[1] = other.neighbours_[1];
    faces_[0] = other.faces_[0];
    faces_[1] = other.faces_[1];
    x_[0] = other.x_[0];
    x_[1] = other.x_[1];
    delta_x_ = other.delta_x_;
    sigma_ = other.sigma_;
    refine_ = other.refine_;
    coarsen_ = other.coarsen_;
    error_ = other.error_;

    thrust::swap(phi_, other.phi_);
    thrust::swap(phi_prime_, other.phi_prime_);
    thrust::swap(intermediate_, other.intermediate_);

    return *this;
}

__host__ __device__
Element_t::Element_t() :
        N_(0),
        neighbours_{0, 0},
        faces_{0, 0},
        x_{0.0, 0.0},
        delta_x_(0.0),
        phi_(nullptr),
        phi_prime_(nullptr),
        intermediate_(nullptr),
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {};

__host__ __device__
Element_t::~Element_t() {
    delete [] phi_;
    delete [] phi_prime_;
    delete [] intermediate_;
}

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

// This should be illegal, but is needed because I can't for the life of me get separable compilation to work correctly.
__device__
void ChebyshevPolynomial_t::polynomial(int N, deviceFloat x, deviceFloat &T_N) {
    T_N = cos(N * acos(x));
}

__device__
void LegendrePolynomial_t::polynomial(int N, deviceFloat x, deviceFloat &L_N) {
    if (N == 0) {
        L_N = 1.0f;
    }
    else if (N == 1) {
        L_N = x;
    }
    else {
        deviceFloat L_N_2 = 1.0f;
        deviceFloat L_N_1 = x;
        deviceFloat L_N_2_prime = 0.0f;
        deviceFloat L_N_1_prime = 1.0f;

        for (int k = 2; k <= N; ++k) {
            L_N = (2 * k - 1) * x * L_N_1/k - (k - 1) * L_N_2/k; // L_N_1(x) ??
            const deviceFloat L_N_prime = L_N_2_prime + (2 * k - 1) * L_N_1;
            L_N_2 = L_N_1;
            L_N_1 = L_N;
            L_N_2_prime = L_N_1_prime;
            L_N_1_prime = L_N_prime;
        }
    }
}

template __device__ void Element_t::estimate_error<ChebyshevPolynomial_t>(const deviceFloat* nodes, const deviceFloat* weights);
template __device__ void Element_t::estimate_error<LegendrePolynomial_t>(const deviceFloat* nodes, const deviceFloat* weights);

template<typename Polynomial>
__device__
void Element_t::estimate_error<Polynomial>(const deviceFloat* nodes, const deviceFloat* weights) {
    const int offset_1D = N_ * (N_ + 1) /2;

    for (int k = 0; k <= N_; ++k) {
        intermediate_[k] = 0.0;
        for (int i = 0; i <= N_; ++i) {
            deviceFloat L_N;
            Polynomial::polynomial(k, nodes[offset_1D + i], L_N);

            intermediate_[k] += (2 * k + 1) * 0.5 * phi_[i] * L_N * weights[offset_1D + i];
        }
    }

    constexpr deviceFloat tolerance_min = 1e-6;     // Refine above this
    constexpr deviceFloat tolerance_max = 1e-14;    // Coarsen below this

    const deviceFloat C = exponential_decay();

    // sum of error
    error_ = std::sqrt(std::pow(C, 2) * 0.5/sigma_) * std::exp(-sigma_ * (N_ + 1));

    if(error_ > tolerance_min){	// need refine
        refine_ = true;
        coarsen_ = false;
    }
    else if(error_ <= tolerance_max ){	// need coarsen
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
    const int n_points_least_squares = min(N_, 4); // Number of points to use for thew least squares reduction, but don't go above N.

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
void SEM::build_elements(size_t N_elements, int N, Element_t* elements, deviceFloat x_min, deviceFloat x_max) {
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
        elements[i] = Element_t(N, neighbour_L, neighbour_R, face_L, face_R, element_x_min, element_y_min);
    }
}

template __global__ void SEM::estimate_error<ChebyshevPolynomial_t>(size_t N_elements, Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights);
template __global__ void SEM::estimate_error<LegendrePolynomial_t>(size_t N_elements, Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights);

template<typename Polynomial>
__global__
void SEM::estimate_error<Polynomial>(size_t N_elements, Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].estimate_error<Polynomial>(nodes, weights);
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
void SEM::get_solution(size_t N_elements, size_t N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* phi, deviceFloat* phi_prime, deviceFloat* intermediate, deviceFloat* sigma, deviceFloat* refine, deviceFloat* coarsen, deviceFloat* error) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        const size_t offset_interp_1D = i * N_interpolation_points;
        const size_t offset_interp = elements[i].N_ * (elements[i].N_ + 1) * N_interpolation_points/2;
        const size_t step = N_interpolation_points/(elements[i].N_ + 1);

        for (size_t j = 0; j < N_interpolation_points; ++j) {
            phi[offset_interp_1D + j] = 0.0f;
            phi_prime[offset_interp_1D + j] = 0.0f;
            for (int k = 0; k <= elements[i].N_; ++k) {
                phi[offset_interp_1D + j] += interpolation_matrices[offset_interp + j * (elements[i].N_ + 1) + k] * elements[i].phi_[k];
                phi_prime[offset_interp_1D + j] += interpolation_matrices[offset_interp + j * (elements[i].N_ + 1) + k] * elements[i].phi_prime_[k]; 
            }
            intermediate[offset_interp_1D + j] = elements[i].intermediate_[min(static_cast<int>(j/step), elements[i].N_)];
            x[offset_interp_1D + j] = j * (elements[i].x_[1] - elements[i].x_[0]) / (N_interpolation_points - 1) + elements[i].x_[0];
            sigma[offset_interp_1D + j] = elements[i].sigma_;
            refine[offset_interp_1D + j] = elements[i].refine_;
            coarsen[offset_interp_1D + j] = elements[i].coarsen_;
            error[offset_interp_1D + j] = elements[i].error_;
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