#include "Element_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include <cmath>
#include <thrust/swap.h>

constexpr deviceFloat pi = 3.14159265358979323846;

__device__ 
SEM::Element_t::Element_t(int N, size_t face_L, size_t face_R, deviceFloat x_L, deviceFloat x_R) : 
        N_(N),
        faces_{face_L, face_R},
        x_{x_L, x_R},
        delta_x_(x_[1] - x_[0]),
        phi_(new deviceFloat[N_ + 1]),
        q_(new deviceFloat[N_ + 1]),
        ux_(new deviceFloat[N_ + 1]),
        phi_prime_(new deviceFloat[N_ + 1]),
        intermediate_(new deviceFloat[N_ + 1]),
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {}

__device__
SEM::Element_t::Element_t(const SEM::Element_t& other) :
        N_(other.N_),
        faces_{other.faces_[0], other.faces_[1]},
        x_{other.x_[0], other.x_[1]},
        delta_x_(other.delta_x_),
        phi_(new deviceFloat[N_ + 1]),
        q_(new deviceFloat[N_ + 1]),
        ux_(new deviceFloat[N_ + 1]),
        phi_prime_(new deviceFloat[N_ + 1]),
        intermediate_(new deviceFloat[N_ + 1]),
        sigma_(other.sigma_),
        refine_(other.refine_),
        coarsen_(other.coarsen_),
        error_(other.error_) {

    for (int i = 0; i <= N_; ++i) {
        phi_[i] = other.phi_[i];
        q_[i] = other.q_[i];
        ux_[i] = other.ux_[i];
        phi_prime_[i] = other.phi_prime_[i];
        intermediate_[i] = other.intermediate_[i];
    }
}

__device__
SEM::Element_t::Element_t(SEM::Element_t&& other) :
        N_(other.N_),
        faces_{other.faces_[0], other.faces_[1]},
        x_{other.x_[0], other.x_[1]},
        delta_x_(other.delta_x_),
        phi_(other.phi_),
        q_(other.q_),
        ux_(other.ux_),
        phi_prime_(other.phi_prime_),
        intermediate_(other.intermediate_),
        sigma_(other.sigma_),
        refine_(other.refine_),
        coarsen_(other.coarsen_),
        error_(other.error_) {
    
    other.phi_ = nullptr;
    other.q_ = nullptr;
    other.ux_ = nullptr;
    other.phi_prime_ = nullptr;
    other.intermediate_ = nullptr;
}

__device__
SEM::Element_t& SEM::Element_t::operator=(const SEM::Element_t& other) {
    if (N_ != other.N_) {
        delete[] phi_;
        delete[] q_;
        delete[] ux_;
        delete[] phi_prime_;
        delete[] intermediate_;

        N_ = other.N_;
        phi_ = new deviceFloat[N_];
        q_ = new deviceFloat[N_];
        ux_ = new deviceFloat[N_];
        phi_prime_ = new deviceFloat[N_];
        intermediate_ = new deviceFloat[N_];
    }

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
        q_[i] = other.q_[i];
        ux_[i] = other.ux_[i];
        phi_prime_[i] = other.phi_prime_[i];
        intermediate_[i] = other.intermediate_[i];
    }
    
    return *this;
}

__device__
SEM::Element_t& SEM::Element_t::operator=(SEM::Element_t&& other) {
    N_ = other.N_;
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
    thrust::swap(q_, other.q_);
    thrust::swap(ux_, other.ux_);
    thrust::swap(phi_prime_, other.phi_prime_);
    thrust::swap(intermediate_, other.intermediate_);
    
    return *this;
}

__host__ __device__
SEM::Element_t::Element_t() :
        N_(0),
        faces_{0, 0},
        x_{0.0, 0.0},
        delta_x_(0.0),
        phi_(nullptr),
        q_(nullptr),
        ux_(nullptr),
        phi_prime_(nullptr),
        intermediate_(nullptr),
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {};

__host__ __device__
SEM::Element_t::~Element_t() {
    delete [] phi_;
    delete [] q_;
    delete [] ux_;
    delete [] phi_prime_;
    delete [] intermediate_;
}

// Algorithm 61
__device__
void SEM::Element_t::interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right, const deviceFloat* lagrange_interpolant_derivative_left, const deviceFloat* lagrange_interpolant_derivative_right) {
    const int offset_1D = N_ * (N_ + 1) /2;
    phi_L_ = 0.0;
    phi_R_ = 0.0;

    for (int j = 0; j <= N_; ++j) {
        phi_L_ += lagrange_interpolant_left[offset_1D + j] * phi_[j];
        phi_R_ += lagrange_interpolant_right[offset_1D + j] * phi_[j];
    }
}

// Algorithm 61
__device__
void SEM::Element_t::interpolate_q_to_boundaries(const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right, const deviceFloat* lagrange_interpolant_derivative_left, const deviceFloat* lagrange_interpolant_derivative_right) {
    const int offset_1D = N_ * (N_ + 1) /2;
    phi_prime_L_ = 0.0;
    phi_prime_R_ = 0.0;

    for (int j = 0; j <= N_; ++j) {
        phi_prime_L_ += lagrange_interpolant_left[offset_1D + j] * q_[j];
        phi_prime_R_ += lagrange_interpolant_right[offset_1D + j] * q_[j];
    }
}

template __device__ void SEM::Element_t::estimate_error<SEM::ChebyshevPolynomial_t>(const deviceFloat* nodes, const deviceFloat* weights);
template __device__ void SEM::Element_t::estimate_error<SEM::LegendrePolynomial_t>(const deviceFloat* nodes, const deviceFloat* weights);

template<typename Polynomial>
__device__
void SEM::Element_t::estimate_error<Polynomial>(const deviceFloat* nodes, const deviceFloat* weights) {
    const int offset_1D = N_ * (N_ + 1) /2;

    for (int k = 0; k <= N_; ++k) {
        intermediate_[k] = 0.0;
        for (int i = 0; i <= N_; ++i) {
            const deviceFloat L_N = Polynomial::polynomial(k, nodes[offset_1D + i]);

            intermediate_[k] += (2 * k + 1) * 0.5 * phi_[i] * L_N * weights[offset_1D + i];
        }
        intermediate_[k] = std::abs(intermediate_[k]);
    }

    constexpr deviceFloat tolerance_min = 1e-6;     // Refine above this
    constexpr deviceFloat tolerance_max = 1e-14;    // Coarsen below this

    const deviceFloat C = exponential_decay();

    // sum of error
    error_ = std::sqrt(C * C * 0.5/sigma_) * std::exp(-sigma_ * (N_ + 1));

    if(error_ > tolerance_min) {	// need refine
        refine_ = true;
        coarsen_ = false;
    }
    else if(error_ <= tolerance_max ) {	// need coarsen
        refine_ = false;
        coarsen_ = true;
    }
    else {	// if error in between then do nothing
        refine_ = false;
        coarsen_ = false;
    }
}

__device__
deviceFloat SEM::Element_t::exponential_decay() {
    const int n_points_least_squares = min(N_, 4); // Number of points to use for thew least squares reduction, but don't go above N.

    deviceFloat x_avg = 0.0;
    deviceFloat y_avg = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        x_avg += N_ - i;
        y_avg += std::log(intermediate_[N_ - i]);
    }

    x_avg /= n_points_least_squares;
    y_avg /= n_points_least_squares;

    deviceFloat numerator = 0.0;
    deviceFloat denominator = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        numerator += (N_ - i - x_avg) * (std::log(intermediate_[N_ - i]) - y_avg);
        denominator += (N_ - i - x_avg) * (N_ - i - x_avg);
    }

    sigma_ = numerator/denominator;

    const deviceFloat C = std::exp(y_avg - sigma_ * x_avg);
    sigma_ = std::abs(sigma_);
    return C;
}

__device__
void SEM::Element_t::interpolate_from(const SEM::Element_t& other, const deviceFloat* nodes, const deviceFloat* barycentric_weights) {
    const int offset = N_ * (N_ + 1) /2;
    const int offset_other = other.N_ * (other.N_ + 1) /2;

    for (int i = 0; i <= N_; ++i) {
        const deviceFloat x = (x_[1] - x_[0]) * (nodes[offset + i] + 1) * 0.5 + x_[0];
        const deviceFloat node = (2 * x - other.x_[0] - other.x_[1])/(other.x_[1] - other.x_[0]);
        deviceFloat numerator = 0.0;
        deviceFloat denominator = 0.0;
        for (int j = 0; j <= other.N_; ++j) {
            if (SEM::almost_equal2(node, nodes[offset_other + j])) {
                numerator = other.phi_[j];
                denominator = 1.0;
                break;
            }
            const deviceFloat t = barycentric_weights[offset_other + j]/(node - nodes[offset_other + j]);
            numerator += t * other.phi_[j];
            denominator += t;
        }
        phi_[i] = numerator/denominator;
    }
}

__global__
void SEM::build_elements(size_t N_elements, int N, SEM::Element_t* elements, deviceFloat x_min, deviceFloat x_max) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const size_t face_L = i;
        const size_t face_R = i + 1;
        const deviceFloat delta_x = (x_max - x_min)/N_elements;
        const deviceFloat element_x_min = x_min + i * delta_x;
        const deviceFloat element_x_max = x_min + (i + 1) * delta_x;

        // Those are uninitialised because they are created via cudaMalloc, so they need to be set if we don't want the move constructor to delete random memory.
        elements[i].phi_ = nullptr;
        elements[i].q_ = nullptr;
        elements[i].ux_ = nullptr;
        elements[i].phi_prime_ = nullptr;
        elements[i].intermediate_ = nullptr;

        elements[i] = SEM::Element_t(N, face_L, face_R, element_x_min, element_x_max);
    }
}

__global__
void SEM::build_boundaries(size_t N_elements, size_t N_elements_global, size_t N_local_boundaries, size_t N_MPI_boundaries, Element_t* elements, size_t global_element_offset, size_t* local_boundary_to_element, size_t* MPI_boundary_to_element, size_t* MPI_boundary_from_element) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_local_boundaries; i += stride) {
        size_t face_L;
        size_t face_R;
        deviceFloat element_x_min;
        deviceFloat element_x_max;

        if (i == 0) { // CHECK this is hardcoded for 1D
            face_L = 0;
            face_R = 0;
            element_x_min = elements[0].x_[0];
            element_x_max = elements[0].x_[0];
            local_boundary_to_element[i] = N_elements - 1;
        }
        else if (i == 1) {
            face_L = N_elements + N_local_boundaries + N_MPI_boundaries - 2;
            face_R = N_elements + N_local_boundaries + N_MPI_boundaries - 2;
            element_x_min = elements[N_elements - 1].x_[1];
            element_x_max = elements[N_elements - 1].x_[1];
            local_boundary_to_element[i] = 0;
        }

        // Those are uninitialised because they are created via cudaMalloc, so they need to be set if we don't want the move constructor to delete random memory.
        elements[N_elements + i].phi_ = nullptr;
        elements[N_elements + i].q_ = nullptr;
        elements[N_elements + i].ux_ = nullptr;
        elements[N_elements + i].phi_prime_ = nullptr;
        elements[N_elements + i].intermediate_ = nullptr;

        elements[N_elements + i] = SEM::Element_t(0, face_L, face_R, element_x_min, element_x_max);
    }

    for (int i = index; i < N_MPI_boundaries; i += stride) {
        size_t face_L;
        size_t face_R;
        deviceFloat element_x_min;
        deviceFloat element_x_max;

        if (i == 0) { // CHECK this is hardcoded for 1D
            face_L = 0;
            face_R = 0;
            element_x_min = elements[0].x_[0];
            element_x_max = elements[0].x_[0];
            MPI_boundary_to_element[i] = (global_element_offset == 0) ? N_elements_global - 1 : global_element_offset - 1;
            MPI_boundary_from_element[i] = global_element_offset;
        }
        else if (i == 1) {
            face_L = N_elements + N_local_boundaries + N_MPI_boundaries - 2;
            face_R = N_elements + N_local_boundaries + N_MPI_boundaries - 2;
            element_x_min = elements[N_elements - 1].x_[1];
            element_x_max = elements[N_elements - 1].x_[1];
            MPI_boundary_to_element[i] = (global_element_offset + N_elements == N_elements_global) ? 0 : global_element_offset + N_elements;
            MPI_boundary_from_element[i] = global_element_offset + N_elements - 1;
        }

        // Those are uninitialised because they are created via cudaMalloc, so they need to be set if we don't want the move constructor to delete random memory.
        elements[N_elements + N_local_boundaries + i].phi_ = nullptr;
        elements[N_elements + N_local_boundaries + i].q_ = nullptr;
        elements[N_elements + N_local_boundaries + i].ux_ = nullptr;
        elements[N_elements + N_local_boundaries + i].phi_prime_ = nullptr;
        elements[N_elements + N_local_boundaries + i].intermediate_ = nullptr;

        elements[N_elements + N_local_boundaries + i] = SEM::Element_t(0, face_L, face_R, element_x_min, element_x_max);
    }
}

__global__
void SEM::copy_boundaries(size_t N_elements, size_t N_elements_global, size_t N_local_boundaries, size_t N_MPI_boundaries, size_t n_additional_elements, Element_t* elements, Element_t* new_elements, Face_t* new_faces, size_t global_element_offset, size_t* local_boundary_to_element, size_t* MPI_boundary_to_element, size_t* MPI_boundary_from_element) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_local_boundaries; i += stride) {
        new_elements[N_elements + n_additional_elements + i].phi_ = nullptr;
        new_elements[N_elements + n_additional_elements + i].q_ = nullptr;
        new_elements[N_elements + n_additional_elements + i].ux_ = nullptr;
        new_elements[N_elements + n_additional_elements + i].phi_prime_ = nullptr;
        new_elements[N_elements + n_additional_elements + i].intermediate_ = nullptr;

        new_elements[N_elements + n_additional_elements + i] = std::move(elements[N_elements + i]);
        const size_t right_element = new_faces[new_elements[N_elements + n_additional_elements + i].faces_[0]].elements_[1];
        new_faces[new_elements[N_elements + n_additional_elements + i].faces_[0]].elements_[right_element == N_elements + i] = N_elements + n_additional_elements + i;

        if (i == 0) { // CHECK this is hardcoded for 1D
            local_boundary_to_element[i] = N_elements + n_additional_elements - 1;
        }
    }

    for (int i = index; i < N_MPI_boundaries; i += stride) {
        new_elements[N_elements + n_additional_elements + N_local_boundaries + i].phi_ = nullptr;
        new_elements[N_elements + n_additional_elements + N_local_boundaries + i].q_ = nullptr;
        new_elements[N_elements + n_additional_elements + N_local_boundaries + i].ux_ = nullptr;
        new_elements[N_elements + n_additional_elements + N_local_boundaries + i].phi_prime_ = nullptr;
        new_elements[N_elements + n_additional_elements + N_local_boundaries + i].intermediate_ = nullptr;

        new_elements[N_elements + n_additional_elements + N_local_boundaries + i] = std::move(elements[N_elements + N_local_boundaries + i]);
        const size_t right_element = new_faces[new_elements[N_elements + n_additional_elements + N_local_boundaries + i].faces_[0]].elements_[1];
        new_faces[new_elements[N_elements + n_additional_elements + N_local_boundaries + i].faces_[0]].elements_[right_element == N_elements + N_local_boundaries + i] = N_elements + n_additional_elements + N_local_boundaries + i;
        
        if (i == 0) { // CHECK this is hardcoded for 1D
            MPI_boundary_to_element[i] = (global_element_offset == 0) ? N_elements_global - 1 : global_element_offset - 1;
            MPI_boundary_from_element[i] = global_element_offset;
        }
        else if (i == 1) {
            MPI_boundary_to_element[i] = (global_element_offset + N_elements + n_additional_elements == N_elements_global) ? 0 : global_element_offset + N_elements + n_additional_elements;
            MPI_boundary_from_element[i] = global_element_offset + N_elements + n_additional_elements - 1;
        }
    }
}

__global__
void SEM::free_elements(size_t N_elements, SEM::Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        delete[] elements[i].phi_;
        delete[] elements[i].q_;
        delete[] elements[i].ux_;
        delete[] elements[i].phi_prime_;
        delete[] elements[i].intermediate_;
        elements[i].phi_ = nullptr;
        elements[i].q_ = nullptr;
        elements[i].ux_ = nullptr;
        elements[i].phi_prime_ = nullptr;
        elements[i].intermediate_ = nullptr;
    }
}

template __global__ void SEM::estimate_error<SEM::ChebyshevPolynomial_t>(size_t N_elements, SEM::Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights);
template __global__ void SEM::estimate_error<SEM::LegendrePolynomial_t>(size_t N_elements, SEM::Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights);

template<typename Polynomial>
__global__
void SEM::estimate_error<Polynomial>(size_t N_elements, SEM::Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].estimate_error<Polynomial>(nodes, weights);
    }
}

__host__ __device__
deviceFloat SEM::g(deviceFloat x) {
    //return (x < -0.2f || x > 0.2f) ? 0.2f : 0.8f;
    return -std::sin(pi * x);
}

__host__ __device__
deviceFloat SEM::g_prime(deviceFloat x) {
    //return (x < -0.2f || x > 0.2f) ? 0.2f : 0.8f;
    return -pi * std::cos(pi * x);
}

__global__
void SEM::initial_conditions(size_t N_elements, SEM::Element_t* elements, const deviceFloat* nodes) {
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
void SEM::get_elements_data(size_t N_elements, const SEM::Element_t* elements, deviceFloat* phi, deviceFloat* phi_prime) {
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

__global__
void SEM::get_phi(size_t N_elements, const SEM::Element_t* elements, deviceFloat** phi) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[i][j] = elements[i].phi_[j];
        }
    }
}

__global__
void SEM::put_phi(size_t N_elements, SEM::Element_t* elements, deviceFloat** phi) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].phi_ = new deviceFloat[elements[i].N_ + 1];
        elements[i].q_ = new deviceFloat[elements[i].N_ + 1];
        elements[i].ux_ = new deviceFloat[elements[i].N_ + 1];
        elements[i].phi_prime_ = new deviceFloat[elements[i].N_ + 1];
        elements[i].intermediate_ = new deviceFloat[elements[i].N_ + 1];

        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_[j] = phi[i][j];
        }
    }
}

__global__
void SEM::move_elements(size_t N_elements, Element_t* elements, Element_t* new_elements, size_t source_start_index, size_t destination_start_index) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        new_elements[i + destination_start_index].phi_ = nullptr;
        new_elements[i + destination_start_index].q_ = nullptr;
        new_elements[i + destination_start_index].ux_ = nullptr;
        new_elements[i + destination_start_index].phi_prime_ = nullptr;
        new_elements[i + destination_start_index].intermediate_ = nullptr;

        new_elements[i + destination_start_index] = std::move(elements[i + source_start_index]);

        new_elements[i + destination_start_index].faces_[0] = new_elements[i + destination_start_index].faces_[0] + destination_start_index - source_start_index;
        new_elements[i + destination_start_index].faces_[1] = new_elements[i + destination_start_index].faces_[1] + destination_start_index - source_start_index;
    }
}

__global__
void SEM::get_solution(size_t N_elements, size_t N_interpolation_points, const SEM::Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* phi, deviceFloat* phi_prime, deviceFloat* intermediate, deviceFloat* x_L, deviceFloat* x_R, int* N, deviceFloat* sigma, bool* refine, bool* coarsen, deviceFloat* error) {
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
        }

        x_L[i] = elements[i].x_[0];
        x_R[i] = elements[i].x_[1];
        N[i] = elements[i].N_;
        sigma[i] = elements[i].sigma_;
        refine[i] = elements[i].refine_;
        coarsen[i] = elements[i].coarsen_;
        error[i] = elements[i].error_;
    }
}

__global__
void SEM::interpolate_to_boundaries(size_t N_elements, SEM::Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right, const deviceFloat* lagrange_interpolant_derivative_left, const deviceFloat* lagrange_interpolant_derivative_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].interpolate_to_boundaries(lagrange_interpolant_left, lagrange_interpolant_right, lagrange_interpolant_derivative_left, lagrange_interpolant_derivative_right);
    }
}

__global__
void SEM::interpolate_q_to_boundaries(size_t N_elements, SEM::Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right, const deviceFloat* lagrange_interpolant_derivative_left, const deviceFloat* lagrange_interpolant_derivative_right) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].interpolate_q_to_boundaries(lagrange_interpolant_left, lagrange_interpolant_right, lagrange_interpolant_derivative_left, lagrange_interpolant_derivative_right);
    }
}

__global__
void SEM::hp_adapt(unsigned long N_elements, SEM::Element_t* elements, SEM::Element_t* new_elements, const unsigned long* block_offsets, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) {
    const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long stride = blockDim.x * gridDim.x;
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    
    for (unsigned long i = index; i < N_elements; i += stride) {
        unsigned long element_index = i + block_offsets[block_id];
        for (unsigned long j = i - thread_id; j < i; ++j) {
            element_index += elements[j].refine_ * (elements[j].sigma_ < 1.0);
        }

        if (elements[i].refine_ && elements[i].sigma_ < 1.0) {            
            // Those are uninitialised because they are created via cudaMalloc, so they need to be set if we don't want the move constructor to delete random memory.
            new_elements[element_index].phi_ = nullptr;
            new_elements[element_index].q_ = nullptr;
            new_elements[element_index].ux_ = nullptr;
            new_elements[element_index].phi_prime_ = nullptr;
            new_elements[element_index].intermediate_ = nullptr;
            new_elements[element_index + 1].phi_ = nullptr;
            new_elements[element_index + 1].q_ = nullptr;
            new_elements[element_index + 1].ux_ = nullptr;
            new_elements[element_index + 1].phi_prime_ = nullptr;
            new_elements[element_index + 1].intermediate_ = nullptr;

            new_elements[element_index] = SEM::Element_t(elements[i].N_, element_index, element_index + 1, elements[i].x_[0], (elements[i].x_[0] + elements[i].x_[1]) * 0.5);
            new_elements[element_index + 1] = SEM::Element_t(elements[i].N_, element_index + 1,  element_index + 2, (elements[i].x_[0] + elements[i].x_[1]) * 0.5, elements[i].x_[1]);
            new_elements[element_index].interpolate_from(elements[i], nodes, barycentric_weights);
            new_elements[element_index + 1].interpolate_from(elements[i], nodes, barycentric_weights);
        }
        else if (elements[i].refine_ && elements[i].N_ < N_max) {
            new_elements[element_index].phi_ = nullptr;
            new_elements[element_index].q_ = nullptr;
            new_elements[element_index].ux_ = nullptr;
            new_elements[element_index].phi_prime_ = nullptr;
            new_elements[element_index].intermediate_ = nullptr;

            new_elements[element_index] = SEM::Element_t(min(elements[i].N_ + 2, N_max), element_index, element_index + 1, elements[i].x_[0], elements[i].x_[1]);
            new_elements[element_index].interpolate_from(elements[i], nodes, barycentric_weights);
        }
        else {
            // Those are uninitialised because they are created via cudaMalloc, so they need to be set if we don't want the move constructor to delete random memory.
            new_elements[element_index].phi_ = nullptr;
            new_elements[element_index].q_ = nullptr;
            new_elements[element_index].ux_ = nullptr;
            new_elements[element_index].phi_prime_ = nullptr;
            new_elements[element_index].intermediate_ = nullptr;
            
            new_elements[element_index] = std::move(elements[i]);
            new_elements[element_index].faces_ = {element_index, element_index + 1};
        }
    }
}

__global__
void SEM::p_adapt(unsigned long N_elements, SEM::Element_t* elements, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) {
    const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long stride = blockDim.x * gridDim.x;
    
    for (unsigned long i = index; i < N_elements; i += stride) {
        if (elements[i].refine_ && elements[i].sigma_ >= 1.0 && elements[i].N_ < N_max) {
            SEM::Element_t new_element(min(elements[i].N_ + 2, N_max), elements[i].faces_[0], elements[i].faces_[1], elements[i].x_[0], elements[i].x_[1]);
            new_element.interpolate_from(elements[i], nodes, barycentric_weights);
            elements[i] = std::move(new_element);
        }
    }
}

// From cppreference.com
__device__
bool SEM::almost_equal2(deviceFloat x, deviceFloat y) {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= FLT_EPSILON * std::abs(x+y) * ulp // CHECK change this to double equivalent if using double instead of float
        // unless the result is subnormal
        || std::abs(x-y) < FLT_MIN; // CHECK change this to 64F if using double instead of float
}

__global__
void SEM::local_boundaries(size_t N_elements, size_t N_local_boundaries, Element_t* elements, const size_t* local_boundary_to_element) {
    const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long stride = blockDim.x * gridDim.x;
    
    for (unsigned long i = index; i < N_local_boundaries; i += stride) {
        elements[N_elements + i].phi_L_ = elements[local_boundary_to_element[i]].phi_L_;
        elements[N_elements + i].phi_R_ = elements[local_boundary_to_element[i]].phi_R_;
        elements[N_elements + i].phi_prime_L_ = elements[local_boundary_to_element[i]].phi_prime_L_;
        elements[N_elements + i].phi_prime_R_ = elements[local_boundary_to_element[i]].phi_prime_R_;
    }
}

__global__
void SEM::get_MPI_boundaries(size_t N_elements, size_t N_local_boundaries, size_t N_MPI_boundaries, const Element_t* elements, const Face_t* faces, deviceFloat* phi_L, deviceFloat* phi_R, deviceFloat* phi_prime_L, deviceFloat* phi_prime_R) {
    const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long stride = blockDim.x * gridDim.x;
    
    for (unsigned long i = index; i < N_MPI_boundaries; i += stride) {
        const Element_t& boundary_element = elements[N_elements + N_local_boundaries + i];
        const Face_t& boundary_face = faces[boundary_element.faces_[0]];
        const Element_t& domain_element = elements[boundary_face.elements_[boundary_face.elements_[0] == N_elements + N_local_boundaries + i]];
        phi_L[i] = domain_element.phi_L_;
        phi_R[i] = domain_element.phi_R_;
        phi_prime_L[i] = domain_element.phi_prime_L_;
        phi_prime_R[i] = domain_element.phi_prime_R_;
    }
}

__global__
void SEM::put_MPI_boundaries(size_t N_elements, size_t N_local_boundaries, size_t N_MPI_boundaries, Element_t* elements, const deviceFloat* phi_L, const deviceFloat* phi_R, const deviceFloat* phi_prime_L, const deviceFloat* phi_prime_R) {
    const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long stride = blockDim.x * gridDim.x;
    
    for (unsigned long i = index; i < N_MPI_boundaries; i += stride) {
        elements[N_elements + N_local_boundaries + i].phi_L_ = phi_L[i];
        elements[N_elements + N_local_boundaries + i].phi_R_ = phi_R[i];
        elements[N_elements + N_local_boundaries + i].phi_prime_L_ = phi_prime_L[i];
        elements[N_elements + N_local_boundaries + i].phi_prime_R_ = phi_prime_R[i];
    }
}
