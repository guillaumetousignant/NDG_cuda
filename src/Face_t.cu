#include "Face_t.cuh"
#include <utility>

__device__ 
SEM::Face_t::Face_t(size_t element_L, size_t element_R) : elements_{element_L, element_R} {};

__device__
SEM::Face_t::Face_t(const SEM::Face_t& other) :
        elements_{other.elements_[0], other.elements_[1]},
        flux_{other.flux_},
        derivative_flux_{other.derivative_flux_} {}

__device__
SEM::Face_t::Face_t(SEM::Face_t&& other) :
        elements_{other.elements_[0], other.elements_[1]},
        flux_{other.flux_},
        derivative_flux_{other.derivative_flux_} {}

__device__
SEM::Face_t& SEM::Face_t::operator=(const SEM::Face_t& other) {
    elements_[0] = other.elements_[0];
    elements_[1] = other.elements_[1];
    flux_ = other.flux_;
    derivative_flux_ = other.derivative_flux_;

    return *this;
}

__device__
SEM::Face_t& SEM::Face_t::operator=(SEM::Face_t&& other) {
    elements_[0] = other.elements_[0];
    elements_[1] = other.elements_[1];
    flux_ = other.flux_;
    derivative_flux_ = other.derivative_flux_;

    return *this;
}

__host__
SEM::Face_t::Face_t() {}

__host__ __device__
SEM::Face_t::~Face_t() {}

__global__
void SEM::build_faces(size_t N_faces, SEM::Face_t* faces) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_faces; i += stride) {
        const size_t neighbour_L = (i > 0) ? i - 1 : N_faces - 1;
        const size_t neighbour_R = (i < N_faces - 1) ? i : N_faces; // Last face links last element to first element
        faces[i] = SEM::Face_t(neighbour_L, neighbour_R);
    }
}

__global__
void SEM::copy_faces(size_t N_faces, const SEM::Face_t* faces, SEM::Face_t* new_faces) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_faces; i += stride) {
        new_faces[i] = std::move(faces[i]);
    }
}