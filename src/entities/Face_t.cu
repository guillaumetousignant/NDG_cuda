#include "entities/Face_t.cuh"
#include <utility>

__host__ __device__ 
SEM::Entities::Face_t::Face_t(size_t element_L, size_t element_R) : 
        elements_{element_L, element_R},
        flux_{0.0},
        derivative_flux_{0.0},
        nl_flux_{0.0} {};

__host__ __device__
SEM::Entities::Face_t::Face_t() :
        elements_{0, 0},
        flux_{0.0},
        derivative_flux_{0.0},
        nl_flux_{0.0} {}

__global__
void SEM::Entities::build_faces(size_t N_faces, SEM::Entities::Face_t* faces) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_faces; i += stride) {
        const size_t neighbour_L = (i > 0) ? i - 1 : N_faces - 1;
        const size_t neighbour_R = (i < N_faces - 1) ? i : N_faces;
        faces[i] = SEM::Entities::Face_t(neighbour_L, neighbour_R);
    }
}
