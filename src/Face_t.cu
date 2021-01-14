#include "Face_t.cuh"

__device__ 
Face_t::Face_t(size_t element_L, size_t element_R) : elements_{element_L, element_R} {};

__host__
Face_t::Face_t() {}

__host__ __device__
Face_t::~Face_t() {}

__global__
void SEM::build_faces(size_t N_faces, Face_t* faces) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_faces; i += stride) {
        const size_t neighbour_L = i;
        const size_t neighbour_R = (i < N_faces - 1) ? i + 1 : 0; // Last face links last element to first element
        faces[i] = Face_t(neighbour_L, neighbour_R);
    }
}