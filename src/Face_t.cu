#include "Face_t.cuh"

__device__ 
Face_t::Face_t(int element_L, int element_R) : elements_{element_L, element_R} {};

__host__
Face_t::Face_t() {}

__host__ __device__
Face_t::~Face_t() {}

__global__
void SEM::build_faces(int N_faces, Face_t* faces) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_faces; i += stride) {
        const int neighbour_L = i;
        const int neighbour_R = (i < N_faces - 1) ? i + 1 : 0; // Last face links last element to first element
        faces[i] = Face_t(neighbour_L, neighbour_R);
    }
}