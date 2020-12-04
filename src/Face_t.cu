#include "Face_t.cuh"

__device__ 
Face_t::Face_t(int element_L, int element_R) : elements_{element_L, element_R}

__host__
Face_t::Face_t() {}

__host__ __device__
Face_t::~Face_t() {}