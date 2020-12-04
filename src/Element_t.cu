#include "Element_t.cuh"

__device__ 
Element_t::Element_t(int N, int neighbour_L, int neighbour_R, int face_L, int face_R, float x_L, float x_R) : 
        N_(N),
        neighbours_{neighbour_L, neighbour_R},
        faces_{face_L, face_R},
        x_{x_L, x_R},
        delta_x_(x_R - x_L) {
    phi_ = new float[N_ + 1];
    phi_prime_ = new float[N_ + 1];
    intermediate_ = new float[N_ + 1];
    for (int i = 0; i <= N_; ++i) {
        intermediate_[i] = 0.0f;
    }
}

__host__ 
Element_t::Element_t() {};

__host__ __device__
Element_t::~Element_t() {
    if (phi_ != nullptr){
        delete[] phi_;
    }
    if (phi_prime_ != nullptr) {
        delete[] phi_prime_;
    }
    if (intermediate_ != nullptr) {
        delete[] intermediate_;
    }
}