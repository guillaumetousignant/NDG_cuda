#ifndef NDG_MESH_T_H
#define NDG_MESH_T_H

#include "Element_t.cuh"
#include "Face_t.cuh"
#include "NDG_t.cuh"
#include "float_types.h"
#include <vector>

class Mesh_t {
public:
    Mesh_t(size_t N_elements, int initial_N, deviceFloat x_min, deviceFloat x_max);
    ~Mesh_t();

    size_t N_elements_;
    size_t N_faces_;
    int initial_N_;
    Element_t* elements_;
    Face_t* faces_;

    deviceFloat** phi_arrays_;
    deviceFloat** phi_prime_arrays_;
    deviceFloat** intermediate_arrays_;

    void set_initial_conditions(const deviceFloat* nodes);
    void print();
    void write_file_data(size_t N_points, deviceFloat time, const deviceFloat* coordinates, const deviceFloat* velocity, const deviceFloat* du_dx, const deviceFloat* intermediate, const deviceFloat* sigma, const deviceFloat* refine, const deviceFloat* coarsen, const deviceFloat* error);
    void write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices);
    
    template<typename Polynomial>
    void solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG);
};

namespace SEM {
    __global__
    void rk3_first_step(size_t N_elements, Element_t* elements, deviceFloat delta_t, deviceFloat g);

    __global__
    void rk3_step(size_t N_elements, Element_t* elements, deviceFloat delta_t, deviceFloat a, deviceFloat g);

    __global__
    void calculate_fluxes(size_t N_faces, Face_t* faces, const Element_t* elements);

    // Algorithm 19
    __device__
    void matrix_vector_derivative(int N, const deviceFloat* derivative_matrices_hat, const deviceFloat* phi, deviceFloat* phi_prime);

    // Algorithm 60 (not really anymore)
    __global__
    void compute_dg_derivative(size_t N_elements, Element_t* elements, const Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    void warp_reduce_velocity(volatile deviceFloat *sdata, unsigned int tid) {
        if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
        if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
        if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
        if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
        if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
        if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
    }

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    void reduce_velocity(size_t N_elements, const Element_t* elements, deviceFloat *g_odata) {
        extern __shared__ deviceFloat sdata[];
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = 0.0;

        while (i < N_elements) { 
            deviceFloat phi_max = 0.0;
            for (int j = 0; j <= elements[i].N_; ++j) {
                phi_max = max(phi_max, abs(elements[i].phi_[j]));
            }
            for (int j = 0; j <= elements[i+blockSize].N_; ++j) {
                phi_max = max(phi_max, abs(elements[i+blockSize].phi_[j]));
            }

            sdata[tid] = max(sdata[tid], phi_max); i += gridSize; 
        }
        __syncthreads();

        if (blockSize >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
        if (blockSize >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

        if (tid < 32) warp_reduce_velocity<blockSize>(sdata, tid);
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}

#endif