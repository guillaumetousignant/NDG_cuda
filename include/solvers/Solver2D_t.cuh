#ifndef NDG_SOLVER2D_T_H
#define NDG_SOLVER2D_T_H

#include "helpers/float_types.h"
#include "entities/NDG_t.cuh"
#include "helpers/DataWriter_t.h"
#include "meshes/Mesh2D_t.cuh"
#include "entities/Face2D_t.cuh"
#include "entities/Element2D_t.cuh"
#include <vector>
#include <array>

namespace SEM { namespace Solvers {
    class Solver2D_t {
        public:
            Solver2D_t(deviceFloat CFL, std::vector<deviceFloat> output_times, deviceFloat viscosity);

            deviceFloat CFL_;
            deviceFloat viscosity_;
            std::vector<deviceFloat> output_times_;

            template<typename Polynomial>
            auto solve(const SEM::Entities::NDG_t<Polynomial> &NDG, SEM::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) -> void;

            auto get_delta_t(SEM::Meshes::Mesh2D_t& mesh) -> deviceFloat;

            __host__ __device__
            static auto x_flux(deviceFloat p, deviceFloat u, deviceFloat v) -> std::array<deviceFloat, 3>;

            __host__ __device__
            static auto y_flux(deviceFloat p, deviceFloat u, deviceFloat v) -> std::array<deviceFloat, 3>;
    };

    __global__
    void calculate_wave_fluxes(size_t N_faces, SEM::Entities::Face2D_t* faces, const SEM::Entities::Element2D_t* elements);

    // Algorithm 60 (not really anymore)
    __global__
    void compute_dg_wave_derivative(size_t N_elements, SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    void warp_reduce_delta_t_2D(volatile deviceFloat *sdata, unsigned int tid) {
        if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
        if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
        if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
        if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
        if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
        if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
    }

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    void reduce_wave_delta_t(deviceFloat CFL, size_t N_elements, const SEM::Entities::Element2D_t* elements, deviceFloat *g_odata) {
        __shared__ deviceFloat sdata[(blockSize >= 64) ? blockSize : blockSize + blockSize/2]; // Because within a warp there is no branching and this is read up until blockSize + blockSize/2
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = std::numeric_limits<deviceFloat>::infinity();

        while (i < N_elements) { 
            deviceFloat delta_t_wave = CFL * elements[i].delta_xy_min_/(elements[i].N_ * elements[i].N_);
    
            if (i+blockSize < N_elements) {
                delta_t_wave = min(delta_t_wave, CFL * elements[i+blockSize].delta_xy_min_/(elements[i+blockSize].N_ * elements[i+blockSize].N_));
            }

            sdata[tid] = min(sdata[tid], delta_t_wave); 
            i += gridSize; 
        }
        __syncthreads();

        if (blockSize >= 8192) { if (tid < 4096) { sdata[tid] = min(sdata[tid], sdata[tid + 4096]); } __syncthreads(); }
        if (blockSize >= 4096) { if (tid < 2048) { sdata[tid] = min(sdata[tid], sdata[tid + 2048]); } __syncthreads(); }
        if (blockSize >= 2048) { if (tid < 1024) { sdata[tid] = min(sdata[tid], sdata[tid + 1024]); } __syncthreads(); }
        if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
        if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
        if (blockSize >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

        if (tid < 32) warp_reduce_delta_t_2D<blockSize>(sdata, tid);
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}}

#endif
