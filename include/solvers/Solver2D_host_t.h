#ifndef NDG_SOLVER2D_HOST_T_H
#define NDG_SOLVER2D_HOST_T_H

#include "helpers/float_types.h"
#include "entities/NDG_host_t.h"
#include "helpers/DataWriter_t.h"
#include "meshes/Mesh2D_host_t.h"
#include "entities/Face2D_host_t.h"
#include "entities/Element2D_host_t.h"
#include <vector>
#include <array>

namespace SEM { namespace Solvers {
    class Solver2D_host_t {
        public:
            Solver2D_host_t(hostFloat CFL, std::vector<hostFloat> output_times, hostFloat viscosity);

            hostFloat CFL_;
            hostFloat viscosity_;
            std::vector<hostFloat> output_times_;

            template<typename Polynomial>
            auto solve(const SEM::Entities::NDG_host_t<Polynomial> &NDG, SEM::Meshes::Mesh2D_host_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) const -> void;

            auto get_delta_t(SEM::Meshes::Mesh2D_host_t& mesh) const -> hostFloat;

            static auto x_flux(hostFloat p, hostFloat u, hostFloat v) -> std::array<hostFloat, 3>;

            static auto y_flux(hostFloat p, hostFloat u, hostFloat v) -> std::array<hostFloat, 3>;

            // Algorithm 19
            static auto matrix_vector_multiply(int N, const hostFloat* matrix, const hostFloat* vector, hostFloat* result) -> void;

            static auto calculate_wave_fluxes(std::vector<Face2D_host_t>& faces) -> void;
    };

    // Algorithm 114
    auto compute_dg_wave_derivative(size_t N_elements, SEM::Entities::Element2D_host_t* elements, const SEM::Entities::Face2D_host_t* faces, const hostFloat* weights, const hostFloat* derivative_matrices_hat, const hostFloat* lagrange_interpolant_left, const hostFloat* lagrange_interpolant_right) -> void;

    auto rk3_first_step(size_t N_elements, SEM::Entities::Element2D_host_t* elements, hostFloat delta_t, hostFloat g) -> void;

    auto rk3_step(size_t N_elements, SEM::Entities::Element2D_host_t* elements, hostFloat delta_t, hostFloat a, hostFloat g) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    auto warp_reduce_delta_t_2D(volatile hostFloat *sdata, unsigned int tid) -> void {
        if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
        if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
        if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
        if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
        if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
        if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
    }

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    auto reduce_wave_delta_t(hostFloat CFL, size_t N_elements, const SEM::Entities::Element2D_host_t* elements, hostFloat *g_odata) -> void {
        __shared__ hostFloat sdata[(blockSize >= 64) ? blockSize : blockSize + blockSize/2]; // Because within a warp there is no branching and this is read up until blockSize + blockSize/2
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = std::numeric_limits<hostFloat>::infinity();

        while (i < N_elements) { 
            hostFloat delta_t_wave = CFL * elements[i].delta_xy_min_/(elements[i].N_ * elements[i].N_);
    
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
