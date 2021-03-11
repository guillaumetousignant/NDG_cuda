#ifndef NDG_MESH_T_H
#define NDG_MESH_T_H

#include "Element_t.cuh"
#include "Face_t.cuh"
#include "NDG_t.cuh"
#include "float_types.h"
#include <vector>
#include <limits>
#include <mpi.h>
#include <array>

namespace SEM {
    class Mesh_t {
        public:
            Mesh_t(size_t N_elements, int initial_N, deviceFloat x_min, deviceFloat x_max, cudaStream_t &stream);
            ~Mesh_t();

            constexpr static int elements_blockSize_ = 32;
            constexpr static int faces_blockSize_ = 32; // Same number of faces as elements for periodic BC
            constexpr static int boundaries_blockSize_ = 32;
            int elements_numBlocks_;
            int faces_numBlocks_;
            int boundaries_numBlocks_;
            
            size_t N_elements_global_;
            size_t N_elements_;
            size_t N_faces_;
            size_t N_local_boundaries_;
            size_t N_MPI_boundaries_;
            size_t global_element_offset_;
            size_t N_elements_per_process_;
            int initial_N_;
            Element_t* elements_;
            Face_t* faces_;
            size_t* local_boundary_to_element_;
            size_t* MPI_boundary_to_element_;
            size_t* MPI_boundary_from_element_;

            void set_initial_conditions(const deviceFloat* nodes);
            void print();
            void write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices);
            deviceFloat get_delta_t(const deviceFloat CFL);
            
            template<typename Polynomial>
            void solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG, deviceFloat viscosity);

        private:
            deviceFloat* device_delta_t_array_;
            std::vector<deviceFloat> host_delta_t_array_;
            unsigned long* device_refine_array_;
            std::vector<unsigned long> host_refine_array_;
            deviceFloat* device_boundary_phi_L_;
            std::vector<deviceFloat> host_boundary_phi_L_;
            deviceFloat* device_boundary_phi_R_;
            std::vector<deviceFloat> host_boundary_phi_R_;
            deviceFloat* device_boundary_phi_prime_L_;
            std::vector<deviceFloat> host_boundary_phi_prime_L_;
            deviceFloat* device_boundary_phi_prime_R_;
            std::vector<deviceFloat> host_boundary_phi_prime_R_;
            std::vector<size_t> host_MPI_boundary_to_element_;
            std::vector<size_t> host_MPI_boundary_from_element_;
            cudaStream_t &stream_;

            std::vector<std::array<double, 4>> send_buffers_;
            std::vector<std::array<double, 4>> receive_buffers_;
            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;

            void write_file_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& coordinates, const std::vector<deviceFloat>& velocity, const std::vector<deviceFloat>& du_dx, const std::vector<deviceFloat>& intermediate, const std::vector<deviceFloat>& x_L, const std::vector<deviceFloat>& x_R, const std::vector<int>& N, const std::vector<deviceFloat>& sigma, const bool* refine, const bool* coarsen, const std::vector<deviceFloat>& error);
            void adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights);
            void boundary_conditions();
    };

    __global__
    void rk3_first_step(size_t N_elements, Element_t* elements, deviceFloat delta_t, deviceFloat g);

    __global__
    void rk3_step(size_t N_elements, Element_t* elements, deviceFloat delta_t, deviceFloat a, deviceFloat g);

    __global__
    void calculate_fluxes(size_t N_faces, Face_t* faces, const Element_t* elements);

    __device__
    void matrix_vector_multiply(int N, const deviceFloat* matrix, const deviceFloat* vector, deviceFloat* result);

    // Algorithm 19
    __device__
    void matrix_vector_derivative(deviceFloat viscosity, int N, const deviceFloat* derivative_matrices_hat, const deviceFloat* g_hat_derivative_matrices, const deviceFloat* phi, deviceFloat* phi_prime);

    // Algorithm 60 (not really anymore)
    __global__
    void compute_dg_derivative(deviceFloat viscosity, size_t N_elements, Element_t* elements, const Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* g_hat_derivative_matrices, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    void warp_reduce_delta_t(volatile deviceFloat *sdata, unsigned int tid) {
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
    void reduce_delta_t(deviceFloat CFL, size_t N_elements, const Element_t* elements, deviceFloat *g_odata) {
        __shared__ deviceFloat sdata[(blockSize >= 64) ? blockSize : blockSize + blockSize/2]; // Because within a warp there is no branching and this is read up until blockSize + blockSize/2
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = std::numeric_limits<deviceFloat>::infinity();

        while (i < N_elements) { 
            deviceFloat phi_max = 0.0;
            for (int j = 0; j <= elements[i].N_; ++j) {
                phi_max = max(phi_max, abs(elements[i].phi_[j]));
            }
            deviceFloat delta_t = CFL * elements[i].delta_x_ * elements[i].delta_x_/(phi_max * elements[i].N_ * elements[i].N_);
 
            if (i+blockSize < N_elements) {
                phi_max = 0.0;
                for (int j = 0; j <= elements[i+blockSize].N_; ++j) {
                    phi_max = max(phi_max, abs(elements[i+blockSize].phi_[j]));
                }
                delta_t = min(delta_t, CFL * elements[i+blockSize].delta_x_ * elements[i+blockSize].delta_x_/(phi_max * elements[i+blockSize].N_ * elements[i+blockSize].N_));
            }

            sdata[tid] = min(sdata[tid], delta_t); 
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

        if (tid < 32) warp_reduce_delta_t<blockSize>(sdata, tid);
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    void warp_reduce_refine(volatile unsigned long *sdata, unsigned int tid) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    void reduce_refine(size_t N_elements, const Element_t* elements, unsigned long *g_odata) {
        __shared__ unsigned long sdata[(blockSize >= 64) ? blockSize : blockSize + blockSize/2]; // Because within a warp there is no branching and this is read up until blockSize + blockSize/2
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = 0;

        while (i < N_elements) { 
            sdata[tid] += elements[i].refine_ * (elements[i].sigma_ < 1.0);
            if (i+blockSize < N_elements) {
                sdata[tid] += elements[i+blockSize].refine_ * (elements[i+blockSize].sigma_ < 1.0);
            }
            i += gridSize; 
        }
        __syncthreads();

        if (blockSize >= 8192) { if (tid < 4096) { sdata[tid] += sdata[tid + 4096]; } __syncthreads(); }
        if (blockSize >= 4096) { if (tid < 2048) { sdata[tid] += sdata[tid + 2048]; } __syncthreads(); }
        if (blockSize >= 2048) { if (tid < 1024) { sdata[tid] += sdata[tid + 1024]; } __syncthreads(); }
        if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
        if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

        if (tid < 32) warp_reduce_refine<blockSize>(sdata, tid);
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}

#endif