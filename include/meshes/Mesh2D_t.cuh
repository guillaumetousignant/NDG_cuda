#ifndef NDG_MESH2D_T_H
#define NDG_MESH2D_T_H

#include "entities/Element2D_t.cuh"
#include "entities/Face2D_t.cuh"
#include "entities/NDG_t.cuh"
#include "entities/device_vector.cuh"
#include "entities/Vec2.cuh"
#include "helpers/float_types.h"
#include "entities/NDG_t.cuh"
#include "helpers/DataWriter_t.h"
#include <array>
#include <tuple>
#include <vector>
#include <limits>
#include <mpi.h>
#include <utility>
#include <filesystem>

namespace SEM { namespace Meshes {
    class Mesh2D_t {
        public:
            Mesh2D_t(std::filesystem::path filename, int initial_N, cudaStream_t &stream);

            SEM::Entities::device_vector<SEM::Entities::Vec2<deviceFloat>> nodes_;
            SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements_;
            SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces_;
            SEM::Entities::device_vector<std::array<size_t, 2>> interfaces_;
            SEM::Entities::device_vector<size_t> wall_boundaries_;
            SEM::Entities::device_vector<size_t> symmetry_boundaries_;
            
            

            constexpr static int elements_blockSize_ = 32;
            constexpr static int faces_blockSize_ = 32; // Same number of faces as elements for periodic BC
            constexpr static int boundaries_blockSize_ = 32;
            int elements_numBlocks_;
            int faces_numBlocks_;
            int interfaces_numBlocks_;
            int wall_boundaries_numBlocks_;
            int symmetry_boundaries_numBlocks_;
            
            size_t N_elements_global_;
            size_t N_elements_;
            size_t N_faces_;
            size_t global_element_offset_;
            size_t N_elements_per_process_;
            int initial_N_;
            deviceFloat delta_x_min_;
            int adaptivity_interval_;

            auto read_su2(std::filesystem::path filename) -> void;
            auto read_cgns(std::filesystem::path filename) -> void;
            auto initial_conditions(const deviceFloat* polynomial_nodes) -> void;
            auto boundary_conditions() -> void;
            auto print() -> void;
            auto write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void;
            auto get_delta_t(const deviceFloat CFL) -> deviceFloat;
            
            template<typename Polynomial>
            auto solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const SEM::Entities::NDG_t<Polynomial> &NDG, deviceFloat viscosity, const SEM::Helpers::DataWriter_t& data_writer) -> void;

            __host__ __device__
            static auto g(SEM::Entities::Vec2<deviceFloat> xy) -> std::array<deviceFloat, 3>;

        private:
            SEM::Entities::device_vector<deviceFloat> device_delta_t_array_;
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

            static auto build_node_to_element(size_t n_nodes, const std::vector<SEM::Entities::Element2D_t>& elements) -> std::vector<std::vector<size_t>>;
            static auto build_element_to_element(const std::vector<SEM::Entities::Element2D_t>& elements, const std::vector<std::vector<size_t>>& node_to_element) -> std::vector<std::vector<size_t>>;
            static auto build_faces(size_t n_nodes, const std::vector<SEM::Entities::Element2D_t>& elements) -> std::tuple<std::vector<SEM::Entities::Face2D_t>, std::vector<std::vector<size_t>>, std::vector<std::array<size_t, 4>>>;

            auto adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) -> void;
    };

    __global__
    auto allocate_element_storage(size_t n_elements, SEM::Entities::Element2D_t* elements) -> void;

    __global__
    auto allocate_face_storage(size_t n_faces, SEM::Entities::Face2D_t* faces) -> void;

    __global__
    auto fill_element_faces(size_t n_elements, SEM::Entities::Element2D_t* elements, const std::array<size_t, 4>* element_to_face) -> void;

    __global__
    auto initial_conditions_2D(size_t n_elements, SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto get_solution(size_t N_elements, size_t N_interpolation_points, SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, int* N) -> void;

    template<typename Polynomial>
    __global__
    void estimate_error(size_t N_elements, SEM::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights);

    __global__
    void interpolate_to_boundaries(size_t N_elements, SEM::Entities::Element2D_t* elements, const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus);

    __global__
    void calculate_wave_fluxes(size_t N_faces, SEM::Entities::Face2D_t* faces, const SEM::Entities::Element2D_t* elements);

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
