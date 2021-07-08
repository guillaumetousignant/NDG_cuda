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
            Mesh2D_t(std::filesystem::path filename, int initial_N, int maximum_N, int max_split_level, int adaptivity_interval, const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const cudaStream_t &stream);

            SEM::Entities::device_vector<SEM::Entities::Vec2<deviceFloat>> nodes_;
            SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements_;
            SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces_;
            SEM::Entities::device_vector<size_t> wall_boundaries_;
            SEM::Entities::device_vector<size_t> symmetry_boundaries_;
            SEM::Entities::device_vector<size_t> interfaces_origin_;
            SEM::Entities::device_vector<size_t> interfaces_origin_side_;
            SEM::Entities::device_vector<size_t> interfaces_destination_;
            std::vector<size_t> mpi_interfaces_size_;  // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_offset_; // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_process_; // Those are only needed on the CPU... right?
            SEM::Entities::device_vector<size_t> mpi_interfaces_origin_;
            SEM::Entities::device_vector<size_t> mpi_interfaces_origin_side_;
            SEM::Entities::device_vector<size_t> mpi_interfaces_destination_;

            SEM::Entities::device_vector<deviceFloat> device_interfaces_p_;
            SEM::Entities::device_vector<deviceFloat> device_interfaces_u_;
            SEM::Entities::device_vector<deviceFloat> device_interfaces_v_;
            std::vector<deviceFloat> host_interfaces_p_;
            std::vector<deviceFloat> host_interfaces_u_;
            std::vector<deviceFloat> host_interfaces_v_;
            std::vector<deviceFloat> host_receiving_interfaces_p_;
            std::vector<deviceFloat> host_receiving_interfaces_u_;
            std::vector<deviceFloat> host_receiving_interfaces_v_;
            
            constexpr static int elements_blockSize_ = 32;
            constexpr static int faces_blockSize_ = 32; // Same number of faces as elements for periodic BC
            constexpr static int boundaries_blockSize_ = 32;
            int elements_numBlocks_;
            int faces_numBlocks_;
            int wall_boundaries_numBlocks_;
            int symmetry_boundaries_numBlocks_;
            int ghosts_numBlocks_;
            int interfaces_numBlocks_;
            int mpi_interfaces_numBlocks_;
            
            size_t N_elements_global_;
            size_t N_elements_;
            size_t global_element_offset_;
            int initial_N_;
            int maximum_N_;
            int max_split_level_;
            int adaptivity_interval_;

            SEM::Entities::device_vector<deviceFloat> device_delta_t_array_;
            std::vector<deviceFloat> host_delta_t_array_;
            const cudaStream_t &stream_;

            auto read_su2(std::filesystem::path filename) -> void;
            auto read_cgns(std::filesystem::path filename) -> void;
            auto initial_conditions(const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes) -> void;
            auto boundary_conditions() -> void;
            auto interpolate_to_boundaries(const SEM::Entities::device_vector<deviceFloat>& lagrange_interpolant_left, const SEM::Entities::device_vector<deviceFloat>& lagrange_interpolant_right) -> void;
            auto project_to_faces(const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            auto project_to_elements(const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& weights, const SEM::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            auto print() const -> void;
            auto write_data(deviceFloat time, size_t N_interpolation_points, const SEM::Entities::device_vector<deviceFloat>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) const -> void;

            __host__ __device__
            static auto g(SEM::Entities::Vec2<deviceFloat> xy, deviceFloat t) -> std::array<deviceFloat, 3>;

            auto adapt(int N_max, const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;

            // From cppreference.com
            __device__
            static auto almost_equal(deviceFloat x, deviceFloat y) -> bool;

        private:
            SEM::Entities::device_vector<size_t> device_refine_array_;
            std::vector<size_t> host_refine_array_;

            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;

            static auto build_node_to_element(size_t n_nodes, const std::vector<SEM::Entities::Element2D_t>& elements) -> std::vector<std::vector<size_t>>;
            static auto build_element_to_element(const std::vector<SEM::Entities::Element2D_t>& elements, const std::vector<std::vector<size_t>>& node_to_element) -> std::vector<std::vector<size_t>>;
            static auto build_faces(size_t n_elements_domain, size_t n_nodes, int initial_N, const std::vector<SEM::Entities::Element2D_t>& elements) -> std::tuple<std::vector<SEM::Entities::Face2D_t>, std::vector<std::vector<size_t>>, std::vector<std::array<size_t, 4>>>;
    };

    __global__
    auto allocate_element_storage(size_t n_elements, SEM::Entities::Element2D_t* elements) -> void;

    __global__
    auto allocate_boundary_storage(size_t n_domain_elements, size_t n_total_elements, SEM::Entities::Element2D_t* elements) -> void;

    __global__
    auto compute_element_geometry(size_t n_elements, SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto allocate_face_storage(size_t n_faces, SEM::Entities::Face2D_t* faces) -> void;

    __global__
    auto fill_element_faces(size_t n_elements, SEM::Entities::Element2D_t* elements, const std::array<size_t, 4>* element_to_face) -> void;

    __global__
    auto compute_face_geometry(size_t n_faces, SEM::Entities::Face2D_t* faces, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes) -> void;

    __global__
    auto initial_conditions_2D(size_t n_elements, SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto get_solution(size_t N_elements, size_t N_interpolation_points, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, int* N, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt, deviceFloat* p_error, deviceFloat* u_error, deviceFloat* v_error, int* refine, int* coarsen, int* split_level) -> void;

    template<typename Polynomial>
    __global__
    auto estimate_error(size_t N_elements, SEM::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

    __global__
    auto interpolate_to_boundaries(size_t N_elements, SEM::Entities::Element2D_t* elements, const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

    __global__
    auto project_to_faces(size_t N_faces, SEM::Entities::Face2D_t* faces, const SEM::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto project_to_elements(size_t N_elements, const SEM::Entities::Face2D_t* faces, SEM::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto local_interfaces(size_t N_local_interfaces, SEM::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void;

    __global__
    auto get_MPI_interfaces(size_t N_MPI_interface_elements, const SEM::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, const size_t* MPI_interfaces_origin_side, int maximum_N, deviceFloat* p_, deviceFloat* u_, deviceFloat* v_) -> void;

    __global__
    auto put_MPI_interfaces(size_t N_MPI_interface_elements, SEM::Entities::Element2D_t* elements, const size_t* MPI_interfaces_destination, int maximum_N, const deviceFloat* p_, const deviceFloat* u_, const deviceFloat* v_) -> void;

    __global__
    void p_adapt(size_t N_elements, SEM::Entities::Element2D_t* elements, int N_max, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    void warp_reduce_refine_2D(volatile size_t *sdata, unsigned int tid) {
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
    void reduce_refine_2D(size_t N_elements, int max_split_level, const SEM::Entities::Element2D_t* elements, size_t* g_odata) {
        __shared__ size_t sdata[(blockSize >= 64) ? blockSize : blockSize + blockSize/2]; // Because within a warp there is no branching and this is read up until blockSize + blockSize/2
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = 0;

        while (i < N_elements) { 
            sdata[tid] += elements[i].refine_ * ((elements[i].p_sigma_ + elements[i].u_sigma_ + elements[i].v_sigma_)/3 < static_cast<deviceFloat>(1)) * (elements[i].split_level_ < max_split_level);
            if (i+blockSize < N_elements) {
                sdata[tid] += elements[i+blockSize].refine_ * ((elements[i+blockSize].p_sigma_ + elements[i+blockSize].u_sigma_ + elements[i+blockSize].v_sigma_)/3 < static_cast<deviceFloat>(1)) * (elements[i+blockSize].split_level_ < max_split_level);
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

        if (tid < 32) warp_reduce_refine_2D<blockSize>(sdata, tid);
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}}

#endif
