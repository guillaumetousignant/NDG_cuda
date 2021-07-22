#ifndef NDG_MESH2D_T_H
#define NDG_MESH2D_T_H

#include "entities/Element2D_t.cuh"
#include "entities/Face2D_t.cuh"
#include "entities/NDG_t.cuh"
#include "entities/device_vector.cuh"
#include "entities/host_vector.cuh"
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
            Mesh2D_t(std::filesystem::path filename, int initial_N, int maximum_N, size_t n_interpolation_points, int max_split_level, int adaptivity_interval, const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const cudaStream_t &stream);

            // Geometry
            SEM::Entities::device_vector<SEM::Entities::Vec2<deviceFloat>> nodes_;
            SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements_;
            SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces_;

            // Boundaries
            SEM::Entities::device_vector<size_t> wall_boundaries_;
            SEM::Entities::device_vector<size_t> symmetry_boundaries_;
            SEM::Entities::device_vector<size_t> inflow_boundaries_;
            SEM::Entities::device_vector<size_t> outflow_boundaries_;

            // Interfaces
            SEM::Entities::device_vector<size_t> interfaces_origin_;
            SEM::Entities::device_vector<size_t> interfaces_origin_side_;
            SEM::Entities::device_vector<size_t> interfaces_destination_;
            std::vector<size_t> mpi_interfaces_size_;  // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_offset_; // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_process_; // Those are only needed on the CPU... right?
            SEM::Entities::device_vector<size_t> mpi_interfaces_origin_;
            SEM::Entities::device_vector<size_t> mpi_interfaces_origin_side_;
            SEM::Entities::device_vector<size_t> mpi_interfaces_destination_;

            // Boundary solution exchange
            SEM::Entities::device_vector<deviceFloat> device_interfaces_p_;
            SEM::Entities::device_vector<deviceFloat> device_interfaces_u_;
            SEM::Entities::device_vector<deviceFloat> device_interfaces_v_;
            SEM::Entities::device_vector<int> device_interfaces_N_;
            SEM::Entities::host_vector<deviceFloat> host_interfaces_p_;
            SEM::Entities::host_vector<deviceFloat> host_interfaces_u_;
            SEM::Entities::host_vector<deviceFloat> host_interfaces_v_;
            std::vector<int> host_interfaces_N_;
            SEM::Entities::host_vector<deviceFloat> host_receiving_interfaces_p_;
            SEM::Entities::host_vector<deviceFloat> host_receiving_interfaces_u_;
            SEM::Entities::host_vector<deviceFloat> host_receiving_interfaces_v_;
            std::vector<int> host_receiving_interfaces_N_;

            // Output
            std::vector<deviceFloat> x_output_host_;
            std::vector<deviceFloat> y_output_host_;
            std::vector<deviceFloat> p_output_host_;
            std::vector<deviceFloat> u_output_host_;
            std::vector<deviceFloat> v_output_host_;
            SEM::Entities::device_vector<deviceFloat> x_output_device_;
            SEM::Entities::device_vector<deviceFloat> y_output_device_;
            SEM::Entities::device_vector<deviceFloat> p_output_device_;
            SEM::Entities::device_vector<deviceFloat> u_output_device_;
            SEM::Entities::device_vector<deviceFloat> v_output_device_;
            
            // Parallel sizings
            constexpr static int elements_blockSize_ = 32;
            constexpr static int faces_blockSize_ = 32; // Same number of faces as elements for periodic BC
            constexpr static int boundaries_blockSize_ = 32;
            int elements_numBlocks_;
            int faces_numBlocks_;
            int wall_boundaries_numBlocks_;
            int symmetry_boundaries_numBlocks_;
            int inflow_boundaries_numBlocks_;
            int outflow_boundaries_numBlocks_;
            int ghosts_numBlocks_;
            int interfaces_numBlocks_;
            int mpi_interfaces_numBlocks_;
            
            // Counts
            size_t n_elements_global_;
            size_t n_elements_;
            size_t global_element_offset_;

            // Parameters
            int initial_N_;
            int maximum_N_;
            size_t n_interpolation_points_;
            int max_split_level_;
            int adaptivity_interval_;
            deviceFloat tolerance_min_;
            deviceFloat tolerance_max_;

            // GPU transfer arrays
            SEM::Entities::device_vector<deviceFloat> device_delta_t_array_;
            SEM::Entities::host_vector<deviceFloat> host_delta_t_array_;
            SEM::Entities::device_vector<size_t> device_refine_array_;
            std::vector<size_t> host_refine_array_;
            SEM::Entities::device_vector<size_t> device_nodes_refine_array_;
            std::vector<size_t> host_nodes_refine_array_;

            const cudaStream_t &stream_;

            auto read_su2(std::filesystem::path filename) -> void;
            auto read_cgns(std::filesystem::path filename) -> void;
            auto initial_conditions(const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes) -> void;
            auto boundary_conditions(deviceFloat t, const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& weights, const SEM::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            auto interpolate_to_boundaries(const SEM::Entities::device_vector<deviceFloat>& lagrange_interpolant_left, const SEM::Entities::device_vector<deviceFloat>& lagrange_interpolant_right) -> void;
            auto project_to_faces(const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            auto project_to_elements(const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& weights, const SEM::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            
            template<typename Polynomial>
            auto estimate_error(const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& weights) -> void;
            
            auto print() const -> void;
            auto write_data(deviceFloat time, const SEM::Entities::device_vector<deviceFloat>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void;
            auto write_complete_data(deviceFloat time, const SEM::Entities::device_vector<deviceFloat>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void;

            __host__ __device__
            static auto g(SEM::Entities::Vec2<deviceFloat> xy, deviceFloat t) -> std::array<deviceFloat, 3>;

            auto adapt(int N_max, const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;

            // From cppreference.com
            __device__
            static auto almost_equal(deviceFloat x, deviceFloat y) -> bool;

        private:
            // MPI exchange variables
            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;
            std::vector<MPI_Request> requests_N_;
            std::vector<MPI_Status> statuses_N_;

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
    auto compute_boundary_geometry(size_t n_domain_elements, size_t n_total_elements, SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto allocate_face_storage(size_t n_faces, SEM::Entities::Face2D_t* faces) -> void;

    __global__
    auto fill_element_faces(size_t n_elements, SEM::Entities::Element2D_t* elements, const std::array<size_t, 4>* element_to_face) -> void;

    __global__
    auto compute_face_geometry(size_t n_faces, SEM::Entities::Face2D_t* faces, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes) -> void;

    __global__
    auto initial_conditions_2D(size_t n_elements, SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto get_solution(size_t n_elements, size_t n_interpolation_points, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void;

    __global__
    auto get_complete_solution(size_t n_elements, size_t n_interpolation_points, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, int* N, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt, deviceFloat* p_error, deviceFloat* u_error, deviceFloat* v_error, deviceFloat* p_sigma, deviceFloat* u_sigma, deviceFloat* v_sigma, int* refine, int* coarsen, int* split_level) -> void;

    template<typename Polynomial>
    __global__
    auto estimate_error(size_t n_elements, SEM::Entities::Element2D_t* elements, deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

    __global__
    auto interpolate_to_boundaries(size_t n_elements, SEM::Entities::Element2D_t* elements, const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

    __global__
    auto project_to_faces(size_t N_faces, SEM::Entities::Face2D_t* faces, const SEM::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto project_to_elements(size_t n_elements, const SEM::Entities::Face2D_t* faces, SEM::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;
    
    __global__
    auto compute_wall_boundaries(size_t n_wall_boundaries, SEM::Entities::Element2D_t* elements, const size_t* wall_boundaries, const SEM::Entities::Face2D_t* faces, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto compute_symmetry_boundaries(size_t n_symmetry_boundaries, SEM::Entities::Element2D_t* elements, const size_t* symmetry_boundaries, const SEM::Entities::Face2D_t* faces, const deviceFloat* polynomial_node, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto compute_inflow_boundaries(size_t n_inflow_boundaries, SEM::Entities::Element2D_t* elements, const size_t* inflow_boundaries, const SEM::Entities::Face2D_t* faces, deviceFloat t, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto compute_outflow_boundaries(size_t n_outflow_boundaries, SEM::Entities::Element2D_t* elements, const size_t* outflow_boundaries, const SEM::Entities::Face2D_t* faces, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto local_interfaces(size_t N_local_interfaces, SEM::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void;

    __global__
    auto get_MPI_interfaces(size_t N_MPI_interface_elements, const SEM::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, const size_t* MPI_interfaces_origin_side, int maximum_N, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void;

    __global__
    auto get_MPI_interfaces_N(size_t N_MPI_interface_elements, const SEM::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, int* N) -> void;

    __global__
    auto put_MPI_interfaces(size_t N_MPI_interface_elements, SEM::Entities::Element2D_t* elements, const size_t* MPI_interfaces_destination, int maximum_N, const deviceFloat* p, const deviceFloat* u, const deviceFloat* v) -> void;

    __global__
    auto put_MPI_interfaces_N(size_t N_MPI_interface_elements, SEM::Entities::Element2D_t* elements, const size_t* MPI_interfaces_destination, const int* N) -> void;
    
    __global__
    auto put_MPI_interfaces_N_and_rebuild(size_t N_MPI_interface_elements, SEM::Entities::Element2D_t* elements, SEM::Entities::Element2D_t* new_elements, const size_t* MPI_interfaces_destination, const int* N) -> void;

    __global__
    auto p_adapt(size_t n_elements, SEM::Entities::Element2D_t* elements, int N_max, const SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;
    
    __global__
    auto hp_adapt(size_t n_elements, size_t n_faces, size_t n_nodes, SEM::Entities::Element2D_t* elements, SEM::Entities::Element2D_t* new_elements, const SEM::Entities::Face2D_t* faces, const size_t* block_offsets, const size_t* nodes_block_offsets, int max_split_level, int N_max, SEM::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto adjust_boundaries(size_t N_boundaries, SEM::Entities::Element2D_t* elements, const size_t* boundaries, const SEM::Entities::Face2D_t* faces) -> void;

    __global__
    auto rebuild_boundaries(size_t N_boundaries, SEM::Entities::Element2D_t* elements, SEM::Entities::Element2D_t* new_elements, const size_t* boundaries, const SEM::Entities::Face2D_t* faces) -> void;

    __global__
    auto adjust_interfaces(size_t N_local_interfaces, SEM::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination) -> void;

    __global__
    auto rebuild_interfaces(size_t N_local_interfaces, SEM::Entities::Element2D_t* elements, SEM::Entities::Element2D_t* new_elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination) -> void;

    __global__
    auto adjust_faces(size_t N_faces, SEM::Entities::Face2D_t* faces, const SEM::Entities::Element2D_t* elements) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    auto warp_reduce_2D(volatile size_t *sdata, unsigned int tid) -> void {
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
    auto reduce_refine_2D(size_t n_elements, int max_split_level, const SEM::Entities::Element2D_t* elements, size_t* g_odata) -> void {
        __shared__ size_t sdata[(blockSize >= 64) ? blockSize : blockSize + blockSize/2]; // Because within a warp there is no branching and this is read up until blockSize + blockSize/2
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = 0;

        while (i < n_elements) { 
            sdata[tid] += elements[i].refine_ * ((elements[i].p_sigma_ + elements[i].u_sigma_ + elements[i].v_sigma_)/3 < static_cast<deviceFloat>(1)) * (elements[i].split_level_ < max_split_level);
            if (i+blockSize < n_elements) {
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

        if (tid < 32) warp_reduce_2D<blockSize>(sdata, tid);
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_nodes_2D(size_t n_elements, int max_split_level, SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, const SEM::Entities::Vec2<deviceFloat>* nodes, size_t* g_odata) -> void {
        __shared__ size_t sdata[(blockSize >= 64) ? blockSize : blockSize + blockSize/2]; // Because within a warp there is no branching and this is read up until blockSize + blockSize/2
        unsigned int tid = threadIdx.x;
        size_t i = blockIdx.x*(blockSize*2) + tid;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = 0;

        while (i < n_elements) {
            SEM::Entities::Element2D_t& element = elements[i];
            element.additional_nodes_ = {false, false, false, false};
            if (element.refine_ * ((element.p_sigma_ + element.u_sigma_ + element.v_sigma_)/3 < static_cast<deviceFloat>(1)) * (element.split_level_ < max_split_level)) {
                // This is the middle node, always needs to be created
                ++sdata[tid];
                ++element.n_additional_nodes_;
                for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                    const std::array<SEM::Entities::Vec2<deviceFloat>, 2> side_nodes = {nodes[element.nodes_[side_index]], (side_index < element.faces_.size() - 1) ? nodes[element.nodes_[side_index + 1]] : nodes[element.nodes_[0]]};
                    const SEM::Entities::Vec2<deviceFloat> new_node = (side_nodes[0] + side_nodes[1])/2;

                    // Here we check if the new node already exists
                    bool found_node = false;
                    for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                        const SEM::Entities::Face2D_t& face = faces[element.faces_[side_index][face_index]];
                        if (nodes[face.nodes_[0]] == new_node || nodes[face.nodes_[1]] == new_node) {
                            found_node = true;
                            break;
                        }
                    }

                    // Here we check if another element would create the same node, and yield if its index is smaller
                    if (!found_node) {
                        for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                            const SEM::Entities::Face2D_t& face = faces[element.faces_[side_index][face_index]];
                            const int face_side = face.elements_[0] == i;
                            const SEM::Entities::Element2D_t& neighbour = elements[face.elements_[face_side]];
                            
                            if (neighbour.refine_ * ((neighbour.p_sigma_ + neighbour.u_sigma_ + neighbour.v_sigma_)/3 < static_cast<deviceFloat>(1)) * (neighbour.split_level_ < max_split_level)) {
                                const std::array<SEM::Entities::Vec2<deviceFloat>, 2> neighbour_nodes = {nodes[neighbour.nodes_[face.elements_side_[face_side]]], (face.elements_side_[face_side] < neighbour.faces_.size() - 1) ? nodes[neighbour.nodes_[face.elements_side_[face_side] + 1]] : nodes[neighbour.nodes_[0]]};
                                const SEM::Entities::Vec2<deviceFloat> neighbour_new_node = (neighbour_nodes[0] + neighbour_nodes[1])/2;

                                if (new_node.almost_equal(neighbour_new_node) && face.elements_[face_side] < i) {
                                    found_node = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (!found_node) {
                        ++sdata[tid];
                        ++element.n_additional_nodes_;
                    }
                }
            }

            if (i+blockSize < n_elements) {
                SEM::Entities::Element2D_t& element = elements[i+blockSize];
                element.additional_nodes_ = {false, false, false, false};
                if (element.refine_ * ((element.p_sigma_ + element.u_sigma_ + element.v_sigma_)/3 < static_cast<deviceFloat>(1)) * (element.split_level_ < max_split_level)) {
                    // This is the middle node, always needs to be created
                    ++sdata[tid];
                    ++element.n_additional_nodes_;
                    for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                        const std::array<SEM::Entities::Vec2<deviceFloat>, 2> side_nodes = {nodes[element.nodes_[side_index]], (side_index < element.faces_.size() - 1) ? nodes[element.nodes_[side_index + 1]] : nodes[element.nodes_[0]]};
                        const SEM::Entities::Vec2<deviceFloat> new_node = (side_nodes[0] + side_nodes[1])/2;
    
                        // Here we check if the new node already exists
                        bool found_node = false;
                        for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                            const SEM::Entities::Face2D_t& face = faces[element.faces_[side_index][face_index]];
                            if (nodes[face.nodes_[0]] == new_node || nodes[face.nodes_[1]] == new_node) {
                                found_node = true;
                                break;
                            }
                        }
    
                        if (!found_node) {
                            for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                                const SEM::Entities::Face2D_t& face = faces[element.faces_[side_index][face_index]];
                                const int face_side = face.elements_[0] == i;
                                const SEM::Entities::Element2D_t& neighbour = elements[face.elements_[face_side]];
                                
                                if (neighbour.refine_ * ((neighbour.p_sigma_ + neighbour.u_sigma_ + neighbour.v_sigma_)/3 < static_cast<deviceFloat>(1)) * (neighbour.split_level_ < max_split_level)) {
                                    const std::array<SEM::Entities::Vec2<deviceFloat>, 2> neighbour_nodes = {nodes[neighbour.nodes_[face.elements_side_[face_side]]], (face.elements_side_[face_side] < neighbour.faces_.size() - 1) ? nodes[neighbour.nodes_[face.elements_side_[face_side] + 1]] : nodes[neighbour.nodes_[0]]};
                                    const SEM::Entities::Vec2<deviceFloat> neighbour_new_node = (neighbour_nodes[0] + neighbour_nodes[1])/2;
    
                                    if (new_node.almost_equal(neighbour_new_node) && face.elements_[face_side] < i+blockSize) {
                                        found_node = true;
                                        break;
                                    }
                                }
                            }
                        }
    
                        if (!found_node) {
                            ++sdata[tid];
                            ++element.n_additional_nodes_;
                        }
                    }
                }
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

        if (tid < 32) warp_reduce_2D<blockSize>(sdata, tid);
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}}

#endif
