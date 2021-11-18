#ifndef NDG_MESHES_MESH2D_T_CUH
#define NDG_MESHES_MESH2D_T_CUH

#include "entities/Element2D_t.cuh"
#include "entities/Face2D_t.cuh"
#include "entities/device_vector.cuh"
#include "entities/host_vector.cuh"
#include "entities/Vec2.cuh"
#include "helpers/float_types.h"
#include "helpers/DataWriter_t.h"
#include <array>
#include <tuple>
#include <vector>
#include <limits>
#include <mpi.h>
#include <utility>
#include <filesystem>

namespace SEM { namespace Device { namespace Meshes {
    class Mesh2D_t {
        public:
            Mesh2D_t(std::filesystem::path filename, int initial_N, int maximum_N, size_t n_interpolation_points, int max_split_level, int adaptivity_interval, int load_balancing_interval, deviceFloat tolerance_min, deviceFloat tolerance_max, const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes, const cudaStream_t &stream);

            // Geometry
            SEM::Device::Entities::device_vector<SEM::Device::Entities::Vec2<deviceFloat>> nodes_;
            SEM::Device::Entities::device_vector<SEM::Device::Entities::Element2D_t> elements_;
            SEM::Device::Entities::device_vector<SEM::Device::Entities::Face2D_t> faces_;

            // Boundaries
            SEM::Device::Entities::device_vector<size_t> wall_boundaries_;
            SEM::Device::Entities::device_vector<size_t> symmetry_boundaries_;
            SEM::Device::Entities::device_vector<size_t> inflow_boundaries_;
            SEM::Device::Entities::device_vector<size_t> outflow_boundaries_;
            enum boundary_type : int {wall = -1, symmetry = -2, inflow = -3, outflow = -4};

            // Interfaces
            SEM::Device::Entities::device_vector<size_t> interfaces_origin_;
            SEM::Device::Entities::device_vector<size_t> interfaces_origin_side_;
            SEM::Device::Entities::device_vector<size_t> interfaces_destination_;
            std::vector<size_t> mpi_interfaces_outgoing_size_;  // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_incoming_size_;  // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_outgoing_offset_; // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_incoming_offset_; // Those are only needed on the CPU... right?
            std::vector<int> mpi_interfaces_process_; // Those are only needed on the CPU... right?
            SEM::Device::Entities::device_vector<size_t> mpi_interfaces_origin_;
            SEM::Device::Entities::device_vector<size_t> mpi_interfaces_origin_side_;
            SEM::Device::Entities::device_vector<size_t> mpi_interfaces_destination_;

            // Boundary solution exchange
            SEM::Device::Entities::device_vector<deviceFloat> device_interfaces_p_;
            SEM::Device::Entities::device_vector<deviceFloat> device_interfaces_u_;
            SEM::Device::Entities::device_vector<deviceFloat> device_interfaces_v_;
            SEM::Device::Entities::device_vector<deviceFloat> device_receiving_interfaces_p_;
            SEM::Device::Entities::device_vector<deviceFloat> device_receiving_interfaces_u_;
            SEM::Device::Entities::device_vector<deviceFloat> device_receiving_interfaces_v_;
            SEM::Device::Entities::device_vector<int> device_interfaces_N_;
            SEM::Device::Entities::host_vector<deviceFloat> host_interfaces_p_;
            SEM::Device::Entities::host_vector<deviceFloat> host_interfaces_u_;
            SEM::Device::Entities::host_vector<deviceFloat> host_interfaces_v_;
            std::vector<int> host_interfaces_N_;
            SEM::Device::Entities::host_vector<deviceFloat> host_receiving_interfaces_p_;
            SEM::Device::Entities::host_vector<deviceFloat> host_receiving_interfaces_u_;
            SEM::Device::Entities::host_vector<deviceFloat> host_receiving_interfaces_v_;
            std::vector<int> host_receiving_interfaces_N_;
            SEM::Device::Entities::device_vector<bool> device_interfaces_refine_;
            SEM::Device::Entities::device_vector<int> device_receiving_interfaces_N_;
            SEM::Device::Entities::device_vector<bool> device_receiving_interfaces_refine_;
            SEM::Device::Entities::host_vector<bool> host_interfaces_refine_;
            SEM::Device::Entities::host_vector<bool> host_receiving_interfaces_refine_;

            // Output
            std::vector<deviceFloat> x_output_host_;
            std::vector<deviceFloat> y_output_host_;
            std::vector<deviceFloat> p_output_host_;
            std::vector<deviceFloat> u_output_host_;
            std::vector<deviceFloat> v_output_host_;
            SEM::Device::Entities::device_vector<deviceFloat> x_output_device_;
            SEM::Device::Entities::device_vector<deviceFloat> y_output_device_;
            SEM::Device::Entities::device_vector<deviceFloat> p_output_device_;
            SEM::Device::Entities::device_vector<deviceFloat> u_output_device_;
            SEM::Device::Entities::device_vector<deviceFloat> v_output_device_;
            
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
            int mpi_interfaces_outgoing_numBlocks_;
            int mpi_interfaces_incoming_numBlocks_;
            
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
            int load_balancing_interval_;
            deviceFloat tolerance_min_;
            deviceFloat tolerance_max_;

            // GPU transfer arrays
            SEM::Device::Entities::device_vector<deviceFloat> device_delta_t_array_;
            SEM::Device::Entities::host_vector<deviceFloat> host_delta_t_array_;
            SEM::Device::Entities::device_vector<size_t> device_refine_array_;
            std::vector<size_t> host_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_faces_refine_array_;
            std::vector<size_t> host_faces_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_wall_boundaries_refine_array_;
            std::vector<size_t> host_wall_boundaries_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_symmetry_boundaries_refine_array_;
            std::vector<size_t> host_symmetry_boundaries_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_inflow_boundaries_refine_array_;
            std::vector<size_t> host_inflow_boundaries_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_outflow_boundaries_refine_array_;
            std::vector<size_t> host_outflow_boundaries_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_interfaces_refine_array_;
            std::vector<size_t> host_interfaces_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_mpi_interfaces_outgoing_refine_array_;
            std::vector<size_t> host_mpi_interfaces_outgoing_refine_array_;
            SEM::Device::Entities::device_vector<size_t> device_mpi_interfaces_incoming_refine_array_;
            std::vector<size_t> host_mpi_interfaces_incoming_refine_array_;

            const cudaStream_t &stream_;

            auto read_su2(std::filesystem::path filename) -> void;
            auto read_cgns(std::filesystem::path filename) -> void;
            auto initial_conditions(const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes) -> void;
            auto boundary_conditions(deviceFloat t, const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Device::Entities::device_vector<deviceFloat>& weights, const SEM::Device::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            auto interpolate_to_boundaries(const SEM::Device::Entities::device_vector<deviceFloat>& lagrange_interpolant_left, const SEM::Device::Entities::device_vector<deviceFloat>& lagrange_interpolant_right) -> void;
            auto project_to_faces(const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Device::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            auto project_to_elements(const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Device::Entities::device_vector<deviceFloat>& weights, const SEM::Device::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            
            template<typename Polynomial>
            auto estimate_error(const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Device::Entities::device_vector<deviceFloat>& weights) -> void;
            
            auto print() const -> void;
            auto write_data(deviceFloat time, const SEM::Device::Entities::device_vector<deviceFloat>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void;
            auto write_complete_data(deviceFloat time, const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Device::Entities::device_vector<deviceFloat>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void;

            auto adapt(int N_max, const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes, const SEM::Device::Entities::device_vector<deviceFloat>& barycentric_weights) -> void;
            
            auto load_balance(const SEM::Device::Entities::device_vector<deviceFloat>& polynomial_nodes) -> void;

            // From cppreference.com
            __device__
            static auto almost_equal(deviceFloat x, deviceFloat y) -> bool;

        private:
            // MPI exchange variables
            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;
            std::vector<MPI_Request> requests_adaptivity_;
            std::vector<MPI_Status> statuses_adaptivity_;

            static auto build_node_to_element(size_t n_nodes, const std::vector<SEM::Device::Entities::Element2D_t>& elements) -> std::vector<std::vector<size_t>>;
            static auto build_element_to_element(const std::vector<SEM::Device::Entities::Element2D_t>& elements, const std::vector<std::vector<size_t>>& node_to_element) -> std::vector<std::vector<size_t>>;
            static auto build_faces(size_t n_elements_domain, size_t n_nodes, int initial_N, const std::vector<SEM::Device::Entities::Element2D_t>& elements) -> std::tuple<std::vector<SEM::Device::Entities::Face2D_t>, std::vector<std::vector<size_t>>, std::vector<std::array<size_t, 4>>>;
    };

    __global__
    auto allocate_element_storage(size_t n_elements, SEM::Device::Entities::Element2D_t* elements) -> void;

    __global__
    auto allocate_boundary_storage(size_t n_domain_elements, size_t n_total_elements, SEM::Device::Entities::Element2D_t* elements) -> void;

    __global__
    auto compute_element_geometry(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;
    
    __global__
    auto compute_boundary_geometry(size_t n_domain_elements, size_t n_total_elements, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto compute_element_status(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes) -> void;
    
    __global__
    auto allocate_face_storage(size_t n_faces, SEM::Device::Entities::Face2D_t* faces) -> void;

    __global__
    auto fill_element_faces(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, const std::array<size_t, 4>* element_to_face) -> void;

    __global__
    auto fill_boundary_element_faces(size_t n_domain_elements, size_t n_total_elements, SEM::Device::Entities::Element2D_t* elements, const std::array<size_t, 4>* element_to_face) -> void;

    __global__
    auto compute_face_geometry(size_t n_faces, SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes) -> void;

    __global__
    auto initial_conditions_2D(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto get_solution(size_t n_elements, size_t n_interpolation_points, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void;

    __global__
    auto get_complete_solution(size_t n_elements, size_t n_interpolation_points, deviceFloat time, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, int* N, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt, deviceFloat* p_error, deviceFloat* u_error, deviceFloat* v_error, deviceFloat* p_sigma, deviceFloat* u_sigma, deviceFloat* v_sigma, int* refine, int* coarsen, int* split_level, deviceFloat* p_analytical_error, deviceFloat* u_analytical_error, deviceFloat* v_analytical_error, int* status, int* rotation) -> void;

    template<typename Polynomial>
    __global__
    auto estimate_error(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

    __global__
    auto interpolate_to_boundaries(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

    __global__
    auto project_to_faces(size_t n_faces, SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto project_to_elements(size_t n_elements, const SEM::Device::Entities::Face2D_t* faces, SEM::Device::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;
    
    __global__
    auto compute_wall_boundaries(size_t n_wall_boundaries, SEM::Device::Entities::Element2D_t* elements, const size_t* wall_boundaries, const SEM::Device::Entities::Face2D_t* faces, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto compute_symmetry_boundaries(size_t n_symmetry_boundaries, SEM::Device::Entities::Element2D_t* elements, const size_t* symmetry_boundaries, const SEM::Device::Entities::Face2D_t* faces, const deviceFloat* polynomial_node, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto compute_inflow_boundaries(size_t n_inflow_boundaries, SEM::Device::Entities::Element2D_t* elements, const size_t* inflow_boundaries, const SEM::Device::Entities::Face2D_t* faces, deviceFloat t, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto compute_outflow_boundaries(size_t n_outflow_boundaries, SEM::Device::Entities::Element2D_t* elements, const size_t* outflow_boundaries, const SEM::Device::Entities::Face2D_t* faces, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void;

    __global__
    auto local_interfaces(size_t n_local_interfaces, SEM::Device::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void;

    __global__
    auto get_MPI_interfaces(size_t n_MPI_interface_elements, const SEM::Device::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, const size_t* MPI_interfaces_origin_side, int maximum_N, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void;

    __global__
    auto get_MPI_interfaces_N(size_t n_MPI_interface_elements, int N_max, const SEM::Device::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, int* N) -> void;

    __global__
    auto get_MPI_interfaces_adaptivity(size_t n_MPI_interface_elements, const SEM::Device::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, int* N, bool* elements_splitting, int max_split_level, int N_max) -> void;

    __global__
    auto put_MPI_interfaces(size_t n_MPI_interface_elements, SEM::Device::Entities::Element2D_t* elements, const size_t* MPI_interfaces_destination, int maximum_N, const deviceFloat* p, const deviceFloat* u, const deviceFloat* v) -> void;

    __global__
    auto adjust_MPI_incoming_interfaces(size_t n_MPI_interface_elements, SEM::Device::Entities::Element2D_t* elements, const size_t* MPI_interfaces_destination, const int* N, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void;
    
    __global__
    auto p_adapt(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, int N_max, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;
    
    __global__
    auto p_adapt_move(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, int N_max, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights, size_t* elements_new_indices) -> void;
    
    __global__
    auto p_adapt_split_faces(size_t n_elements, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, const SEM::Device::Entities::Face2D_t* faces, int N_max, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights, const size_t* faces_block_offsets, int faces_blockSize, size_t* elements_new_indices) -> void;
    
    __global__
    auto hp_adapt(size_t n_elements, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, SEM::Device::Entities::Face2D_t* faces, SEM::Device::Entities::Face2D_t* new_faces, const size_t* block_offsets, const size_t* faces_block_offsets, int max_split_level, int N_max, SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights, int faces_blockSize, size_t* elements_new_indices) -> void;
    
    __global__
    auto split_faces(size_t n_faces, size_t n_nodes, size_t n_splitting_elements, SEM::Device::Entities::Face2D_t* faces, SEM::Device::Entities::Face2D_t* new_faces, const SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* faces_block_offsets, int max_split_level, int N_max, const size_t* elements_new_indices) -> void;

    __global__
    auto find_nodes(size_t n_elements, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, int max_split_level) -> void;
    
    __global__
    auto no_new_nodes(size_t n_elements, SEM::Device::Entities::Element2D_t* elements) -> void;

    __global__
    auto copy_boundaries_error(size_t n_boundaries, SEM::Device::Entities::Element2D_t* elements, const size_t* boundaries, const SEM::Device::Entities::Face2D_t* faces) -> void;

    __global__
    auto copy_interfaces_error(size_t n_local_interfaces, SEM::Device::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void;
    
    __global__
    auto copy_mpi_interfaces_error(size_t n_MPI_interface_elements, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* MPI_interfaces_destination, const int* N, const bool* elements_splitting) -> void;

    __global__
    auto split_boundaries(size_t n_boundaries, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, size_t offset, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, const size_t* boundaries, size_t* new_boundaries, const SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* faces_block_offsets, const size_t* boundary_block_offsets, const deviceFloat* polynomial_nodes, int faces_blockSize, size_t* elements_new_indices) -> void;

    __global__
    auto split_interfaces(size_t n_local_interfaces, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, size_t offset, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination, size_t* new_local_interfaces_origin, size_t* new_local_interfaces_origin_side, size_t* new_local_interfaces_destination, const SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* block_offsets, const size_t* faces_block_offsets, const size_t* interface_block_offsets, int max_split_level, int N_max, const deviceFloat* polynomial_nodes, int elements_blockSize, int faces_blockSize, size_t* elements_new_indices) -> void;
    
    __global__
    auto split_mpi_outgoing_interfaces(size_t n_MPI_interface_elements, const SEM::Device::Entities::Element2D_t* elements, const size_t* mpi_interfaces_origin, const size_t* mpi_interfaces_origin_side, size_t* new_mpi_interfaces_origin, size_t* new_mpi_interfaces_origin_side, const size_t* mpi_interface_block_offsets, int max_split_level, const size_t* block_offsets, int elements_blockSize) -> void;
    
    __global__
    auto split_mpi_incoming_interfaces(size_t n_MPI_interface_elements, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, size_t offset, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, const size_t* mpi_interfaces_destination, size_t* new_mpi_interfaces_destination, const SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* faces_block_offsets, const size_t* mpi_interface_block_offsets, const deviceFloat* polynomial_nodes, int faces_blockSize, const int* N, const bool* elements_splitting, size_t* elements_new_indices) -> void;

    __global__
    auto adjust_boundaries(size_t n_boundaries, SEM::Device::Entities::Element2D_t* elements, const size_t* boundaries, const SEM::Device::Entities::Face2D_t* faces) -> void;

    __global__
    auto adjust_interfaces(size_t n_local_interfaces, SEM::Device::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination) -> void;

    __global__
    auto adjust_faces(size_t n_faces, SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Element2D_t* elements) -> void;
    
    __global__
    auto adjust_faces_neighbours(size_t n_faces, SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, int max_split_level, int N_max, const size_t* elements_new_indices) -> void;

    __global__
    auto move_boundaries(size_t n_boundaries, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, const size_t* boundaries, const SEM::Device::Entities::Face2D_t* faces, size_t* elements_new_indices) -> void;

    __global__
    auto move_interfaces(size_t n_local_interfaces, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination, size_t* elements_new_indices) -> void;

    __global__
    auto print_element_faces(size_t n_elements, const SEM::Device::Entities::Element2D_t* elements) -> void;

    __global__
    auto print_boundary_element_faces(size_t n_domain_elements, size_t n_total_elements, const SEM::Device::Entities::Element2D_t* elements) -> void;

    __global__
    auto get_transfer_solution(size_t n_elements, const SEM::Device::Entities::Element2D_t* elements, int maximum_N, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, deviceFloat* solution, size_t* n_neighbours, deviceFloat* element_nodes) -> void;
    
    __global__
    auto fill_received_elements(size_t n_elements, size_t n_domain_elements, size_t n_elements_recv_left, size_t n_elements_recv_right, SEM::Device::Entities::Element2D_t* elements, int maximum_N, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* solution, const size_t* n_neighbours, const size_t* neighbours_offsets, const size_t* received_node_indices, const size_t* neighbours_indices, const size_t* neighbours_sides, const deviceFloat* polynomial_nodes) -> void;

    __global__
    auto get_neighbours(size_t n_elements_send, size_t start_index, size_t n_domain_elements, size_t n_wall_boundaries, size_t n_symmetry_boundaries, size_t n_inflow_boundaries, size_t n_outflow_boundaries, size_t n_local_interfaces, size_t n_MPI_interface_elements_receiving, int rank, int n_procs, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* wall_boundaries, const size_t* symmetry_boundaries, const size_t* inflow_boundaries, const size_t* outflow_boundaries, const size_t* interfaces_destination, const size_t* interfaces_origin, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_process, const size_t* mpi_interfaces_local_indices, const size_t* mpi_interfaces_side, const size_t* offsets, const size_t* n_elements_received_left, const size_t* n_elements_sent_left, const size_t* n_elements_received_right, const size_t* n_elements_sent_right, const size_t* global_element_offset, const size_t* global_element_offset_new, const size_t* global_element_offset_end_new, size_t* neighbours, deviceFloat* neighbours_nodes, int* neighbours_proc, size_t* neighbours_side, int* neighbours_N) -> void;

    __global__
    auto move_elements(size_t n_elements_move, size_t n_elements_send_left, size_t n_elements_recv_left, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, SEM::Device::Entities::Face2D_t* faces) -> void;

    __global__
    auto get_interface_n_processes(size_t n_mpi_interfaces, size_t n_mpi_interfaces_incoming, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, const size_t* mpi_interfaces_origin, const size_t* mpi_interfaces_origin_side, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, size_t* n_processes) -> void;
    
    __global__
    auto get_interface_processes(size_t n_mpi_interfaces, size_t n_mpi_interfaces_incoming, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, const size_t* mpi_interfaces_origin, const size_t* mpi_interfaces_origin_side, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, const size_t* process_offsets, int* processes) -> void;
    
    __global__
    auto find_received_nodes(size_t n_received_nodes, size_t n_nodes, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* received_nodes, bool* missing_nodes, bool* missing_received_nodes, size_t* received_nodes_indices, size_t* received_node_received_indices) -> void;
    
    __global__
    auto add_new_received_nodes(size_t n_received_nodes, size_t n_nodes, SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* received_nodes, const bool* missing_nodes, const bool* missing_received_nodes, size_t* received_nodes_indices, const size_t* received_node_received_indices, const size_t* received_nodes_block_offsets) -> void;

    __global__
    auto find_received_neighbour_nodes(size_t n_received_neighbour_nodes, size_t n_received_nodes, size_t n_nodes, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* received_neighbour_nodes, const deviceFloat* received_nodes, bool* missing_neighbour_nodes, bool* missing_received_neighbour_nodes, size_t* received_neighbour_nodes_indices, size_t* received_neighbour_node_received_indices) -> void;
  
    __global__
    auto add_new_received_neighbour_nodes(size_t n_received_neighbour_nodes, size_t n_received_nodes, size_t n_nodes, SEM::Device::Entities::Vec2<deviceFloat>* nodes, const deviceFloat* received_neighbour_nodes, const bool* missing_neighbour_nodes, const bool* missing_nodes, const bool* missing_received_neighbour_nodes, size_t* received_neighbour_nodes_indices, const size_t* received_neighbour_node_received_indices, const size_t* received_neighbour_nodes_block_offsets, const size_t* received_nodes_block_offsets) -> void;

    __global__
    auto find_faces_to_delete(size_t n_faces, size_t n_domain_elements, size_t n_elements_send_left, size_t n_elements_send_right, const SEM::Device::Entities::Face2D_t* faces, bool* faces_to_delete) -> void;

    __global__
    auto move_faces(size_t n_faces, SEM::Device::Entities::Face2D_t* faces, SEM::Device::Entities::Face2D_t* new_faces) -> void;

    __global__
    auto move_required_faces(size_t n_faces, SEM::Device::Entities::Face2D_t* faces, SEM::Device::Entities::Face2D_t* new_faces, SEM::Device::Entities::Element2D_t* elements, const bool* faces_to_delete, const size_t* faces_to_delete_block_offsets) -> void;

    __global__
    auto find_boundary_elements_to_delete(size_t n_boundary_elements, size_t n_domain_elements, size_t n_elements_send_left, size_t n_elements_send_right, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, bool* boundary_elements_to_delete) -> void;

    __global__
    auto find_mpi_interface_elements_to_delete(size_t n_mpi_interface_elements, size_t n_domain_elements, int rank, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, bool* boundary_elements_to_delete) -> void;
    
    __global__
    auto find_mpi_interface_elements_to_keep(size_t n, size_t n_incoming, int rank, size_t n_domain_elements, const int* neighbour_procs, const size_t* neighbour_indices, const size_t* neighbour_sides, const int* incoming_procs, const size_t* incoming_indices, const size_t* incoming_sides, const size_t* mpi_interfaces_destination, bool* boundary_elements_to_delete) -> void;

    __global__
    auto move_boundary_elements(size_t n_boundary_elements, size_t n_domain_elements, size_t new_n_domain_elements, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, SEM::Device::Entities::Face2D_t* faces, const bool* boundary_elements_to_delete, const size_t* boundary_elements_to_delete_block_offsets) -> void;

    __global__
    auto find_boundaries_to_delete(size_t n_boundary_elements, size_t n_domain_elements, const size_t* boundary, const bool* boundary_elements_to_delete, bool* boundaries_to_delete) -> void;

    __global__
    auto move_all_boundaries(size_t n_boundary_elements, size_t n_domain_elements, size_t new_n_domain_elements, const size_t* boundary, size_t* new_boundary, const bool* boundary_elements_to_delete, const size_t* boundary_elements_block_offsets, int boundary_elements_blockSize) -> void;

    __global__
    auto move_required_boundaries(size_t n_boundary_elements, size_t n_domain_elements, size_t new_n_domain_elements, const size_t* boundary, size_t* new_boundary, const bool* boundaries_to_delete, const size_t* boundaries_to_delete_block_offsets, const bool* boundary_elements_to_delete, const size_t* boundary_elements_block_offsets, int boundary_elements_blockSize) -> void;

    __global__
    auto find_mpi_origins_to_delete(size_t n_mpi_origins, size_t n_domain_elements, size_t n_elements_send_left, size_t n_elements_send_right, const size_t* mpi_interfaces_origin, bool* mpi_origins_to_delete) -> void;
    
    __global__
    auto find_obstructed_mpi_origins_to_delete(size_t n_mpi_origins, size_t n_domain_elements, size_t n_elements_send_left, size_t n_elements_recv_left, size_t n_mpi_destinations, const size_t* mpi_interfaces_origin, const size_t* mpi_interfaces_origin_side, const SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, const int* mpi_origins_process, bool* mpi_origins_to_delete, size_t n_boundary_elements_old, const bool* boundary_elements_to_delete) -> void;

    __global__
    auto move_required_mpi_origins(size_t n_mpi_origins, size_t n_domain_elements, size_t new_n_domain_elements, const size_t* mpi_origins, size_t* new_mpi_origins, const size_t* mpi_origins_side, size_t* new_mpi_origins_side, const bool* mpi_origins_to_delete, const size_t* mpi_origins_to_delete_block_offsets, const bool* boundary_elements_to_delete, const size_t* boundary_elements_block_offsets, int boundary_elements_blockSize) -> void;
    
    __global__
    auto create_sent_mpi_boundaries_destinations(size_t n_sent_elements, size_t sent_elements_offset, size_t new_elements_offset, size_t mpi_interfaces_destination_offset, SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Element2D_t* new_elements, SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* elements_send_destinations_offset, const int* elements_send_destinations_keep, size_t* mpi_interfaces_destination, const deviceFloat* polynomial_nodes) -> void;
    
    __global__
    auto recv_mpi_boundaries_destinations_reuse_faces(size_t n_mpi_interfaces_incoming, size_t n_domain_elements, size_t n_elements_recv_left, size_t n_elements_recv_right, int rank, const SEM::Device::Entities::Element2D_t* elements, SEM::Device::Entities::Face2D_t* new_faces, const size_t* mpi_interfaces_incoming, const int* mpi_interfaces_new_process_incoming, const size_t* mpi_interfaces_new_local_index_incoming, const size_t* mpi_interfaces_new_side_incoming_device) -> void;

    __global__
    auto create_received_neighbours(size_t n_neighbours, int rank, size_t n_mpi_destinations, size_t elements_offset, size_t wall_offset, size_t symmetry_offset, size_t inflow_offset, size_t outflow_offset, size_t mpi_destinations_offset, size_t n_new_wall, size_t n_new_symmetry, size_t n_new_inflow, size_t n_new_outflow, size_t n_new_mpi_destinations, const size_t* neighbour_indices, const int* neighbour_procs, const size_t* neighbour_sides, const int* neighbour_N, const size_t* neighbour_node_indices, const size_t* mpi_destinations_indices, const int* mpi_destinations_procs, const size_t* mpi_destinations_sides, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Vec2<deviceFloat>* nodes, const size_t* old_mpi_destinations, size_t* neighbour_given_indices, size_t* neighbour_given_sides, size_t* wall_boundaries, size_t* symmetry_boundaries, size_t* inflow_boundaries, size_t* outflow_boundaries, size_t* mpi_destinations, const size_t* wall_block_offsets, const size_t* symmetry_block_offsets, const size_t* inflow_block_offsets, const size_t* outflow_block_offsets, const size_t* mpi_destinations_block_offsets, const deviceFloat* polynomial_nodes) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    auto warp_reduce_2D(volatile size_t *sdata, unsigned int tid) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    auto warp_reduce_2D(volatile size_t *sdata_0, volatile size_t *sdata_1, volatile size_t *sdata_2, volatile size_t *sdata_3, volatile size_t *sdata_4, unsigned int tid) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_refine_2D(size_t n_elements, int max_split_level, const SEM::Device::Entities::Element2D_t* elements, size_t* g_odata) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_faces_refine_2D(size_t n_faces, int max_split_level, SEM::Device::Entities::Face2D_t* faces, const SEM::Device::Entities::Element2D_t* elements, size_t* g_odata) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_boundaries_refine_2D(size_t n_boundaries, const SEM::Device::Entities::Element2D_t* elements, const size_t* boundaries, const SEM::Device::Entities::Face2D_t* faces, size_t* g_odata) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_interfaces_refine_2D(size_t n_local_interfaces, int max_split_level, const SEM::Device::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, size_t* g_odata) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_bools(size_t n, const bool* data, size_t* g_odata) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_received_neighbours(size_t n, size_t n_incoming, int rank, const int* neighbour_procs, const size_t* neighbour_indices, const size_t* neighbour_sides, const int* incoming_procs, const size_t* incoming_indices, const size_t* incoming_sides, size_t* wall_odata, size_t* symmetry_odata, size_t* inflow_odata, size_t* outflow_odata, size_t* mpi_destinations_odata) -> void;
}}}

#include "meshes/Mesh2D_t.tcu"

#endif
