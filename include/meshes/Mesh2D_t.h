#ifndef NDG_MESHES_MESH2D_T_H
#define NDG_MESHES_MESH2D_T_H

#include "entities/Element2D_t.h"
#include "entities/Face2D_t.h"
#include "entities/Vec2.h"
#include "helpers/float_types.h"
#include "helpers/DataWriter_t.h"
#include <array>
#include <tuple>
#include <vector>
#include <limits>
#include <mpi.h>
#include <utility>
#include <filesystem>
#include <memory>

namespace SEM { namespace Host { namespace Meshes {
    class Mesh2D_t {
        public:
            Mesh2D_t(std::filesystem::path filename, int initial_N, int maximum_N, size_t n_interpolation_points, int max_split_level, size_t adaptivity_interval, size_t load_balancing_interval, hostFloat tolerance_min, hostFloat tolerance_max, hostFloat load_balancing_threshold, const std::vector<std::vector<hostFloat>>& polynomial_nodes);

            // Geometry
            std::vector<SEM::Host::Entities::Vec2<hostFloat>> nodes_;
            std::vector<SEM::Host::Entities::Element2D_t> elements_;
            std::vector<SEM::Host::Entities::Face2D_t> faces_;

            // Boundaries
            std::vector<size_t> wall_boundaries_;
            std::vector<size_t> symmetry_boundaries_;
            std::vector<size_t> inflow_boundaries_;
            std::vector<size_t> outflow_boundaries_;
            enum boundary_type : int {wall = -1, symmetry = -2, inflow = -3, outflow = -4};

            // Interfaces
            std::vector<size_t> interfaces_origin_;
            std::vector<size_t> interfaces_origin_side_;
            std::vector<size_t> interfaces_destination_;
            std::vector<size_t> mpi_interfaces_outgoing_size_;  // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_incoming_size_;  // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_outgoing_offset_; // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_incoming_offset_; // Those are only needed on the CPU... right?
            std::vector<int> mpi_interfaces_process_; // Those are only needed on the CPU... right?
            std::vector<size_t> mpi_interfaces_origin_;
            std::vector<size_t> mpi_interfaces_origin_side_;
            std::vector<size_t> mpi_interfaces_destination_;

            // Boundary solution exchange
            std::vector<int> interfaces_N_;
            std::vector<hostFloat> interfaces_p_;
            std::vector<hostFloat> interfaces_u_;
            std::vector<hostFloat> interfaces_v_;
            std::vector<hostFloat> receiving_interfaces_p_;
            std::vector<hostFloat> receiving_interfaces_u_;
            std::vector<hostFloat> receiving_interfaces_v_;
            std::vector<int> receiving_interfaces_N_;
            std::unique_ptr<bool[]> interfaces_refine_;
            std::unique_ptr<bool[]> receiving_interfaces_refine_;
            std::unique_ptr<bool[]> receiving_interfaces_refine_without_splitting_;
            std::unique_ptr<bool[]> receiving_interfaces_creating_node_;
            std::unique_ptr<bool[]> interfaces_refine_without_splitting_;

            // Output
            std::vector<hostFloat> x_output_;
            std::vector<hostFloat> y_output_;
            std::vector<hostFloat> p_output_;
            std::vector<hostFloat> u_output_;
            std::vector<hostFloat> v_output_;

            // MPI offsets
            constexpr static size_t mpi_build_n_transfers = 1;
            constexpr static size_t mpi_boundaries_offset = mpi_build_n_transfers;
            constexpr static size_t mpi_boundaries_n_transfers = 3;
            constexpr static size_t mpi_adaptivity_split_offset = mpi_boundaries_offset + mpi_boundaries_n_transfers;
            constexpr static size_t mpi_adaptivity_split_n_transfers = 2;
            constexpr static size_t mpi_load_balancing_interfaces_offset = mpi_adaptivity_split_offset + mpi_adaptivity_split_n_transfers;
            constexpr static size_t mpi_load_balancing_interfaces_n_transfers = 3;
            constexpr static size_t mpi_load_balancing_n_neighbours_offset = mpi_load_balancing_interfaces_offset + mpi_load_balancing_interfaces_n_transfers;
            constexpr static size_t mpi_load_balancing_n_neighbours_n_transfers = 1;
            constexpr static size_t mpi_load_balancing_solution_offset = mpi_load_balancing_n_neighbours_offset + mpi_load_balancing_n_neighbours_n_transfers;
            constexpr static size_t mpi_load_balancing_solution_n_transfers = 9;
            constexpr static size_t mpi_load_balancing_incoming_offset = mpi_load_balancing_solution_offset + mpi_load_balancing_solution_n_transfers;
            constexpr static size_t mpi_load_balancing_incoming_n_transfers = 1;
            constexpr static size_t mpi_load_balancing_origins_offset = mpi_load_balancing_incoming_offset + mpi_load_balancing_incoming_n_transfers;
            constexpr static size_t mpi_load_balancing_origins_n_transfers = 2;
            
            // Counts
            size_t n_elements_global_;
            size_t n_elements_;
            size_t global_element_offset_;

            // Parameters
            int initial_N_;
            int maximum_N_;
            size_t n_interpolation_points_;
            int max_split_level_;
            size_t adaptivity_interval_;
            size_t load_balancing_interval_;
            hostFloat tolerance_min_;
            hostFloat tolerance_max_;
            hostFloat load_balancing_threshold_;

            auto read_su2(std::filesystem::path filename) -> void;
            auto read_cgns(std::filesystem::path filename) -> void;
            auto initial_conditions(const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void;
            auto boundary_conditions(hostFloat t, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;
            auto interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) -> void;
            auto project_to_faces(const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;
            auto project_to_elements(const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;
            
            auto estimate_error(const std::vector<std::vector<hostFloat>>& polynomials) -> void;
            
            auto print() const -> void;
            auto print_to_file(std::filesystem::path filename) const -> void;
            auto write_data(hostFloat time, const std::vector<std::vector<hostFloat>>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void;
            auto write_complete_data(hostFloat time, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void;

            auto adapt(int N_max, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;
            
            auto load_balance(const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void;

            // From cppreference.com
            static auto almost_equal(hostFloat x, hostFloat y) -> bool;

            auto get_solution(const std::vector<std::vector<hostFloat>>& interpolation_matrices) -> void;
            auto get_complete_solution(hostFloat time, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<int>& N, std::vector<hostFloat>& dp_dt, std::vector<hostFloat>& du_dt, std::vector<hostFloat>& dv_dt, std::vector<hostFloat>& p_error, std::vector<hostFloat>& u_error, std::vector<hostFloat>& v_error, std::vector<hostFloat>& p_sigma, std::vector<hostFloat>& u_sigma, std::vector<hostFloat>& v_sigma, std::vector<int>& refine, std::vector<int>& coarsen, std::vector<int>& split_level, std::vector<hostFloat>& p_analytical_error, std::vector<hostFloat>& u_analytical_error, std::vector<hostFloat>& v_analytical_error, std::vector<int>& status, std::vector<int>& rotation) -> void;

        private:
            // MPI exchange variables
            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;
            std::vector<MPI_Request> requests_adaptivity_;
            std::vector<MPI_Status> statuses_adaptivity_;

            static auto build_node_to_element(size_t n_nodes, const std::vector<SEM::Host::Entities::Element2D_t>& elements) -> std::vector<std::vector<size_t>>;
            static auto build_element_to_element(const std::vector<SEM::Host::Entities::Element2D_t>& elements, const std::vector<std::vector<size_t>>& node_to_element) -> std::vector<std::vector<size_t>>;
            static auto build_faces(size_t n_elements_domain, size_t n_nodes, int initial_N, std::vector<SEM::Host::Entities::Element2D_t>& elements) -> std::tuple<std::vector<SEM::Host::Entities::Face2D_t>, std::vector<std::vector<size_t>>, std::vector<std::array<size_t, 4>>>;
    };

    auto allocate_element_storage(size_t n_elements, std::vector<SEM::Host::Entities::Element2D_t>& elements) -> void;

    auto allocate_boundary_storage(size_t n_domain_elements, std::vector<SEM::Host::Entities::Element2D_t>& elements) -> void;

    auto compute_element_geometry(size_t n_elements, std::vector<SEM::Host::Entities::Element2D_t>& elements, const std::vector<SEM::Host::Entities::Vec2<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void;
    
    auto compute_boundary_geometry(size_t n_domain_elements, std::vector<SEM::Host::Entities::Element2D_t>& elements, const std::vector<SEM::Host::Entities::Vec2<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void;

    auto compute_element_status(size_t n_elements, std::vector<SEM::Host::Entities::Element2D_t>& elements, const std::vector<SEM::Host::Entities::Vec2<hostFloat>>& nodes) -> void;
    
    auto allocate_face_storage(std::vector<SEM::Host::Entities::Face2D_t>& faces) -> void;

    auto fill_element_faces(size_t n_elements, std::vector<SEM::Host::Entities::Element2D_t>& elements, const std::vector<std::array<size_t, 4>> element_to_face) -> void;

    auto fill_boundary_element_faces(size_t n_domain_elements, std::vector<SEM::Host::Entities::Element2D_t>& elements, const std::vector<std::array<size_t, 4>>& element_to_face) -> void;

    auto compute_face_geometry(std::vector<SEM::Host::Entities::Face2D_t>& faces, const std::vector<SEM::Host::Entities::Element2D_t>& elements, const std::vector<SEM::Host::Entities::Vec2<hostFloat>>& nodes) -> void;

    auto compute_wall_boundaries(size_t n_wall_boundaries, SEM::Host::Entities::Element2D_t* elements, const size_t* wall_boundaries, const SEM::Host::Entities::Face2D_t* faces, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;

    auto compute_symmetry_boundaries(size_t n_symmetry_boundaries, SEM::Host::Entities::Element2D_t* elements, const size_t* symmetry_boundaries, const SEM::Host::Entities::Face2D_t* faces, const std::vector<std::vector<hostFloat>>& polynomial_node, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;

    auto compute_inflow_boundaries(size_t n_inflow_boundaries, SEM::Host::Entities::Element2D_t* elements, const size_t* inflow_boundaries, const SEM::Host::Entities::Face2D_t* faces, hostFloat t, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void;

    auto compute_outflow_boundaries(size_t n_outflow_boundaries, SEM::Host::Entities::Element2D_t* elements, const size_t* outflow_boundaries, const SEM::Host::Entities::Face2D_t* faces, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;

    auto local_interfaces(size_t n_local_interfaces, SEM::Host::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void;

    auto get_MPI_interfaces(size_t n_MPI_interface_elements, const SEM::Host::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, const size_t* MPI_interfaces_origin_side, int maximum_N, hostFloat* p, hostFloat* u, hostFloat* v) -> void;

    auto get_MPI_interfaces_N(size_t n_MPI_interface_elements, int N_max, const SEM::Host::Entities::Element2D_t* elements, const size_t* MPI_interfaces_origin, int* N) -> void;

    auto get_MPI_interfaces_adaptivity(size_t n_MPI_interface_elements, size_t n_domain_elements, size_t n_processes, const SEM::Host::Entities::Element2D_t* elements,  const SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const size_t* MPI_interfaces_origin, const size_t* MPI_interfaces_origin_side, const size_t* MPI_interfaces_destination, const int* MPI_process, const size_t* origin_process_size, const size_t* destination_process_size, const size_t* destination_process_offset, int* N, bool* elements_splitting, bool* elements_refining_without_splitting, int max_split_level, int N_max) -> void;

    auto put_MPI_interfaces(size_t n_MPI_interface_elements, SEM::Host::Entities::Element2D_t* elements, const size_t* MPI_interfaces_destination, int maximum_N, const hostFloat* p, const hostFloat* u, const hostFloat* v) -> void;

    auto adjust_MPI_incoming_interfaces(size_t n_MPI_interface_elements, size_t nodes_offset, SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, const size_t* MPI_interfaces_destination, const int* N, SEM::Host::Entities::Vec2<hostFloat>* nodes, const bool* refine, const bool* refine_without_splitting, const bool* creating_node, const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void;
    
    auto p_adapt(size_t n_elements, SEM::Host::Entities::Element2D_t* elements, int N_max, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void;
    
    auto p_adapt_move(size_t n_elements, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, int N_max, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights, size_t* elements_new_indices) -> void;
    
    auto p_adapt_split_faces(size_t n_elements, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const SEM::Host::Entities::Face2D_t* faces, int N_max, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights, size_t* elements_new_indices) -> void;
    
    auto hp_adapt(size_t n_elements, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const SEM::Host::Entities::Face2D_t* faces, SEM::Host::Entities::Face2D_t* new_faces, int max_split_level, int N_max, SEM::Host::Entities::Vec2<hostFloat>* nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights, size_t* elements_new_indices) -> void;
    
    auto split_faces(size_t n_faces, size_t n_nodes, size_t n_splitting_elements, SEM::Host::Entities::Face2D_t* faces, SEM::Host::Entities::Face2D_t* new_faces, const SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Vec2<hostFloat>* nodes, int max_split_level, int N_max, const size_t* elements_new_indices) -> void;

    auto find_nodes(size_t n_elements, SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, int max_split_level) -> void;
    
    auto no_new_nodes(size_t n_elements, SEM::Host::Entities::Element2D_t* elements) -> void;

    auto copy_boundaries_error(size_t n_boundaries, SEM::Host::Entities::Element2D_t* elements, const size_t* boundaries, const SEM::Host::Entities::Face2D_t* faces) -> void;

    auto copy_interfaces_error(size_t n_local_interfaces, SEM::Host::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void;
    
    auto copy_mpi_interfaces_error(size_t n_MPI_interface_elements, SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const size_t* MPI_interfaces_destination, const int* N, const bool* elements_splitting, bool* elements_refining_without_splitting, bool* elements_creating_node) -> void;

    auto split_boundaries(size_t n_boundaries, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, size_t offset, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const size_t* boundaries, size_t* new_boundaries, const SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes, size_t* elements_new_indices) -> void;

    auto split_interfaces(size_t n_local_interfaces, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, size_t offset, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination, size_t* new_local_interfaces_origin, size_t* new_local_interfaces_origin_side, size_t* new_local_interfaces_destination, const SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, int max_split_level, int N_max, const std::vector<std::vector<hostFloat>>& polynomial_nodes, size_t* elements_new_indices) -> void;
    
    auto split_mpi_outgoing_interfaces(size_t n_MPI_interface_elements, size_t n_domain_elements, size_t n_processes, const SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, SEM::Host::Entities::Vec2<hostFloat>* nodes, const size_t* mpi_interfaces_origin, const size_t* mpi_interfaces_origin_side, const size_t* MPI_interfaces_destination, size_t* new_mpi_interfaces_origin, size_t* new_mpi_interfaces_origin_side, const int* MPI_process, const size_t* origin_process_size, const size_t* destination_process_size, const size_t* destination_process_offset, const bool* elements_splitting, const bool* elements_refining_without_splitting, int max_split_level) -> void;
    
    auto split_mpi_incoming_interfaces(size_t n_MPI_interface_elements, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, size_t n_splitting_faces, size_t offset, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const size_t* mpi_interfaces_destination, size_t* new_mpi_interfaces_destination, const SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const int* N, const bool* elements_splitting, const bool* elements_refining_without_splitting, const bool* elements_creating_node, size_t* elements_new_indices) -> void;

    auto adjust_boundaries(size_t n_boundaries, SEM::Host::Entities::Element2D_t* elements, const size_t* boundaries, const SEM::Host::Entities::Face2D_t* faces) -> void;

    auto adjust_interfaces(size_t n_local_interfaces, SEM::Host::Entities::Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination) -> void;

    auto adjust_faces(size_t n_faces, SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Element2D_t* elements) -> void;
    
    auto adjust_faces_neighbours(size_t n_faces, SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Vec2<hostFloat>* nodes, int max_split_level, int N_max, const size_t* elements_new_indices) -> void;

    auto move_boundaries(size_t n_boundaries, size_t offset, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, size_t* boundaries, const SEM::Host::Entities::Face2D_t* faces, size_t* elements_new_indices) -> void;

    auto move_interfaces(size_t n_local_interfaces, size_t offset, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination, size_t* elements_new_indices) -> void;

    auto get_transfer_solution(size_t n_elements, const SEM::Host::Entities::Element2D_t* elements, int maximum_N, const SEM::Host::Entities::Vec2<hostFloat>* nodes, hostFloat* solution, size_t* n_neighbours, hostFloat* element_nodes) -> void;
    
    auto fill_received_elements(size_t n_elements, size_t element_offset, SEM::Host::Entities::Element2D_t* elements, int maximum_N, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const hostFloat* solution, const size_t* n_neighbours, const size_t* received_node_indices, const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void;

    auto fill_received_elements_faces(size_t n_elements, size_t element_offset, size_t face_offset, size_t n_domain_elements, size_t n_elements_recv_left, size_t n_elements_recv_right, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const size_t* n_neighbours, const size_t* n_neighbours_left, const size_t* n_neighbours_right, const size_t* neighbours_offsets, const size_t* neighbours_offsets_left, const size_t* neighbours_offsets_right, const size_t* neighbours_indices, const size_t* neighbours_sides, const int* neighbours_procs, const size_t* face_offsets, const size_t* face_offsets_left, const size_t* face_offsets_right) -> void;

    auto get_neighbours(size_t n_elements_send, size_t start_index, size_t n_domain_elements_old, size_t n_wall_boundaries, size_t n_symmetry_boundaries, size_t n_inflow_boundaries, size_t n_outflow_boundaries, size_t n_local_interfaces, size_t n_MPI_interface_elements_receiving, int rank, int n_procs, const SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const size_t* wall_boundaries, const size_t* symmetry_boundaries, const size_t* inflow_boundaries, const size_t* outflow_boundaries, const size_t* interfaces_destination, const size_t* interfaces_origin, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_process, const size_t* mpi_interfaces_local_indices, const size_t* mpi_interfaces_side, const size_t* offsets, const size_t* n_elements_received_left, const size_t* n_elements_sent_left, const size_t* n_elements_received_right, const size_t* n_elements_sent_right, const size_t* global_element_offset, const size_t* global_element_offset_new, const size_t* global_element_offset_end_new, size_t* neighbours, hostFloat* neighbours_nodes, int* neighbours_proc, size_t* neighbours_side, int* neighbours_N) -> void;

    auto move_elements(size_t n_elements_move, size_t n_elements_send_left, size_t n_elements_recv_left, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const bool* faces_to_delete, const size_t* faces_to_delete_block_offsets) -> void;

    auto get_interface_n_processes(size_t n_mpi_interfaces, size_t n_mpi_interfaces_incoming, const SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, const size_t* mpi_interfaces_origin, const size_t* mpi_interfaces_origin_side, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, size_t* n_processes) -> void;
    
    auto get_interface_processes(size_t n_mpi_interfaces, size_t n_mpi_interfaces_incoming, const SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, const size_t* mpi_interfaces_origin, const size_t* mpi_interfaces_origin_side, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, const size_t* process_offsets, int* processes) -> void;
    
    auto find_received_nodes(size_t n_nodes, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const hostFloat* received_nodes, std::vector<bool>& missing_nodes, std::vector<bool>& missing_received_nodes, size_t* received_nodes_indices, size_t* received_node_received_indices) -> void;

    auto add_new_received_nodes(const std::vector<bool>& missing_nodes, const std::vector<bool>& missing_received_nodes, size_t n_nodes, SEM::Host::Entities::Vec2<hostFloat>* nodes, const hostFloat* received_nodes, size_t* received_nodes_indices, const size_t* received_node_received_indices, const size_t* received_nodes_block_offsets) -> void;

    auto find_received_neighbour_nodes(size_t n_received_neighbour_nodes, size_t n_received_nodes, size_t n_nodes, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const hostFloat* received_neighbour_nodes, const hostFloat* received_nodes, bool* missing_neighbour_nodes, bool* missing_received_neighbour_nodes, size_t* received_neighbour_nodes_indices, size_t* received_neighbour_node_received_indices) -> void;
  
    auto add_new_received_neighbour_nodes(size_t n_received_neighbour_nodes, size_t n_received_nodes, size_t n_nodes, SEM::Host::Entities::Vec2<hostFloat>* nodes, const hostFloat* received_neighbour_nodes, const bool* missing_neighbour_nodes, const bool* missing_nodes, const bool* missing_received_neighbour_nodes, size_t* received_neighbour_nodes_indices, const size_t* received_neighbour_node_received_indices, const size_t* received_neighbour_nodes_block_offsets, const size_t* received_nodes_block_offsets) -> void;

    auto find_faces_to_delete(size_t n_domain_elements, size_t n_elements_send_left, size_t n_elements_send_right, const SEM::Host::Entities::Face2D_t* faces, std::vector<bool>& faces_to_delete) -> void;

    auto no_faces_to_delete(std::vector<bool>& faces_to_delete) -> void;

    auto move_faces(size_t n_faces, size_t n_domain_elements, size_t new_n_domain_elements, size_t n_elements_recv_left, size_t n_elements_recv_right, size_t n_mpi_destinations, int rank, SEM::Host::Entities::Face2D_t* faces, SEM::Host::Entities::Face2D_t* new_faces, const std::vector<bool>& boundary_elements_to_delete, const size_t* boundary_elements_to_delete_block_offsets, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, const size_t* mpi_interfaces_new_local_index_incoming, const size_t* mpi_interfaces_new_side_incoming) -> void;

    auto move_required_faces(size_t n_domain_elements, size_t new_n_domain_elements, size_t n_elements_send_left, size_t n_elements_send_right, size_t n_elements_recv_left, size_t n_elements_recv_right, size_t new_elements_offset_left, size_t new_elements_offset_right, size_t n_mpi_destinations, int rank, SEM::Host::Entities::Face2D_t* faces, SEM::Host::Entities::Face2D_t* new_faces, const std::vector<bool>& faces_to_delete, const size_t* faces_to_delete_block_offsets, const std::vector<bool>& boundary_elements_to_delete, const size_t* boundary_elements_to_delete_block_offsets, const size_t* elements_send_destinations_offset_left, const size_t* elements_send_destinations_offset_right, const int* elements_send_destinations_keep_left, const int* elements_send_destinations_keep_right, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, const size_t* mpi_interfaces_new_local_index_incoming, const size_t* mpi_interfaces_new_side_incoming) -> void;

    auto find_boundary_elements_to_delete(size_t n_boundary_elements, size_t n_domain_elements, size_t n_elements_send_left, size_t n_elements_send_right, const SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, std::vector<bool>& boundary_elements_to_delete, const std::vector<bool>& faces_to_delete) -> void;

    auto find_mpi_interface_elements_to_delete(size_t n_mpi_interface_elements, size_t n_domain_elements, int rank, const size_t* mpi_interfaces_destination, const int* mpi_interfaces_new_process_incoming, std::vector<bool>& boundary_elements_to_delete) -> void;

    auto find_mpi_interface_elements_to_keep(size_t n_mpi_destinations, size_t n_neighbours, int rank, size_t n_domain_elements, const int* neighbour_procs, const size_t* neighbour_indices, const size_t* neighbour_sides, const int* mpi_destination_procs, const size_t* mpi_destination_local_indices, const size_t* mpi_destination_sides, const size_t* mpi_interfaces_destination, std::vector<bool>& boundary_elements_to_delete) -> void;

    auto move_boundary_elements(size_t n_boundary_elements, size_t n_domain_elements, size_t new_n_domain_elements, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const std::vector<bool>& boundary_elements_to_delete, const size_t* boundary_elements_to_delete_block_offsets, const std::vector<bool>& faces_to_delete, const size_t* faces_to_delete_block_offsets) -> void;

    auto find_boundaries_to_delete(const std::vector<size_t>& n_boundary_elements, size_t n_domain_elements, const std::vector<bool>& boundary_elements_to_delete, std::vector<bool>& boundaries_to_delete) -> void;

    auto move_all_boundaries(size_t n_domain_elements, size_t new_n_domain_elements, const std::vector<size_t>& boundary, std::vector<size_t>& new_boundary, const std::vector<bool>& boundary_elements_to_delete) -> void;

    auto move_required_boundaries(size_t n_domain_elements, size_t new_n_domain_elements, const std::vector<size_t>& boundary, std::vector<size_t>& new_boundary, const std::vector<bool>& boundaries_to_delete, const std::vector<bool>& boundary_elements_to_delete) -> void;

    auto move_required_boundaries(size_t n_boundary_elements, size_t n_domain_elements, size_t new_n_domain_elements, const size_t* boundary, size_t* new_boundary, int* new_boundary_process, size_t* new_local_index, size_t* new_side, const int* mpi_interfaces_process, const size_t* mpi_interfaces_local_index, const size_t* mpi_interfaces_side, const bool* boundaries_to_delete, const size_t* boundaries_to_delete_block_offsets, const bool* boundary_elements_to_delete, const size_t* boundary_elements_block_offsets, int boundary_elements_blockSize) -> void;

    auto create_sent_mpi_boundaries_destinations(size_t n_sent_elements, size_t sent_elements_offset, size_t new_elements_offset, size_t mpi_interfaces_destination_offset, SEM::Host::Entities::Element2D_t* elements, SEM::Host::Entities::Element2D_t* new_elements, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const size_t* elements_send_destinations_offset, const int* elements_send_destinations_keep, const int* destination_process, const size_t* destination_local_index, size_t* mpi_interfaces_destination, int* mpi_interfaces_destination_process, size_t* mpi_interfaces_destination_local_index, size_t* mpi_interfaces_destination_side, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const bool* faces_to_delete, const size_t* faces_to_delete_block_offsets) -> void;
    
    auto create_received_neighbours(size_t n_neighbours, int rank, size_t n_mpi_destinations, size_t elements_offset, size_t wall_offset, size_t symmetry_offset, size_t inflow_offset, size_t outflow_offset, size_t mpi_destinations_offset, size_t n_new_wall, size_t n_new_symmetry, size_t n_new_inflow, size_t n_new_outflow, size_t n_new_mpi_destinations, size_t n_domain_elements, size_t new_n_domain_elements, const size_t* neighbour_indices, const int* neighbour_procs, const size_t* neighbour_sides, const int* neighbour_N, const size_t* neighbour_node_indices, const size_t* mpi_destinations_indices, const int* mpi_destinations_procs, const size_t* mpi_destinations_sides, SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Vec2<hostFloat>* nodes, const size_t* old_mpi_destinations, size_t* neighbour_given_indices, size_t* neighbour_given_sides, size_t* wall_boundaries, size_t* symmetry_boundaries, size_t* inflow_boundaries, size_t* outflow_boundaries, size_t* mpi_destinations, int* mpi_destinations_process, size_t* mpi_destinations_local_index, size_t* mpi_destinations_side, const size_t* wall_block_offsets, const size_t* symmetry_block_offsets, const size_t* inflow_block_offsets, const size_t* outflow_block_offsets, const size_t* mpi_destinations_block_offsets, const std::vector<std::vector<hostFloat>>& polynomial_nodes, const bool* boundary_elements_to_delete, const size_t* boundary_elements_to_delete_block_offsets) -> void;
   
    auto add_new_faces_to_mpi_destination_elements(size_t n_mpi_destinations, size_t face_offset, size_t n_new_faces, const size_t* mpi_destinations, SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces) -> void;
 
}}}

#endif
