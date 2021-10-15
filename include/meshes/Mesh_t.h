#ifndef NDG_MESHES_MESH_T_H
#define NDG_MESHES_MESH_T_H

#include "helpers/float_types.h"
#include "entities/Element_t.h"
#include "entities/Face_t.h"
#include "entities/NDG_t.h"
#include <vector>
#include <mpi.h>
#include <array>

namespace SEM { namespace Host { namespace Meshes {
    class Mesh_t {
        public:
            Mesh_t(size_t N_elements, int initial_N, hostFloat delta_x_min, hostFloat x_min, hostFloat x_max, int adaptivity_interval);

            size_t N_elements_global_;
            size_t N_elements_;
            size_t N_local_boundaries_;
            size_t N_MPI_boundaries_;
            size_t global_element_offset_;
            size_t N_elements_per_process_;
            int initial_N_;
            hostFloat delta_x_min_;
            int adaptivity_interval_;
            std::vector<SEM::Host::Entities::Element_t> elements_;
            std::vector<SEM::Host::Entities::Face_t> faces_;
            std::vector<size_t> local_boundary_to_element_;
            std::vector<size_t> MPI_boundary_to_element_;
            std::vector<size_t> MPI_boundary_from_element_;

            void set_initial_conditions(const std::vector<std::vector<hostFloat>>& nodes);
            void boundary_conditions();
            void print();
            void write_data(hostFloat time, size_t n_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices);
            hostFloat get_delta_t(const hostFloat CFL);
            
            template<typename Polynomial>
            void solve(const hostFloat CFL, const std::vector<hostFloat> output_times, const SEM::Host::Entities::NDG_t<Polynomial> &NDG, hostFloat viscosity);

            void build_elements(hostFloat x_min, hostFloat x_max);
            void build_boundaries();
            void adjust_boundaries();
            void build_faces();
            void get_solution(size_t n_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<hostFloat>& phi, std::vector<hostFloat>& x, std::vector<hostFloat>& phi_prime, std::vector<hostFloat>& intermediate, std::vector<hostFloat>& x_L, std::vector<hostFloat>& x_R, std::vector<int>& N, std::vector<hostFloat>& sigma, std::vector<bool>& refine, std::vector<bool>& coarsen, std::vector<hostFloat>& error);

        private:
            std::vector<std::array<double, 4>> send_buffers_;
            std::vector<std::array<double, 4>> receive_buffers_;
            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;
            std::vector<size_t> refine_array_;

            static hostFloat g(hostFloat x);

            void interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

            void interpolate_q_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);
            
            static void write_file_data(size_t n_interpolation_points, size_t N_elements, hostFloat time, int rank, const std::vector<hostFloat>& velocity, const std::vector<hostFloat>& coordinates, const std::vector<hostFloat>& du_dx, const std::vector<hostFloat>& intermediate, const std::vector<hostFloat>& x_L, const std::vector<hostFloat>& x_R, const std::vector<int>& N, const std::vector<hostFloat>& sigma, const std::vector<bool>& refine, const std::vector<bool>& coarsen, const std::vector<hostFloat>& error);

            void local_boundaries();

            void get_MPI_boundaries();

            void put_MPI_boundaries();

            void move_elements(size_t N_elements, std::vector<SEM::Host::Entities::Element_t>& temp_elements, size_t source_start_index, size_t destination_start_index);

            void calculate_fluxes();

            void calculate_q_fluxes();

            void adapt(int N_max, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights);

            void p_adapt(int N_max, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights);

            void hp_adapt(int N_max, std::vector<SEM::Host::Entities::Element_t>& new_elements, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights);

            // Algorithm 60 (not really anymore)
            void compute_dg_derivative(const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

            // Algorithm 60 (not really anymore)
            void compute_dg_derivative2(hostFloat viscosity, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

            void rk3_first_step(hostFloat delta_t, hostFloat g);

            void rk3_step(hostFloat delta_t, hostFloat a, hostFloat g);

            template<typename Polynomial>
            void estimate_error(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);
    };

    void matrix_vector_multiply(int N, const std::vector<hostFloat>& matrix, const std::vector<hostFloat>& vector, std::vector<hostFloat>& result);

    // Algorithm 19
    void matrix_vector_derivative(int N, const std::vector<hostFloat>& derivative_matrices_hat,  const std::vector<hostFloat>& phi, std::vector<hostFloat>& phi_prime);
}}}

#endif
