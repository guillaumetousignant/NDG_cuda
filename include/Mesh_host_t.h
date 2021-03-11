#ifndef NDG_MESH_HOST_T_H
#define NDG_MESH_HOST_T_H

#include "float_types.h"
#include "Element_host_t.h"
#include "Face_host_t.h"
#include "NDG_host_t.h"
#include <vector>
#include <mpi.h>
#include <array>

namespace SEM {
    class Mesh_host_t {
        public:
            Mesh_host_t(size_t N_elements, int initial_N, hostFloat x_min, hostFloat x_max);
            ~Mesh_host_t();

            size_t N_elements_global_;
            size_t N_elements_;
            size_t N_local_boundaries_;
            size_t N_MPI_boundaries_;
            size_t global_element_offset_;
            size_t N_elements_per_process_;
            int initial_N_;
            std::vector<Element_host_t> elements_;
            std::vector<Face_host_t> faces_;
            std::vector<size_t> local_boundary_to_element_;
            std::vector<size_t> MPI_boundary_to_element_;
            std::vector<size_t> MPI_boundary_from_element_;

            void set_initial_conditions(const std::vector<std::vector<hostFloat>>& nodes);
            void print();
            void write_data(hostFloat time, size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices);
            
            template<typename Polynomial>
            void solve(hostFloat delta_t, const std::vector<hostFloat> output_times, const NDG_host_t<Polynomial> &NDG);

            void build_elements(hostFloat x_min, hostFloat x_max);
            void build_boundaries(hostFloat x_min, hostFloat x_max);
            void build_faces();
            void get_solution(size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<hostFloat>& phi, std::vector<hostFloat>& x, std::vector<hostFloat>& phi_prime, std::vector<hostFloat>& intermediate, std::vector<hostFloat>& x_L, std::vector<hostFloat>& x_R, std::vector<int>& N, std::vector<hostFloat>& sigma, std::vector<bool>& refine, std::vector<bool>& coarsen, std::vector<hostFloat>& error);

        private:
            std::vector<std::array<double, 4>> send_buffers_;
            std::vector<std::array<double, 4>> receive_buffers_;
            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;

            static hostFloat g(hostFloat x);
            void interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);
            static void write_file_data(size_t N_interpolation_points, size_t N_elements, hostFloat time, int rank, const std::vector<hostFloat>& velocity, const std::vector<hostFloat>& coordinates, const std::vector<hostFloat>& du_dx, const std::vector<hostFloat>& intermediate, const std::vector<hostFloat>& x_L, const std::vector<hostFloat>& x_R, const std::vector<int>& N, const std::vector<hostFloat>& sigma, const std::vector<bool>& refine, const std::vector<bool>& coarsen, const std::vector<hostFloat>& error);
            hostFloat get_delta_t(const hostFloat CFL);

            void calculate_fluxes();

            // Algorithm 60 (not really anymore)
            void compute_dg_derivative(const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

            void rk3_step(hostFloat delta_t, hostFloat a, hostFloat g);
    };

    // Algorithm 19
    void matrix_vector_derivative(const std::vector<hostFloat>& derivative_matrices_hat, const std::vector<hostFloat>& phi, std::vector<hostFloat>& phi_prime);
}

#endif