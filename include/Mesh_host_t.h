#ifndef NDG_MESH_HOST_T_H
#define NDG_MESH_HOST_T_H

#include "float_types.h"
#include "Element_host_t.h"
#include "Face_host_t.h"
#include "NDG_host_t.h"
#include <vector>

class Mesh_host_t {
    public:
        Mesh_host_t(size_t N_elements, int initial_N, hostFloat x_min, hostFloat x_max);
        ~Mesh_host_t();

        int initial_N_;
        std::vector<Element_host_t> elements_;
        std::vector<Face_host_t> faces_;

        void set_initial_conditions(const std::vector<std::vector<hostFloat>>& nodes);
        void print();
        void write_data(hostFloat time, size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices);
        
        template<typename Polynomial>
        void solve(hostFloat delta_t, const std::vector<hostFloat> output_times, const NDG_host_t<Polynomial> &NDG);

        void build_elements(hostFloat x_min, hostFloat x_max);
        void build_faces();
        void get_solution(size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<hostFloat>& phi, std::vector<hostFloat>& x);

    private:
        static hostFloat g(hostFloat x);
        void interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);
        static void write_file_data(hostFloat time, const std::vector<hostFloat>& velocity, const std::vector<hostFloat>& coordinates);

        void calculate_fluxes();

        // Algorithm 60 (not really anymore)
        void compute_dg_derivative(const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

        void rk3_step(hostFloat delta_t, hostFloat a, hostFloat g);
};

namespace SEM {
    // Algorithm 19
    void matrix_vector_derivative(const std::vector<hostFloat>& derivative_matrices_hat, const std::vector<hostFloat>& phi, std::vector<hostFloat>& phi_prime);
}

#endif