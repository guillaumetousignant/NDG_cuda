#ifndef NDG_MESH_HOST_T_H
#define NDG_MESH_HOST_T_H

#include "float_types.h"
#include "Element_t.cuh"
#include "Face_t.cuh"
#include "NDG_t.cuh"
#include <vector>

class Mesh_host_t {
    public:
        Mesh_host_t(int N_elements, int initial_N, hostFloat x_min, hostFloat x_max);
        ~Mesh_host_t();

        int initial_N_;
        std::vector<Element_t> elements_;
        std::vector<Face_t> faces_;

        void set_initial_conditions(const std::vector<hostFloat>& nodes);
        void print();
        void write_data(hostFloat time, int N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices);
        
        template<typename Polynomial>
        void solve(const float delta_t, const std::vector<float> output_times, const NDG_t<Polynomial> &NDG);

        void build_elements(hostFloat x_min, hostFloat x_max);
        void build_faces();
        void get_solution(size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<hostFloat>& phi, std::vector<hostFloat>& x);

    private:
        static hostFloat g(hostFloat x);
        void interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);
        static void write_file_data(hostFloat time, const std::vector<hostFloat>& velocity, const std::vector<hostFloat>& coordinates);
};

namespace SEM {
    void rk3_step(int N_elements, Element_t* elements, float delta_t, float a, float g);

    void calculate_fluxes(int N_faces, Face_t* faces, const Element_t* elements);

    // Algorithm 19
    void matrix_vector_derivative(int N, const float* derivative_matrices_hat, const float* phi, float* phi_prime);

    // Algorithm 60 (not really anymore)
    void compute_dg_derivative(int N_elements, Element_t* elements, const Face_t* faces, const float* weights, const float* derivative_matrices_hat, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right);
}

#endif