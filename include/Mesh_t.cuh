#ifndef NDG_MESH_T_H
#define NDG_MESH_T_H

#include "Element_t.cuh"
#include "Face_t.cuh"
#include "NDG_t.cuh"
#include "float_types.h"
#include <vector>

class Mesh_t {
public:
    Mesh_t(size_t N_elements, int initial_N, deviceFloat x_min, deviceFloat x_max);
    ~Mesh_t();

    size_t N_elements_;
    size_t N_faces_;
    int initial_N_;
    Element_t* elements_;
    Face_t* faces_;

    void set_initial_conditions(const deviceFloat* nodes);
    void print();
    void write_file_data(size_t N_points, deviceFloat time, const deviceFloat* velocity, const deviceFloat* coordinates);
    void write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices);
    
    template<typename Polynomial>
    void solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG);
};

namespace SEM {
    __global__
    void rk3_step(size_t N_elements, Element_t* elements, deviceFloat delta_t, deviceFloat a, deviceFloat g);

    __global__
    void calculate_fluxes(size_t N_faces, Face_t* faces, const Element_t* elements);

    // Algorithm 19
    __device__
    void matrix_vector_derivative(int N, const deviceFloat* derivative_matrices_hat, const deviceFloat* phi, deviceFloat* phi_prime);

    // Algorithm 60 (not really anymore)
    __global__
    void compute_dg_derivative(size_t N_elements, Element_t* elements, const Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);
}

#endif