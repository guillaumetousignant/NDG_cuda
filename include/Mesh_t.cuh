#ifndef MESH_T_H
#define MESH_T_H

namespace SEM {
    __global__
    void rk3_step(int N_elements, Element_t* elements, float delta_t, float a, float g);

    __global__
    void calculate_fluxes(int N_faces, Face_t* faces, const Element_t* elements);

    // Algorithm 60 (not really anymore)
    __global__
    void compute_dg_derivative(int N_elements, Element_t* elements, const Face_t* faces, const float* weights, const float* derivative_matrices_hat, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right);
}

class Mesh_t {
public:
    Mesh_t(int N_elements, int initial_N, float x_min, float x_max) : N_elements_(N_elements), N_faces_(N_elements), initial_N_(initial_N);
    ~Mesh_t();

    int N_elements_;
    int N_faces_;
    int initial_N_;
    Element_t* elements_;
    Face_t* faces_;

    void set_initial_conditions(const float* nodes);
    void print();
    void write_file_data(int N_points, float time, const float* velocity, const float* coordinates);
    void write_data(float time, int N_interpolation_points, const float* interpolation_matrices);
    void solve(const float delta_t, const std::vector<float> output_times, const NDG_t &NDG);
};

#endif