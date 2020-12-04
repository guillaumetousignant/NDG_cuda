#ifndef ELEMENT_T_H
#define ELEMENT_T_H

namespace SEM {
    __global__
    void build_elements(int N_elements, int N, Element_t* elements, float x_min, float x_max);

    __device__
    float g(float x);

    __global__
    void initial_conditions(int N_elements, Element_t* elements, const float* nodes);

    // Basically useless, find better solution when multiple elements.
    __global__
    void get_elements_data(int N_elements, const Element_t* elements, float* phi, float* phi_prime);

    // Basically useless, find better solution when multiple elements.
    __global__
    void get_phi(int N_elements, const Element_t* elements, float* phi);

    __global__
    void get_solution(int N_elements, int N_interpolation_points, const Element_t* elements, const float* interpolation_matrices, float* phi, float* x);

    __global__
    void interpolate_to_boundaries(int N_elements, Element_t* elements, const float* lagrange_interpolant_left, const float* lagrange_interpolant_right);
}

class Element_t { // Turn this into separate vectors, because cache exists
public:
    __device__ 
    Element_t(int N, int neighbour_L, int neighbour_R, int face_L, int face_R, float x_L, float x_R);

    __host__ 
    Element_t();

    __host__ __device__
    ~Element_t();

    int N_;
    int neighbours_[2]; // Could also be pointers
    int faces_[2]; // Could also be pointers. left, right
    float x_[2];
    float delta_x_;
    float phi_L_;
    float phi_R_;
    float* phi_; // Solution
    float* phi_prime_;
    float* intermediate_;
};

#endif