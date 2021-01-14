#ifndef NDG_ELEMENT_T_H
#define NDG_ELEMENT_T_H

#include "float_types.h"

class Element_t { // Turn this into separate vectors, because cache exists
public:
    __device__ 
    Element_t(int N, int neighbour_L, int neighbour_R, int face_L, int face_R, deviceFloat x_L, deviceFloat x_R);

    __host__ 
    Element_t();

    __host__ __device__
    ~Element_t();

    int N_;
    int neighbours_[2]; // Could also be pointers
    int faces_[2]; // Could also be pointers. left, right
    deviceFloat x_[2];
    deviceFloat delta_x_;
    deviceFloat phi_L_;
    deviceFloat phi_R_;
    deviceFloat* phi_; // Solution
    deviceFloat* phi_prime_;
    deviceFloat* intermediate_;
};

namespace SEM {
    __global__
    void build_elements(int N_elements, int N, Element_t* elements, deviceFloat x_min, deviceFloat x_max);

    __device__
    deviceFloat g(deviceFloat x);

    __global__
    void initial_conditions(int N_elements, Element_t* elements, const deviceFloat* nodes);

    // Basically useless, find better solution when multiple elements.
    __global__
    void get_elements_data(int N_elements, const Element_t* elements, deviceFloat* phi, deviceFloat* phi_prime);

    // Basically useless, find better solution when multiple elements.
    __global__
    void get_phi(int N_elements, const Element_t* elements, deviceFloat* phi);

    __global__
    void get_solution(int N_elements, int N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* phi, deviceFloat* x);

    // Algorithm 61
    __device__
    deviceFloat interpolate_to_boundary(int N, const deviceFloat* phi, const deviceFloat* lagrange_interpolant);
    
    __global__
    void interpolate_to_boundaries(int N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);
}

#endif