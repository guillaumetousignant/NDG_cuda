#ifndef NDG_ELEMENT_T_H
#define NDG_ELEMENT_T_H

#include "float_types.h"

class Element_t { // Turn this into separate vectors, because cache exists
public:
    __device__ 
    Element_t(int N, size_t neighbour_L, size_t neighbour_R, size_t face_L, size_t face_R, deviceFloat x_L, deviceFloat x_R, deviceFloat* phi_array, deviceFloat* phi_prime_array, deviceFloat* intermediate_array);

    __host__ 
    Element_t();

    __host__ __device__
    ~Element_t();

    int N_;
    size_t neighbours_[2]; // Could also be pointers
    size_t faces_[2]; // Could also be pointers. left, right
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
    void build_elements(size_t N_elements, int N, Element_t* elements, deviceFloat x_min, deviceFloat x_max, deviceFloat** phi_arrays, deviceFloat** phi_prime_arrays, deviceFloat** intermediate_arrays);

    __device__
    deviceFloat g(deviceFloat x);

    __global__
    void initial_conditions(size_t N_elements, Element_t* elements, const deviceFloat* nodes);

    // Basically useless, find better solution when multiple elements.
    __global__
    void get_elements_data(size_t N_elements, const Element_t* elements, deviceFloat* phi, deviceFloat* phi_prime);

    // Basically useless, find better solution when multiple elements.
    __global__
    void get_phi(size_t N_elements, const Element_t* elements, deviceFloat* phi);

    __global__
    void get_solution(size_t N_elements, size_t N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* phi, deviceFloat* x);

    // Algorithm 61
    __device__
    deviceFloat interpolate_to_boundary(int N, const deviceFloat* phi, const deviceFloat* lagrange_interpolant);
    
    __global__
    void interpolate_to_boundaries(size_t N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);
}

#endif