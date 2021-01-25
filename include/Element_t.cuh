#ifndef NDG_ELEMENT_T_H
#define NDG_ELEMENT_T_H

#include "float_types.h"

class Element_t { // Turn this into separate vectors, because cache exists
    public:
        __device__ 
        Element_t(int N, size_t neighbour_L, size_t neighbour_R, size_t face_L, size_t face_R, deviceFloat x_L, deviceFloat x_R);

        __device__
        Element_t(const Element_t& other);

        __device__
        Element_t(Element_t&& other);	

        __device__
        Element_t& operator=(const Element_t& other);

        __device__
        Element_t& operator=(Element_t&& other);

        __host__ __device__
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
        deviceFloat* intermediate_; // This is used for RK3, and also for adaptivity. So don't try to adapt between rk steps.

        deviceFloat sigma_;
        bool refine_;
        bool coarsen_;
        deviceFloat error_;

        // Algorithm 61
        __device__
        void interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

        __device__
        void estimate_error(const deviceFloat* nodes, const deviceFloat* weights);

    private:
        __device__
        deviceFloat exponential_decay();
};

namespace SEM {
    __global__
    void build_elements(size_t N_elements, int N, Element_t* elements, deviceFloat x_min, deviceFloat x_max);

    __global__
    void estimate_error(size_t N_elements, Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights);

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
    void get_solution(size_t N_elements, size_t N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* phi, deviceFloat* phi_prime, deviceFloat* intermediate, deviceFloat* sigma, deviceFloat* refine, deviceFloat* coarsen, deviceFloat* error);
    
    __global__
    void interpolate_to_boundaries(size_t N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    // Algorithm 22
    __device__
    void polynomial_and_derivative(int N, deviceFloat x, deviceFloat &L_N, deviceFloat &L_N_prime);
}

#endif