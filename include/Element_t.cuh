#ifndef NDG_ELEMENT_T_H
#define NDG_ELEMENT_T_H

#include "float_types.h"
#include "Face_t.cuh"

namespace SEM {
    class Element_t { // Turn this into separate vectors, because cache exists
        public:
            __device__ 
            Element_t(int N, size_t face_L, size_t face_R, deviceFloat x_L, deviceFloat x_R);

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
            size_t faces_[2]; // Could also be pointers. left, right
            deviceFloat x_[2];
            deviceFloat delta_x_;
            deviceFloat phi_L_;
            deviceFloat phi_R_;
            deviceFloat phi_prime_L_;
            deviceFloat phi_prime_R_;
            deviceFloat* phi_; // Solution
            deviceFloat* phi_prime_;
            deviceFloat* intermediate_; // This is used for RK3, and also for adaptivity. So don't try to adapt between rk steps.

            deviceFloat sigma_;
            bool refine_;
            bool coarsen_;
            deviceFloat error_;

            // Algorithm 61
            __device__
            void interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right, const deviceFloat* lagrange_interpolant_derivative_left, const deviceFloat* lagrange_interpolant_derivative_right);

            template<typename Polynomial>
            __device__
            void estimate_error(const deviceFloat* nodes, const deviceFloat* weights);

            __device__
            void interpolate_from(const Element_t& other, const deviceFloat* nodes, const deviceFloat* barycentric_weights);

        private:
            __device__
            deviceFloat exponential_decay();
    };

    __global__
    void build_elements(size_t N_elements, int N, Element_t* elements, deviceFloat x_min, deviceFloat x_max);

    __global__
    void build_boundaries(size_t N_elements, size_t N_elements_global, size_t N_local_boundaries, size_t N_MPI_boundaries, int N, Element_t* elements, deviceFloat x_min, deviceFloat x_max, size_t global_element_offset, size_t* local_boundary_to_element, size_t* MPI_boundary_to_element, size_t* MPI_boundary_from_element);

    __global__
    void free_elements(size_t N_elements, Element_t* elements);

    template<typename Polynomial>
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
    void get_solution(size_t N_elements, size_t N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* phi, deviceFloat* phi_prime, deviceFloat* intermediate, deviceFloat* x_L, deviceFloat* x_R, int* N, deviceFloat* sigma, bool* refine, bool* coarsen, deviceFloat* error);
    
    __global__
    void interpolate_to_boundaries(size_t N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right, const deviceFloat* lagrange_interpolant_derivative_left, const deviceFloat* lagrange_interpolant_derivative_right);

    __global__
    void adapt(unsigned long N_elements, Element_t* elements, Element_t* new_elements, Face_t* new_faces, const unsigned long* block_offsets, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights);

    __global__
    void p_adapt(unsigned long N_elements, Element_t* elements, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights);

    // From cppreference.com
    __device__
    bool almost_equal2(deviceFloat x, deviceFloat y);

    __global__
    void local_boundaries(size_t N_elements, size_t N_local_boundaries, Element_t* elements, const size_t* local_boundary_to_element);

    __global__
    void get_MPI_boundaries(size_t N_elements, size_t N_local_boundaries, size_t N_MPI_boundaries, const Element_t* elements, deviceFloat* phi_L, deviceFloat* phi_R, deviceFloat* phi_prime_L, deviceFloat* phi_prime_R);

    __global__
    void put_MPI_boundaries(size_t N_elements, size_t N_local_boundaries, size_t N_MPI_boundaries, Element_t* elements, const deviceFloat* phi_L, const deviceFloat* phi_R, const deviceFloat* phi_prime_L, const deviceFloat* phi_prime_R);
}

#endif