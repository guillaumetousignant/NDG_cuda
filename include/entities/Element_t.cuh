#ifndef NDG_ELEMENT_T_H
#define NDG_ELEMENT_T_H

#include "helpers/float_types.h"
#include "Face_t.cuh"
#include <array>

namespace SEM { namespace Entities {
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
            std::array<size_t, 2> faces_; // Could also be pointers. left, right
            std::array<deviceFloat, 2> x_;
            deviceFloat delta_x_;
            deviceFloat phi_L_;
            deviceFloat phi_R_;
            deviceFloat phi_prime_L_;
            deviceFloat phi_prime_R_;
            deviceFloat* phi_; // Solution
            deviceFloat* q_;
            deviceFloat* ux_;
            deviceFloat* phi_prime_;
            deviceFloat* intermediate_; // This is used for RK3, and also for adaptivity. So don't try to adapt between rk steps.

            deviceFloat sigma_;
            bool refine_;
            bool coarsen_;
            deviceFloat error_;

            // Algorithm 61
            __device__
            void interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

            // Algorithm 61
            __device__
            void interpolate_q_to_boundaries(const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

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
    void build_boundaries(size_t N_elements, size_t N_elements_global, size_t N_local_boundaries, size_t N_MPI_boundaries, Element_t* elements, size_t global_element_offset, size_t* local_boundary_to_element, size_t* MPI_boundary_to_element, size_t* MPI_boundary_from_element);

    __global__
    void adjust_boundaries(size_t N_elements, size_t N_elements_global, size_t N_MPI_boundaries, size_t global_element_offset, size_t* MPI_boundary_to_element, size_t* MPI_boundary_from_element);
 
    __global__
    void free_elements(size_t N_elements, Element_t* elements);

    template<typename Polynomial>
    __global__
    void estimate_error(size_t N_elements, Element_t* elements, const deviceFloat* nodes, const deviceFloat* weights);

    __host__ __device__
    deviceFloat g(deviceFloat x);

    __host__ __device__
    deviceFloat g_prime(deviceFloat x);

    __global__
    void initial_conditions(size_t N_elements, Element_t* elements, const deviceFloat* nodes);

    // Basically useless, find better solution when multiple elements.
    __global__
    void get_elements_data(size_t N_elements, const Element_t* elements, deviceFloat* phi, deviceFloat* phi_prime);

    __global__
    void get_phi(size_t N_elements, const Element_t* elements, deviceFloat** phi);

    __global__
    void put_phi(size_t N_elements, Element_t* elements, deviceFloat** phi);

    __global__
    void move_elements(size_t N_elements, Element_t* elements, Element_t* new_elements, size_t source_start_index, size_t destination_start_index);

    __global__
    void get_solution(size_t N_elements, size_t N_interpolation_points, const Element_t* elements, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* phi, deviceFloat* phi_prime, deviceFloat* intermediate, deviceFloat* x_L, deviceFloat* x_R, int* N, deviceFloat* sigma, bool* refine, bool* coarsen, deviceFloat* error, deviceFloat* delta_x);
    
    __global__
    void interpolate_to_boundaries(size_t N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    __global__
    void interpolate_q_to_boundaries(size_t N_elements, Element_t* elements, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    __global__
    void hp_adapt(unsigned long N_elements, Element_t* elements, Element_t* new_elements, const unsigned long* block_offsets, deviceFloat delta_x_min, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights);

    __global__
    void p_adapt(unsigned long N_elements, Element_t* elements, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights);

    // From cppreference.com
    __device__
    bool almost_equal2(deviceFloat x, deviceFloat y);

    __global__
    void local_boundaries(size_t N_elements, size_t N_local_boundaries, Element_t* elements, const size_t* local_boundary_to_element);

    __global__
    void get_MPI_boundaries(size_t N_elements, size_t N_local_boundaries, size_t N_MPI_boundaries, const Element_t* elements, const Face_t* faces, deviceFloat* phi_L, deviceFloat* phi_R, deviceFloat* phi_prime_L, deviceFloat* phi_prime_R);

    __global__
    void put_MPI_boundaries(size_t N_elements, size_t N_local_boundaries, size_t N_MPI_boundaries, Element_t* elements, const deviceFloat* phi_L, const deviceFloat* phi_R, const deviceFloat* phi_prime_L, const deviceFloat* phi_prime_R);
}}

#endif
