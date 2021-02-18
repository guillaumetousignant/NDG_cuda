#ifndef NDG_NDG_T_H
#define NDG_NDG_T_H

#include "float_types.h"

namespace SEM {
    template<typename Polynomial>
    class NDG_t { 
        public: 
            NDG_t(int N_max, size_t N_interpolation_points);
            ~NDG_t();

            int N_max_;
            size_t N_interpolation_points_;
            size_t vector_length_; // Flattened length of all N one after the other
            size_t matrix_length_; // Flattened length of all N² one after the other
            size_t interpolation_length_;
            deviceFloat* nodes_;
            deviceFloat* weights_;
            deviceFloat* barycentric_weights_;
            deviceFloat* lagrange_interpolant_left_;
            deviceFloat* lagrange_interpolant_right_;
            deviceFloat* lagrange_interpolant_derivative_left_;
            deviceFloat* lagrange_interpolant_derivative_right_;
            deviceFloat* derivative_matrices_;
            deviceFloat* g_hat_derivative_matrices_;
            deviceFloat* derivative_matrices_hat_;
            deviceFloat* interpolation_matrices_;

            void print();
    };

    // Algorithm 30
    __global__
    void calculate_barycentric_weights(int N, const deviceFloat* nodes, deviceFloat* barycentric_weights) ;

    // From cppreference.com
    __device__
    bool almost_equal(deviceFloat x, deviceFloat y);

    // This will not work if we are on a node, or at least be pretty inefficient
    // Algorithm 34
    __global__
    void lagrange_interpolating_polynomials(deviceFloat x, int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_interpolant);

    // Algorithm 34
    __global__
    void normalize_lagrange_interpolating_polynomials(int N_max, deviceFloat* lagrange_interpolant);

    // This will not work if we are on a node, or at least be pretty inefficient
    // Algorithm 36
    __global__
    void lagrange_interpolating_derivative_polynomials(deviceFloat x, int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_derivative_interpolant);

    // Algorithm 36
    __global__
    void normalize_lagrange_interpolating_derivative_polynomials(deviceFloat x, int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* lagrange_derivative_interpolant);

    // Be sure to compute the diagonal afterwards
    // Algorithm 37
    __global__
    void polynomial_derivative_matrices(int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* derivative_matrices);

    // Algorithm 37
    __global__
    void polynomial_derivative_matrices_diagonal(int N, deviceFloat* derivative_matrices);

    // Algorithm 57
    __global__
    void polynomial_cg_derivative_matrices(int N, const deviceFloat* weights, const deviceFloat* derivative_matrices, deviceFloat* g_hat_derivative_matrices);

    __global__
    void polynomial_derivative_matrices_hat(int N, const deviceFloat* weights, const deviceFloat* derivative_matrices, deviceFloat* derivative_matrices_hat);

    // Will interpolate N_interpolation_points between -1 and 1
    __global__
    void create_interpolation_matrices(int N, size_t N_interpolation_points, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* interpolation_matrices);
}

#endif