#ifndef NDG_NDG_T_H
#define NDG_NDG_T_H

#include "float_types.h"

template<typename Polynomial>
class NDG_t { 
public: 
    NDG_t(int N_max, int N_interpolation_points);
    ~NDG_t();

    int N_max_;
    int N_interpolation_points_;
    int vector_length_; // Flattened length of all N one after the other
    int matrix_length_; // Flattened length of all N² one after the other
    int interpolation_length_;
    deviceFloat* nodes_;
    deviceFloat* weights_;
    deviceFloat* barycentric_weights_;
    deviceFloat* lagrange_interpolant_left_;
    deviceFloat* lagrange_interpolant_right_;
    deviceFloat* derivative_matrices_;
    deviceFloat* derivative_matrices_hat_;
    deviceFloat* interpolation_matrices_;

    void print();
};

namespace SEM {
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

    // Be sure to compute the diagonal afterwards
    // Algorithm 37
    __global__
    void polynomial_derivative_matrices(int N, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* derivative_matrices);

    // Algorithm 37
    __global__
    void polynomial_derivative_matrices_diagonal(int N, deviceFloat* derivative_matrices);

    __global__
    void polynomial_derivative_matrices_hat(int N, const deviceFloat* weights, const deviceFloat* derivative_matrices, deviceFloat* derivative_matrices_hat);

    // Will interpolate N_interpolation_points between -1 and 1
    __global__
    void create_interpolation_matrices(int N, int N_interpolation_points, const deviceFloat* nodes, const deviceFloat* barycentric_weights, deviceFloat* interpolation_matrices);
}

#endif