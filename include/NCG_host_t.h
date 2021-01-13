#ifndef NDG_NDG_HOST_T_H
#define NDG_NDG_HOST_T_H

#include "float_types.h"
#include <vector>

template<typename Polynomial>
class NDG_host_t { 
public: 
    NDG_host_t(int N_max, int N_interpolation_points);
    ~NDG_host_t();

    int N_max_;
    int N_interpolation_points_;
    std::vector<std::vector<hostFloat>> nodes_;
    std::vector<std::vector<hostFloat>> weights_;
    std::vector<std::vector<hostFloat>> barycentric_weights_;
    std::vector<std::vector<hostFloat>> lagrange_interpolant_left_;
    std::vector<std::vector<hostFloat>> lagrange_interpolant_right_;
    std::vector<std::vector<hostFloat>> derivative_matrices_;
    std::vector<std::vector<hostFloat>> derivative_matrices_hat_;
    std::vector<std::vector<hostFloat>> interpolation_matrices_;

    void print();
};

namespace SEM {
    // Algorithm 30
    void calculate_barycentric_weights(int N, const float* nodes, float* barycentric_weights) ;

    // From cppreference.com
    bool almost_equal(float x, float y);

    // This will not work if we are on a node, or at least be pretty inefficient
    // Algorithm 34
    void lagrange_interpolating_polynomials(float x, int N, const float* nodes, const float* barycentric_weights, float* lagrange_interpolant);

    // Algorithm 34
    void normalize_lagrange_interpolating_polynomials(int N_max, float* lagrange_interpolant);

    // Be sure to compute the diagonal afterwards
    // Algorithm 37
    void polynomial_derivative_matrices(int N, const float* nodes, const float* barycentric_weights, float* derivative_matrices);

    // Algorithm 37
    void polynomial_derivative_matrices_diagonal(int N, float* derivative_matrices);

    void polynomial_derivative_matrices_hat(int N, const float* weights, const float* derivative_matrices, float* derivative_matrices_hat);

    // Will interpolate N_interpolation_points between -1 and 1
    void create_interpolation_matrices(int N, int N_interpolation_points, const float* nodes, const float* barycentric_weights, float* interpolation_matrices);
}

#endif