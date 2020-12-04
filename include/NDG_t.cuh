#ifndef NDG_T_H
#define NDG_T_H

class NDG_t { 
public: 
    NDG_t(int N_max, int N_interpolation_points);
    ~NDG_t();

    int N_max_;
    int N_interpolation_points_;
    int vector_length_; // Flattened length of all N one after the other
    int matrix_length_; // Flattened length of all N² one after the other
    int interpolation_length_;
    float* nodes_;
    float* weights_;
    float* barycentric_weights_;
    float* lagrange_interpolant_left_;
    float* lagrange_interpolant_right_;
    float* derivative_matrices_;
    float* derivative_matrices_hat_;
    float* interpolation_matrices_;

    void print();
};

namespace SEM {
    // Algorithm 26
    __global__
    void chebyshev_gauss_nodes_and_weights(int N, float* nodes, float* weights);

    // Algorithm 22
    __device__
    void legendre_polynomial_and_derivative(int N, float x, float &L_N, float &L_N_prime);

    // Algorithm 23
    __global__
    void legendre_gauss_nodes_and_weights(int N, float* nodes, float* weights);

    // Algorithm 30
    __global__
    void calculate_barycentric_weights(int N, const float* nodes, float* barycentric_weights) ;

    // From cppreference.com
    __device__
    bool almost_equal(float x, float y);

    // This will not work if we are on a node, or at least be pretty inefficient
    // Algorithm 34
    __global__
    void lagrange_interpolating_polynomials(float x, int N, const float* nodes, const float* barycentric_weights, float* lagrange_interpolant);

    // Algorithm 34
    __global__
    void normalize_lagrange_interpolating_polynomials(int N_max, float* lagrange_interpolant);

    // Be sure to compute the diagonal afterwards
    // Algorithm 37
    __global__
    void polynomial_derivative_matrices(int N, const float* nodes, const float* barycentric_weights, float* derivative_matrices);

    // Algorithm 37
    __global__
    void polynomial_derivative_matrices_diagonal(int N, float* derivative_matrices);

    __global__
    void polynomial_derivative_matrices_hat(int N, const float* weights, const float* derivative_matrices, float* derivative_matrices_hat);

    // Will interpolate N_interpolation_points between -1 and 1
    __global__
    void create_interpolation_matrices(int N, int N_interpolation_points, const float* nodes, const float* barycentric_weights, float* interpolation_matrices);

    // Algorithm 19
    __device__
    void matrix_vector_derivative(int N, const float* derivative_matrices_hat, const float* phi, float* phi_prime);
    
    // Algorithm 61
    __device__
    float interpolate_to_boundary(int N, const float* phi, const float* lagrange_interpolant);
}

#endif