#ifndef NDG_NDG_HOST_T_H
#define NDG_NDG_HOST_T_H

#include "helpers/float_types.h"
#include <vector>

namespace SEM { namespace Entities {
    template<typename Polynomial>
    class NDG_host_t { 
        public: 
            NDG_host_t(int N_max, size_t N_interpolation_points);

            int N_max_;
            size_t N_interpolation_points_;
            std::vector<std::vector<hostFloat>> nodes_;
            std::vector<std::vector<hostFloat>> weights_;
            std::vector<std::vector<hostFloat>> barycentric_weights_;
            std::vector<std::vector<hostFloat>> lagrange_interpolant_left_;
            std::vector<std::vector<hostFloat>> lagrange_interpolant_right_;
            std::vector<std::vector<hostFloat>> lagrange_interpolant_derivative_left_;
            std::vector<std::vector<hostFloat>> lagrange_interpolant_derivative_right_;
            std::vector<std::vector<hostFloat>> derivative_matrices_;
            std::vector<std::vector<hostFloat>> g_hat_derivative_matrices_;
            std::vector<std::vector<hostFloat>> derivative_matrices_hat_;
            std::vector<std::vector<hostFloat>> interpolation_matrices_;

            void print();

        private:
            // Algorithm 30
            static void calculate_barycentric_weights(int N, const std::vector<hostFloat>& nodes, std::vector<hostFloat>& barycentric_weights) ;

            // From cppreference.com
            static bool almost_equal(hostFloat x, hostFloat y);

            // This will not work if we are on a node, or at least be pretty inefficient
            // Algorithm 34
            static void lagrange_interpolating_polynomials(hostFloat x, int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& lagrange_interpolant);

            // Algorithm 34
            static void normalize_lagrange_interpolating_polynomials(int N, std::vector<hostFloat>& lagrange_interpolant);

            // This will not work if we are on a node, or at least be pretty inefficient
            // Algorithm 36
            static void lagrange_interpolating_derivative_polynomials(hostFloat x, int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& lagrange_derivative_interpolant);

            // Algorithm 36
            static void normalize_lagrange_interpolating_derivative_polynomials(hostFloat x, int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& lagrange_derivative_interpolant);

            // Be sure to compute the diagonal afterwards
            // Algorithm 37
            static void polynomial_derivative_matrices(int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& derivative_matrices);

            // Algorithm 37
            static void polynomial_derivative_matrices_diagonal(int N, std::vector<hostFloat>& derivative_matrices);

            static void polynomial_derivative_matrices_hat(int N, const std::vector<hostFloat>& weights, const std::vector<hostFloat>& derivative_matrices, std::vector<hostFloat>& derivative_matrices_hat);

            // Algorithm 57
            static void polynomial_cg_derivative_matrices(int N, const std::vector<hostFloat>& weights, const std::vector<hostFloat>& derivative_matrices, std::vector<hostFloat>& g_hat_derivative_matrices);

            // Will interpolate N_interpolation_points between -1 and 1
            static void create_interpolation_matrices(int N, size_t N_interpolation_points, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& interpolation_matrices);
    };
}}

#endif
