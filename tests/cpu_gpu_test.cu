#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include "helpers/float_types.h"
#include "entities/NDG_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "entities/NDG_host_t.h"
#include "polynomials/ChebyshevPolynomial_host_t.h"
#include "polynomials/LegendrePolynomial_host_t.h"

TEST_CASE("ChebyshevPolynomials_CPU_GPU", "Compares the Chebyshev polynomials between the CPU and GPU implementations."){
    const int N_max = 16;
    const size_t n_interpolation_points = N_max * 8;
    const double error = 1e-6;

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::Entities::NDG_t<SEM::Polynomials::ChebyshevPolynomial_t> NDG(N_max, n_interpolation_points, stream);
    SEM::Entities::NDG_host_t<SEM::Polynomials::ChebyshevPolynomial_host_t> NDG_host(N_max, n_interpolation_points);

    std::vector<deviceFloat> host_nodes(NDG.vector_length_);
    std::vector<deviceFloat> host_weights(NDG.vector_length_);
    std::vector<deviceFloat> host_barycentric_weights(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_left(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_right(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_derivative_left(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_derivative_right(NDG.vector_length_);
    std::vector<deviceFloat> host_derivative_matrices(NDG.matrix_length_);
    std::vector<deviceFloat> host_g_hat_derivative_matrices(NDG.matrix_length_);
    std::vector<deviceFloat> host_derivative_matrices_hat(NDG.matrix_length_);
    std::vector<deviceFloat> host_interpolation_matrices(NDG.interpolation_length_);

    NDG.nodes_.copy_to(host_nodes, stream);
    NDG.weights_.copy_to(host_weights, stream);
    NDG.barycentric_weights_.copy_to(host_barycentric_weights, stream);
    NDG.lagrange_interpolant_left_.copy_to(host_lagrange_interpolant_left, stream);
    NDG.lagrange_interpolant_right_.copy_to(host_lagrange_interpolant_right, stream);
    NDG.lagrange_interpolant_derivative_left_.copy_to(host_lagrange_interpolant_derivative_left, stream);
    NDG.lagrange_interpolant_derivative_right_.copy_to(host_lagrange_interpolant_derivative_right, stream);
    NDG.derivative_matrices_.copy_to(host_derivative_matrices, stream);
    NDG.g_hat_derivative_matrices_.copy_to(host_g_hat_derivative_matrices, stream);
    NDG.derivative_matrices_hat_.copy_to(host_derivative_matrices_hat, stream);
    NDG.interpolation_matrices_.copy_to(host_interpolation_matrices, stream);
    cudaStreamSynchronize(stream);

    for (int N_test = 0; N_test <= N_max; ++N_test) {
        const size_t offset_1D = N_test * (N_test + 1) /2;
        const size_t offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
        const size_t offset_interp = N_test * (N_test + 1) * n_interpolation_points/2;

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_nodes[offset_1D + i] - NDG_host.nodes_[N_test][i]) < error);
        }
            
        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_weights[offset_1D + i] - NDG_host.weights_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_barycentric_weights[offset_1D + i] - NDG_host.barycentric_weights_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_left[offset_1D + i] - NDG_host.lagrange_interpolant_left_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_right[offset_1D + i] - NDG_host.lagrange_interpolant_right_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_derivative_left[offset_1D + i] - NDG_host.lagrange_interpolant_derivative_left_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_derivative_right[offset_1D + i] - NDG_host.lagrange_interpolant_derivative_right_[N_test][i]) < error);
        }

        for (int i = 0; i < std::pow(N_test + 1, 2); ++i) {
            REQUIRE(std::abs(host_derivative_matrices[offset_2D + i] - NDG_host.derivative_matrices_[N_test][i]) < error);
        }

        for (int i = 0; i < std::pow(N_test + 1, 2); ++i) {
            REQUIRE(std::abs(host_g_hat_derivative_matrices[offset_2D + i] - NDG_host.g_hat_derivative_matrices_[N_test][i]) < error);
        }

        for (int i = 0; i < std::pow(N_test + 1, 2); ++i) {
            REQUIRE(std::abs(host_derivative_matrices_hat[offset_2D + i] - NDG_host.derivative_matrices_hat_[N_test][i]) < error);
        }
        
        for (int i = 0; i < (N_test + 1) * n_interpolation_points; ++i) {
            REQUIRE(std::abs(host_interpolation_matrices[offset_interp + i] - NDG_host.interpolation_matrices_[N_test][i]) < error);
        }
    }

    cudaStreamDestroy(stream);
}

TEST_CASE("LegendrePolynomials_CPU_GPU", "Compares the Legendre polynomials between the CPU and GPU implementations."){
    const int N_max = 16;
    const size_t n_interpolation_points = N_max * 8;
    const double error = 1e-6;

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, n_interpolation_points, stream);
    SEM::Entities::NDG_host_t<SEM::Polynomials::LegendrePolynomial_host_t> NDG_host(N_max, n_interpolation_points);

    std::vector<deviceFloat> host_nodes(NDG.vector_length_);
    std::vector<deviceFloat> host_weights(NDG.vector_length_);
    std::vector<deviceFloat> host_barycentric_weights(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_left(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_right(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_derivative_left(NDG.vector_length_);
    std::vector<deviceFloat> host_lagrange_interpolant_derivative_right(NDG.vector_length_);
    std::vector<deviceFloat> host_derivative_matrices(NDG.matrix_length_);
    std::vector<deviceFloat> host_g_hat_derivative_matrices(NDG.matrix_length_);
    std::vector<deviceFloat> host_derivative_matrices_hat(NDG.matrix_length_);
    std::vector<deviceFloat> host_interpolation_matrices(NDG.interpolation_length_);

    NDG.nodes_.copy_to(host_nodes, stream);
    NDG.weights_.copy_to(host_weights, stream);
    NDG.barycentric_weights_.copy_to(host_barycentric_weights, stream);
    NDG.lagrange_interpolant_left_.copy_to(host_lagrange_interpolant_left, stream);
    NDG.lagrange_interpolant_right_.copy_to(host_lagrange_interpolant_right, stream);
    NDG.lagrange_interpolant_derivative_left_.copy_to(host_lagrange_interpolant_derivative_left, stream);
    NDG.lagrange_interpolant_derivative_right_.copy_to(host_lagrange_interpolant_derivative_right, stream);
    NDG.derivative_matrices_.copy_to(host_derivative_matrices, stream);
    NDG.g_hat_derivative_matrices_.copy_to(host_g_hat_derivative_matrices, stream);
    NDG.derivative_matrices_hat_.copy_to(host_derivative_matrices_hat, stream);
    NDG.interpolation_matrices_.copy_to(host_interpolation_matrices, stream);
    cudaStreamSynchronize(stream);

    for (int N_test = 0; N_test <= N_max; ++N_test) {
        const size_t offset_1D = N_test * (N_test + 1) /2;
        const size_t offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
        const size_t offset_interp = N_test * (N_test + 1) * n_interpolation_points/2;
        
        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_nodes[offset_1D + i] - NDG_host.nodes_[N_test][i]) < error);
        }
            
        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_weights[offset_1D + i] - NDG_host.weights_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_barycentric_weights[offset_1D + i] - NDG_host.barycentric_weights_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_left[offset_1D + i] - NDG_host.lagrange_interpolant_left_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_right[offset_1D + i] - NDG_host.lagrange_interpolant_right_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_derivative_left[offset_1D + i] - NDG_host.lagrange_interpolant_derivative_left_[N_test][i]) < error);
        }

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(host_lagrange_interpolant_derivative_right[offset_1D + i] - NDG_host.lagrange_interpolant_derivative_right_[N_test][i]) < error);
        }

        for (int i = 0; i < std::pow(N_test + 1, 2); ++i) {
            REQUIRE(std::abs(host_derivative_matrices[offset_2D + i] - NDG_host.derivative_matrices_[N_test][i]) < error);
        }

        for (int i = 0; i < std::pow(N_test + 1, 2); ++i) {
            REQUIRE(std::abs(host_g_hat_derivative_matrices[offset_2D + i] - NDG_host.g_hat_derivative_matrices_[N_test][i]) < error);
        }

        for (int i = 0; i < std::pow(N_test + 1, 2); ++i) {
            REQUIRE(std::abs(host_derivative_matrices_hat[offset_2D + i] - NDG_host.derivative_matrices_hat_[N_test][i]) < error);
        }
        
        for (int i = 0; i < (N_test + 1) * n_interpolation_points; ++i) {
            REQUIRE(std::abs(host_interpolation_matrices[offset_interp + i] - NDG_host.interpolation_matrices_[N_test][i]) < error);
        }
    }

    cudaStreamDestroy(stream);
}