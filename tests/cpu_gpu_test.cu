#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include "float_types.h"
#include "NDG_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include "NDG_host_t.h"
#include "ChebyshevPolynomial_host_t.h"
#include "LegendrePolynomial_host_t.h"

TEST_CASE("ChebyshevPolynomials_CPU_GPU", "Compares the Chebyshev polynomials between the CPU and GPU implementations."){
    const int N_max = 16;
    const size_t N_interpolation_points = N_max * 8;
    const double error = 1e-6;

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::NDG_t<SEM::ChebyshevPolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::NDG_host_t<SEM::ChebyshevPolynomial_host_t> NDG_host(N_max, N_interpolation_points);

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

    cudaMemcpy(host_nodes.data(), NDG.nodes_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights.data(), NDG.weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights.data(), NDG.barycentric_weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left.data(), NDG.lagrange_interpolant_left_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right.data(), NDG.lagrange_interpolant_right_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_derivative_left.data(), NDG.lagrange_interpolant_derivative_left_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_derivative_right.data(), NDG.lagrange_interpolant_derivative_right_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices.data(), NDG.derivative_matrices_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_g_hat_derivative_matrices.data(), NDG.g_hat_derivative_matrices_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices_hat.data(), NDG.derivative_matrices_hat_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_interpolation_matrices.data(), NDG.interpolation_matrices_, NDG.interpolation_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    for (int N_test = 0; N_test <= N_max; ++N_test) {
        const size_t offset_1D = N_test * (N_test + 1) /2;
        const size_t offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
        const size_t offset_interp = N_test * (N_test + 1) * N_interpolation_points/2;

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
        
        for (int i = 0; i < (N_test + 1) * N_interpolation_points; ++i) {
            REQUIRE(std::abs(host_interpolation_matrices[offset_interp + i] - NDG_host.interpolation_matrices_[N_test][i]) < error);
        }
    }

    cudaStreamDestroy(stream);
}

TEST_CASE("LegendrePolynomials_CPU_GPU", "Compares the Legendre polynomials between the CPU and GPU implementations."){
    const int N_max = 16;
    const size_t N_interpolation_points = N_max * 8;
    const double error = 1e-6;

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::NDG_t<SEM::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::NDG_host_t<SEM::LegendrePolynomial_host_t> NDG_host(N_max, N_interpolation_points);

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

    cudaMemcpy(host_nodes.data(), NDG.nodes_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights.data(), NDG.weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights.data(), NDG.barycentric_weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left.data(), NDG.lagrange_interpolant_left_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right.data(), NDG.lagrange_interpolant_right_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_derivative_left.data(), NDG.lagrange_interpolant_derivative_left_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_derivative_right.data(), NDG.lagrange_interpolant_derivative_right_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices.data(), NDG.derivative_matrices_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_g_hat_derivative_matrices.data(), NDG.g_hat_derivative_matrices_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices_hat.data(), NDG.derivative_matrices_hat_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_interpolation_matrices.data(), NDG.interpolation_matrices_, NDG.interpolation_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    for (int N_test = 0; N_test <= N_max; ++N_test) {
        const size_t offset_1D = N_test * (N_test + 1) /2;
        const size_t offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
        const size_t offset_interp = N_test * (N_test + 1) * N_interpolation_points/2;
        
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
        
        for (int i = 0; i < (N_test + 1) * N_interpolation_points; ++i) {
            REQUIRE(std::abs(host_interpolation_matrices[offset_interp + i] - NDG_host.interpolation_matrices_[N_test][i]) < error);
        }
    }

    cudaStreamDestroy(stream);
}