#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include "NDG_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include "float_types.h"

TEST_CASE("ChebyshevPolynomials", "Checks the Chebyshev polynomials"){
    const int N_max = 16;
    const int N_test = 16;
    const int N_interpolation_points = N_max * 8;
    const int offset_1D = N_test * (N_test + 1) /2;
    const int offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
    const int offset_interp = N_test * (N_test + 1) * N_interpolation_points /2;
    const double error = 1e-6;
    
    NDG_t<ChebyshevPolynomial_t> NDG(N_max, N_interpolation_points);

    deviceFloat* host_nodes = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_weights = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_barycentric_weights = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_lagrange_interpolant_left = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_lagrange_interpolant_right = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_derivative_matrices = new deviceFloat[NDG.matrix_length_];
    deviceFloat* host_derivative_matrices_hat = new deviceFloat[NDG.matrix_length_];
    deviceFloat* host_interpolation_matrices = new deviceFloat[NDG.interpolation_length_];

    cudaDeviceSynchronize();

    cudaMemcpy(host_nodes, NDG.nodes_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights, NDG.weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights, NDG.barycentric_weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left, NDG.lagrange_interpolant_left_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right, NDG.lagrange_interpolant_right_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices, NDG.derivative_matrices_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices_hat, NDG.derivative_matrices_hat_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_interpolation_matrices, NDG.interpolation_matrices_, NDG.interpolation_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    REQUIRE(N_test <= N_max);

    SECTION("Polynomial nodes") {
        const double nodes[] = {-0.9957341762950345218712,
                                -0.9618256431728190704088,
                                -0.895163291355062322067,
                                -0.7980172272802395033328,
                                -0.673695643646557211713,
                                -0.5264321628773558002446,
                                -0.3612416661871529487447,
                                -0.1837495178165703315744,
                                 0,
                                 0.183749517816570331574,
                                 0.3612416661871529487447,
                                 0.5264321628773558002446,
                                 0.6736956436465572117127,
                                 0.7980172272802395033328,
                                 0.895163291355062322067,
                                 0.9618256431728190704088,
                                 0.9957341762950345218712};

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(nodes[i] - host_nodes[offset_1D + i]) < error);
        }
    }

    SECTION("Polynomial weights") {
        const double weights[] = {0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743,
                                  0.1847995678582231316743};
            
        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(weights[i] - host_weights[offset_1D + i]) < error);
        }
    }

    delete[] host_nodes;
    delete[] host_weights;
    delete[] host_barycentric_weights;
    delete[] host_lagrange_interpolant_left;
    delete[] host_lagrange_interpolant_right;
    delete[] host_derivative_matrices;
    delete[] host_derivative_matrices_hat;
    delete[] host_interpolation_matrices;
}

TEST_CASE("LegendrePolynomials", "Checks the Legendre polynomials"){
    const int N_max = 16;
    const int N_test = 16;
    const int N_interpolation_points = N_max * 8;
    const int offset_1D = N_test * (N_test + 1) /2;
    const int offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
    const int offset_interp = N_test * (N_test + 1) * N_interpolation_points /2;
    const double error = 1e-6;
    
    NDG_t<LegendrePolynomial_t> NDG(N_max, N_interpolation_points);

    deviceFloat* host_nodes = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_weights = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_barycentric_weights = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_lagrange_interpolant_left = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_lagrange_interpolant_right = new deviceFloat[NDG.vector_length_];
    deviceFloat* host_derivative_matrices = new deviceFloat[NDG.matrix_length_];
    deviceFloat* host_derivative_matrices_hat = new deviceFloat[NDG.matrix_length_];
    deviceFloat* host_interpolation_matrices = new deviceFloat[NDG.interpolation_length_];

    cudaDeviceSynchronize();

    cudaMemcpy(host_nodes, NDG.nodes_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights, NDG.weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights, NDG.barycentric_weights_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left, NDG.lagrange_interpolant_left_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right, NDG.lagrange_interpolant_right_, NDG.vector_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices, NDG.derivative_matrices_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_derivative_matrices_hat, NDG.derivative_matrices_hat_, NDG.matrix_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_interpolation_matrices, NDG.interpolation_matrices_, NDG.interpolation_length_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    REQUIRE(N_test <= N_max);

    SECTION("Polynomial nodes") {
        const double nodes[] = {-0.9905754753144173356754,
                                -0.9506755217687677612227,
                                -0.880239153726985902123,
                                -0.7815140038968014069252,
                                -0.6576711592166907658503,
                                -0.5126905370864769678863,
                                -0.3512317634538763152972,
                                -0.1784841814958478558507,
                                 0,
                                 0.1784841814958478558507,
                                 0.3512317634538763152972,
                                 0.5126905370864769678863,
                                 0.6576711592166907658503,
                                 0.7815140038968014069252,
                                 0.880239153726985902123,
                                 0.9506755217687677612227,
                                 0.9905754753144173356754};

        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(nodes[i] - host_nodes[offset_1D + i]) < error);
        }
    }

    SECTION("Polynomial weights") {
        const double weights[] = {0.0241483028685479319601,
                                  0.0554595293739872011294,
                                  0.0850361483171791808835,
                                  0.111883847193403971095,
                                  0.1351363684685254732863,
                                  0.1540457610768102880814,
                                  0.16800410215645004451,
                                  0.1765627053669926463253,
                                  0.1794464703562065254583,
                                  0.1765627053669926463253,
                                  0.16800410215645004451,
                                  0.1540457610768102880814,
                                  0.1351363684685254732863,
                                  0.111883847193403971095,
                                  0.0850361483171791808835,
                                  0.055459529373987201129,
                                  0.0241483028685479319601};
            
        for (int i = 0; i <= N_test; ++i) {
            REQUIRE(std::abs(weights[i] - host_weights[offset_1D + i]) < error);
        }
    }

    delete[] host_nodes;
    delete[] host_weights;
    delete[] host_barycentric_weights;
    delete[] host_lagrange_interpolant_left;
    delete[] host_lagrange_interpolant_right;
    delete[] host_derivative_matrices;
    delete[] host_derivative_matrices_hat;
    delete[] host_interpolation_matrices;
}