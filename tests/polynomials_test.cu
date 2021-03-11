#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include "NDG_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include "float_types.h"

constexpr double pi = 3.14159265358979323846;

TEST_CASE("ChebyshevPolynomials", "Checks the Chebyshev polynomials"){
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max * 8;
    const size_t offset_1D = N_test * (N_test + 1) /2;
    const size_t offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
    const double error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::NDG_t<SEM::ChebyshevPolynomial_t> NDG(N_max, N_interpolation_points, stream);

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

    const std::array<double, N_test+1> nodes {-0.9957341762950345218712,
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

    const std::array<double, N_test+1> weights {0.1847995678582231316743,
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

    std::array<double, N_test+1> phi;
    for (int i = 0; i <= N_test; ++i) {
        phi[i] = std::sin(pi * host_nodes[offset_1D + i]);
    }

    std::array<double, N_test+1> phi_prime_expected;
    for (int i = 0; i <= N_test; ++i) {
        phi_prime_expected[i] = pi * std::cos(pi * host_nodes[offset_1D + i]);
    }

    std::array<double, N_test+1> phi_prime;
    for (int i = 0; i <= N_test; ++i) {
        phi_prime[i] = 0.0;
        for (int j = 0; j <= N_test; ++j) {
            phi_prime[i] += host_derivative_matrices[offset_2D + i * (N_test + 1) + j] * phi[j];
        }
    }
    
    for (int i = 0; i <= N_test; ++i) {
        REQUIRE(std::abs(phi_prime[i] - phi_prime_expected[i]) < error*100);
    }

    double phi_L_expected = std::sin(pi * -1.0);
    double phi_R_expected = std::sin(pi * 1.0);
    double phi_L = 0.0;
    double phi_R = 0.0;

    for (int j = 0; j <= N_test; ++j) {
        phi_L += host_lagrange_interpolant_left[offset_1D + j] * phi[j];
        phi_R += host_lagrange_interpolant_right[offset_1D + j] * phi[j];
    }
    
    REQUIRE(std::abs(phi_L - phi_L_expected) < error);
    REQUIRE(std::abs(phi_R - phi_R_expected) < error);

    double phi_prime_L = 0.0;
    double phi_prime_R = 0.0;
    double phi_prime_L_expected = pi * std::cos(pi * -1.0);
    double phi_prime_R_expected = pi * std::cos(pi * 1.0);

    for (int j = 0; j <= N_test; ++j) {
        phi_prime_L += host_lagrange_interpolant_derivative_left[offset_1D + j] * (phi_L - phi[j]);
        phi_prime_R += host_lagrange_interpolant_derivative_right[offset_1D + j] * (phi_R - phi[j]);
    }
    
    REQUIRE(std::abs(phi_prime_L - phi_prime_L_expected) < error);
    REQUIRE(std::abs(phi_prime_R - phi_prime_R_expected) < error);

    cudaStreamDestroy(stream);
}

TEST_CASE("LegendrePolynomials", "Checks the Legendre polynomials"){
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max * 8;
    const size_t offset_1D = N_test * (N_test + 1) /2;
    const size_t offset_2D = N_test * (N_test + 1) * (2 * N_test + 1) /6;
    const double error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::NDG_t<SEM::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);

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

    const std::array<double, N_test+1> nodes {-0.9905754753144173356754,
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

    const std::array<double, N_test+1> weights {0.0241483028685479319601,
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

    std::array<double, N_test+1> phi;
    for (int i = 0; i <= N_test; ++i) {
        phi[i] = std::sin(pi * host_nodes[offset_1D + i]);
    }

    std::array<double, N_test+1> phi_prime_expected;
    for (int i = 0; i <= N_test; ++i) {
        phi_prime_expected[i] = pi * std::cos(pi * host_nodes[offset_1D + i]);
    }

    std::array<double, N_test+1> phi_prime;
    for (int i = 0; i <= N_test; ++i) {
        phi_prime[i] = 0.0;
        for (int j = 0; j <= N_test; ++j) {
            phi_prime[i] += host_derivative_matrices[offset_2D + i * (N_test + 1) + j] * phi[j];
        }
    }
    
    for (int i = 0; i <= N_test; ++i) {
        REQUIRE(std::abs(phi_prime[i] - phi_prime_expected[i]) < error*100);
    }

    double phi_L_expected = std::sin(pi * -1.0);
    double phi_R_expected = std::sin(pi * 1.0);
    double phi_L = 0.0;
    double phi_R = 0.0;

    for (int j = 0; j <= N_test; ++j) {
        phi_L += host_lagrange_interpolant_left[offset_1D + j] * phi[j];
        phi_R += host_lagrange_interpolant_right[offset_1D + j] * phi[j];
    }
    
    REQUIRE(std::abs(phi_L - phi_L_expected) < error);
    REQUIRE(std::abs(phi_R - phi_R_expected) < error);

    double phi_prime_L = 0.0;
    double phi_prime_R = 0.0;
    double phi_prime_L_expected = pi * std::cos(pi * -1.0);
    double phi_prime_R_expected = pi * std::cos(pi * 1.0);

    for (int j = 0; j <= N_test; ++j) {
        phi_prime_L += host_lagrange_interpolant_derivative_left[offset_1D + j] * (phi_L - phi[j]);
        phi_prime_R += host_lagrange_interpolant_derivative_right[offset_1D + j] * (phi_R - phi[j]);
    }
    
    REQUIRE(std::abs(phi_prime_L - phi_prime_L_expected) < error);
    REQUIRE(std::abs(phi_prime_R - phi_prime_R_expected) < error);

    cudaStreamDestroy(stream);
}