#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include <mpi.h>
#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include "Face_t.cuh"
#include "Element_t.cuh"
#include "float_types.h"

TEST_CASE("Initial conditions solution value", "Checks the node values are correct after initial conditions.") {   
    MPI_Init(0, nullptr);

    const size_t N_elements = 16;
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max * 8;
    const std::array<deviceFloat, 2> x_span {-1.0, 1.0};
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::NDG_t<SEM::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Mesh_t mesh(N_elements, N_test, x_span[0], x_span[1], stream);
    mesh.set_initial_conditions(NDG.nodes_);
    cudaDeviceSynchronize();
    
    deviceFloat* x;
    deviceFloat* phi;
    deviceFloat* phi_prime;
    deviceFloat* intermediate;
    deviceFloat* x_L;
    deviceFloat* x_R;
    int* N;
    deviceFloat* sigma;
    bool* refine;
    bool* coarsen;
    deviceFloat* error;
    deviceFloat* host_x = new deviceFloat[mesh.N_elements_ * N_interpolation_points];
    deviceFloat* host_phi = new deviceFloat[mesh.N_elements_ * N_interpolation_points];
    deviceFloat* host_phi_prime = new deviceFloat[mesh.N_elements_ * N_interpolation_points];
    deviceFloat* host_intermediate = new deviceFloat[mesh.N_elements_ * N_interpolation_points];
    deviceFloat* host_x_L = new deviceFloat[mesh.N_elements_];
    deviceFloat* host_x_R = new deviceFloat[mesh.N_elements_];
    int* host_N = new int[mesh.N_elements_];
    deviceFloat* host_sigma = new deviceFloat[mesh.N_elements_];
    bool* host_refine = new bool[mesh.N_elements_];
    bool* host_coarsen = new bool[mesh.N_elements_];
    deviceFloat* host_error = new deviceFloat[mesh.N_elements_];
    cudaMalloc(&x, mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi, mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi_prime, mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&intermediate, mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&x_L, mesh.N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&x_R, mesh.N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&N, mesh.N_elements_ * sizeof(int));
    cudaMalloc(&sigma, mesh.N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&refine, mesh.N_elements_ * sizeof(bool));
    cudaMalloc(&coarsen, mesh.N_elements_ * sizeof(bool));
    cudaMalloc(&error, mesh.N_elements_ * sizeof(deviceFloat));

    SEM::get_solution<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, stream>>>(mesh.N_elements_, N_interpolation_points, mesh.elements_, NDG.interpolation_matrices_, x, phi, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);
    
    cudaMemcpy(host_x, x , mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi, phi, mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime, phi_prime, mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_intermediate, intermediate, mesh.N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_L, x_L, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_R, x_R, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_N, N, mesh.N_elements_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sigma, sigma, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_refine, refine, mesh.N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_coarsen, coarsen, mesh.N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_error, error, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_elements * N_interpolation_points; ++i) {
        const double expected = SEM::g(host_x[i]);
        REQUIRE(std::abs(expected - host_phi[i]) < max_error);
    }

    delete[] host_x;
    delete[] host_phi;
    delete[] host_phi_prime;
    delete[] host_intermediate;
    delete[] host_x_L;
    delete[] host_x_R;
    delete[] host_N;
    delete[] host_sigma;
    delete[] host_refine;
    delete[] host_coarsen;
    delete[] host_error;
    cudaFree(x);
    cudaFree(phi);
    cudaFree(phi_prime);
    cudaFree(intermediate);
    cudaFree(x_L);
    cudaFree(x_R);
    cudaFree(N);
    cudaFree(sigma);
    cudaFree(refine);
    cudaFree(coarsen);
    cudaFree(error);
    MPI_Finalize();
}

TEST_CASE("Initial conditions boundary values", "Checks the extrapolated boundary values are correct after initial conditions.") {   
    MPI_Init(0, nullptr);

    const size_t N_elements = 16;
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max * 8;
    const std::array<deviceFloat, 2> x_span {-1.0, 1.0};
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::NDG_t<SEM::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Mesh_t mesh(N_elements, N_test, x_span[0], x_span[1], stream);
    mesh.set_initial_conditions(NDG.nodes_);
    cudaDeviceSynchronize();
    SEM::interpolate_to_boundaries<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, stream>>>(mesh.N_elements_, mesh.elements_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_, NDG.lagrange_interpolant_derivative_left_, NDG.lagrange_interpolant_derivative_right_);
    cudaDeviceSynchronize();

    SEM::Face_t* host_faces = new SEM::Face_t[mesh.N_faces_];
    SEM::Element_t* host_elements = new SEM::Element_t[mesh.N_elements_ + mesh.N_local_boundaries_ + mesh.N_MPI_boundaries_];
    cudaMemcpy(host_faces, mesh.faces_, mesh.N_faces_ * sizeof(SEM::Face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_elements, mesh.elements_, (mesh.N_elements_ + mesh.N_local_boundaries_ + mesh.N_MPI_boundaries_) * sizeof(SEM::Element_t), cudaMemcpyDeviceToHost);
    // Invalidate GPU pointers, or else they will be deleted on the CPU, where they point to random stuff
    for (size_t i = 0; i < mesh.N_elements_ + mesh.N_local_boundaries_ + mesh.N_MPI_boundaries_; ++i) {
        host_elements[i].phi_ = nullptr;
        host_elements[i].q_ = nullptr;
        host_elements[i].ux_ = nullptr;
        host_elements[i].phi_prime_ = nullptr;
        host_elements[i].intermediate_ = nullptr;
    }
    
    for (int i = 0; i < N_elements; ++i) {
        const double phi_L_expected = SEM::g(host_elements[i].x_[0]);
        const double phi_R_expected = SEM::g(host_elements[i].x_[1]);

        REQUIRE(std::abs(phi_L_expected - host_elements[i].phi_L_) < max_error);
        REQUIRE(std::abs(phi_R_expected - host_elements[i].phi_R_) < max_error);
    }

    for (int i = 0; i < N_elements; ++i) {
        const double phi_prime_L_expected = SEM::g_prime(host_elements[i].x_[0]);
        const double phi_prime_R_expected = SEM::g_prime(host_elements[i].x_[1]);

        REQUIRE(std::abs(phi_prime_L_expected - host_elements[i].phi_prime_L_) < max_error);
        REQUIRE(std::abs(phi_prime_R_expected - host_elements[i].phi_prime_R_) < max_error);
    }

    delete[] host_faces;
    delete[] host_elements;
}
