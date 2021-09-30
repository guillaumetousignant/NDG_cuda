#include <catch2/catch.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <mpi.h>
#include "entities/NDG_t.cuh"
#include "meshes/Mesh_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "entities/Face_t.cuh"
#include "entities/Element_t.cuh"
#include "helpers/float_types.h"

TEST_CASE("Initial conditions solution value", "Checks the node values are correct after initial conditions.") {   
    const size_t N_elements = 16;
    const int N_max = 16;
    const int N_test = 16;
    const size_t n_interpolation_points = N_max * 8;
    const std::array<deviceFloat, 2> x_span {-1.0, 1.0};
    const deviceFloat max_splits = 3;
    const deviceFloat delta_x_min = (x_span[1] - x_span[0])/(N_elements * std::pow(2, max_splits));
    const int adaptivity_interval = 100;
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, n_interpolation_points, stream);
    SEM::Meshes::Mesh_t mesh(N_elements, N_test, delta_x_min, x_span[0], x_span[1], adaptivity_interval, stream);
    mesh.set_initial_conditions(NDG.nodes_.data());
    
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
    deviceFloat* delta_x;
    std::vector<deviceFloat> host_x(mesh.N_elements_ * n_interpolation_points);
    std::vector<deviceFloat> host_phi(mesh.N_elements_ * n_interpolation_points);
    std::vector<deviceFloat> host_phi_prime(mesh.N_elements_ * n_interpolation_points);
    std::vector<deviceFloat> host_intermediate(mesh.N_elements_ * n_interpolation_points);
    std::vector<deviceFloat> host_x_L(mesh.N_elements_);
    std::vector<deviceFloat> host_x_R(mesh.N_elements_);
    std::vector<int> host_N(mesh.N_elements_);
    std::vector<deviceFloat> host_sigma(mesh.N_elements_);
    bool* host_refine = new bool[mesh.N_elements_]; // Vectors of bools can be messed-up by some implementations
    bool* host_coarsen = new bool[mesh.N_elements_]; // Like they won't be an array of bools but packed in integers, in which case getting them from Cuda will fail.
    std::vector<deviceFloat> host_error(mesh.N_elements_);
    std::vector<deviceFloat> host_delta_x(mesh.N_elements_);
    cudaMallocAsync(&x, mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), stream);
    cudaMallocAsync(&phi, mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), stream);
    cudaMallocAsync(&phi_prime, mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), stream);
    cudaMallocAsync(&intermediate, mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), stream);
    cudaMallocAsync(&x_L, mesh.N_elements_ * sizeof(deviceFloat), stream);
    cudaMallocAsync(&x_R, mesh.N_elements_ * sizeof(deviceFloat), stream);
    cudaMallocAsync(&N, mesh.N_elements_ * sizeof(int), stream);
    cudaMallocAsync(&sigma, mesh.N_elements_ * sizeof(deviceFloat), stream);
    cudaMallocAsync(&refine, mesh.N_elements_ * sizeof(bool), stream);
    cudaMallocAsync(&coarsen, mesh.N_elements_ * sizeof(bool), stream);
    cudaMallocAsync(&error, mesh.N_elements_ * sizeof(deviceFloat), stream);
    cudaMallocAsync(&delta_x, mesh.N_elements_ * sizeof(deviceFloat), stream);

    SEM::Entities::get_solution<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, stream>>>(mesh.N_elements_, n_interpolation_points, mesh.elements_, NDG.interpolation_matrices_.data(), x, phi, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error, delta_x);
    
    cudaMemcpy(host_x.data(), x , mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi.data(), phi, mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime.data(), phi_prime, mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_intermediate.data(), intermediate, mesh.N_elements_ * n_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_L.data(), x_L, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_R.data(), x_R, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_N.data(), N, mesh.N_elements_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sigma.data(), sigma, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_refine, refine, mesh.N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_coarsen, coarsen, mesh.N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_error.data(), error, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_delta_x.data(), delta_x, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_elements * n_interpolation_points; ++i) {
        const double expected = SEM::Entities::g(host_x[i]);
        REQUIRE(std::abs(expected - host_phi[i]) < max_error);
    }

    delete[] host_refine;
    delete[] host_coarsen;
    cudaFreeAsync(x, stream);
    cudaFreeAsync(phi, stream);
    cudaFreeAsync(phi_prime, stream);
    cudaFreeAsync(intermediate, stream);
    cudaFreeAsync(x_L, stream);
    cudaFreeAsync(x_R, stream);
    cudaFreeAsync(N, stream);
    cudaFreeAsync(sigma, stream);
    cudaFreeAsync(refine, stream);
    cudaFreeAsync(coarsen, stream);
    cudaFreeAsync(error, stream);
    cudaFreeAsync(delta_x, stream);

    cudaStreamDestroy(stream);
}

TEST_CASE("Initial conditions boundary values", "Checks the extrapolated boundary values are correct after initial conditions.") {   
    const size_t N_elements = 16;
    const int N_max = 16;
    const int N_test = 16;
    const size_t n_interpolation_points = N_max * 8;
    const std::array<deviceFloat, 2> x_span {-1.0, 1.0};
    const deviceFloat max_splits = 3;
    const deviceFloat delta_x_min = (x_span[1] - x_span[0])/(N_elements * std::pow(2, max_splits));
    const int adaptivity_interval = 100;
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, n_interpolation_points, stream);
    SEM::Meshes::Mesh_t mesh(N_elements, N_test, delta_x_min, x_span[0], x_span[1], adaptivity_interval, stream);
    mesh.set_initial_conditions(NDG.nodes_.data());
    SEM::Entities::interpolate_to_boundaries<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, stream>>>(mesh.N_elements_, mesh.elements_, NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
    mesh.boundary_conditions();
    SEM::Meshes::calculate_fluxes<<<mesh.faces_numBlocks_, mesh.faces_blockSize_, 0, stream>>>(mesh.N_faces_, mesh.faces_, mesh.elements_);
    SEM::Meshes::compute_dg_derivative<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, stream>>>(mesh.N_elements_, mesh.elements_, mesh.faces_, NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
    SEM::Entities::interpolate_q_to_boundaries<<<mesh.elements_numBlocks_, mesh.elements_blockSize_, 0, stream>>>(mesh.N_elements_, mesh.elements_, NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());

    std::vector<SEM::Entities::Face_t> host_faces(mesh.N_faces_);
    std::vector<SEM::Entities::Element_t> host_elements(mesh.N_elements_ + mesh.N_local_boundaries_ + mesh.N_MPI_boundaries_);
    cudaMemcpy(host_faces.data(), mesh.faces_, mesh.N_faces_ * sizeof(SEM::Entities::Face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_elements.data(), mesh.elements_, (mesh.N_elements_ + mesh.N_local_boundaries_ + mesh.N_MPI_boundaries_) * sizeof(SEM::Entities::Element_t), cudaMemcpyDeviceToHost);

    // Invalidate GPU pointers, or else they will be deleted on the CPU, where they point to random stuff
    for (size_t i = 0; i < mesh.N_elements_ + mesh.N_local_boundaries_ + mesh.N_MPI_boundaries_; ++i) {
        host_elements[i].phi_ = nullptr;
        host_elements[i].q_ = nullptr;
        host_elements[i].ux_ = nullptr;
        host_elements[i].phi_prime_ = nullptr;
        host_elements[i].intermediate_ = nullptr;
    }
    
    for (int i = 0; i < N_elements; ++i) {
        const double phi_L_expected = SEM::Entities::g(host_elements[i].x_[0]);
        const double phi_R_expected = SEM::Entities::g(host_elements[i].x_[1]);

        REQUIRE(std::abs(phi_L_expected - host_elements[i].phi_L_) < max_error);
        REQUIRE(std::abs(phi_R_expected - host_elements[i].phi_R_) < max_error);
    }

    for (int i = 0; i < N_elements; ++i) {
        const double phi_prime_L_expected = SEM::Entities::g_prime(host_elements[i].x_[0]);
        const double phi_prime_R_expected = SEM::Entities::g_prime(host_elements[i].x_[1]);

        REQUIRE(std::abs(phi_prime_L_expected + host_elements[i].phi_prime_L_) < max_error);
        REQUIRE(std::abs(phi_prime_R_expected + host_elements[i].phi_prime_R_) < max_error);
    }

    cudaStreamDestroy(stream);
}
