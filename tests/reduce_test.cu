#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include "helpers/float_types.h"
#include "entities/NDG_t.cuh"
#include "meshes/Mesh_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"

TEST_CASE("Reduction", "Checks the reduction returns the right result.") {
    const size_t N_elements = 1024; // N needs to be big enough for a value to be close to the max.
    const int N_max = 4;
    const int N_test = N_max;
    REQUIRE(N_test <= N_max);
    const size_t n_interpolation_points = N_max * 8;
    const std::array<deviceFloat, 2> x {-1.0, 1.0};
    const deviceFloat max_splits = 3;
    const deviceFloat delta_x_min = (x[1] - x[0])/(N_elements * std::pow(2, max_splits));
    const int adaptivity_interval = 100;
    const double error = 1e-4;
    const deviceFloat CFL = 0.5f;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    SEM::Device::Entities::NDG_t<SEM::Device::Polynomials::LegendrePolynomial_t> NDG(N_max, n_interpolation_points, stream);
    SEM::Device::Meshes::Mesh_t mesh(N_elements, N_test, delta_x_min, x[0], x[1], adaptivity_interval, stream);
    mesh.set_initial_conditions(NDG.nodes_.data());

    const deviceFloat delta_t_min = mesh.get_delta_t(CFL);

    const deviceFloat u_max = 1.0;
    const deviceFloat delta_x = (x[1] - x[0])/N_elements;
    const deviceFloat delta_t = CFL * delta_x * delta_x/(u_max * N_test * N_test);

    REQUIRE(std::abs(delta_t_min - delta_t) < error);

    cudaStreamDestroy(stream);
}
