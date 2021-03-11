#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include <limits>
#include <mpi.h>
#include "float_types.h"
#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include "LegendrePolynomial_t.cuh"

TEST_CASE("Reduction", "Checks the reduction returns the right result.") {
    MPI_Init(0, nullptr);

    const size_t N_elements = 1024; // N needs to be big enough for a value to be close to the max.
    const int N_max = 4;
    const int N_test = N_max;
    REQUIRE(N_test <= N_max);
    const size_t N_interpolation_points = N_max * 8;
    const std::array<deviceFloat, 2> x {-1.0, 1.0};
    const double error = 1e-4;
    const deviceFloat CFL = 0.5f;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    SEM::NDG_t<SEM::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Mesh_t mesh(N_elements, N_test, x[0], x[1], stream);
    mesh.set_initial_conditions(NDG.nodes_);
    
    const deviceFloat delta_t_min = mesh.get_delta_t(CFL);

    const deviceFloat u_max = 1.0;
    const deviceFloat delta_x = (x[1] - x[0])/N_elements;
    const deviceFloat delta_t = CFL * delta_x * delta_x/(u_max * N_test * N_test);

    REQUIRE(std::abs(delta_t_min - delta_t) < error);

    cudaStreamDestroy(stream);
    delete[] host_g_odata;
    cudaFree(g_odata);
    MPI_Finalize();
}
