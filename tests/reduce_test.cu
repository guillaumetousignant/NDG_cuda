#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include "float_types.h"
#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include "LegendrePolynomial_t.cuh"

TEST_CASE("Reduction", "Checks the reduction returns the right result."){
    const size_t N_elements = 128; // N needs to be big enough for a value to be close to the max.
    const int N_max = 4;
    const int N_test = N_max;
    REQUIRE(N_test <= N_max);
    const size_t N_interpolation_points = N_max * 8;
    const std::array<deviceFloat, 2> x {-1.0, 1.0};
    const double error = 1e-4;
    
    NDG_t<LegendrePolynomial_t> NDG(N_max, N_interpolation_points);
    Mesh_t mesh(N_elements, N_test, x[0], x[1]);
    mesh.set_initial_conditions(NDG.nodes_);

    constexpr int elements_blockSize = 32;
    const int elements_numBlocks = (mesh.N_elements_ + elements_blockSize - 1) / elements_blockSize;
    deviceFloat* g_odata;
    deviceFloat* host_g_odata = new deviceFloat[elements_numBlocks];
    cudaMalloc(&g_odata, mesh.N_elements_ * sizeof(deviceFloat));

    SEM::reduce_velocity<elements_blockSize><<<elements_numBlocks, elements_blockSize>>>(mesh.N_elements_, mesh.elements_, g_odata);
    cudaMemcpy(host_g_odata, g_odata, mesh.N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    deviceFloat phi_max = 0.0;
    for (int i = 0; i < elements_numBlocks; ++i) {
        phi_max = max(phi_max, host_g_odata[i]);
    }

    constexpr deviceFloat target = 1.0;
    REQUIRE(std::abs(phi_max - target) < error);

    delete[] host_g_odata;
    cudaFree(g_odata);
}
