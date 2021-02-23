#include <catch2/catch.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include <limits>
#include "float_types.h"
#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include "LegendrePolynomial_t.cuh"

TEST_CASE("Reduction", "Checks the reduction returns the right result."){
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

    constexpr int elements_blockSize = 32;
    const int elements_numBlocks = (mesh.N_elements_/2 + elements_blockSize - 1) / elements_blockSize;
    deviceFloat* g_odata;
    deviceFloat* host_g_odata = new deviceFloat[elements_numBlocks];
    cudaMalloc(&g_odata, elements_numBlocks * sizeof(deviceFloat));

    SEM::reduce_delta_t<elements_blockSize><<<elements_numBlocks, elements_blockSize>>>(CFL, mesh.N_elements_, mesh.elements_, g_odata);
    cudaMemcpy(host_g_odata, g_odata, elements_numBlocks * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    deviceFloat delta_t_min = std::numeric_limits<deviceFloat>::infinity();
    for (int i = 0; i < elements_numBlocks; ++i) {
        delta_t_min = min(delta_t_min, host_g_odata[i]);
    }

    const deviceFloat u_max = 1.0;
    const deviceFloat delta_x = (x[1] - x[0])/N_elements;
    const deviceFloat delta_t = CFL * delta_x/(u_max * N_test * N_test);

    REQUIRE(std::abs(delta_t_min - delta_t) < error);

    cudaStreamDestroy(stream);
    delete[] host_g_odata;
    cudaFree(g_odata);
}
