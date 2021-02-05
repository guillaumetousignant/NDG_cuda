#include "float_types.h"
#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include <iostream>
#include <chrono>
#include <vector>
#include <array>

int main(void) {
    const size_t N_elements = 8;
    const int N_max = 16;
    const std::array<deviceFloat, 2> x {-1.0, 1.0};
    const deviceFloat CFL = 0.5f;
    const deviceFloat viscosity = 0.1;
    std::vector<deviceFloat> output_times{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    const int initial_N = 4;
    const size_t N_interpolation_points = N_max * 8;

    std::cout << "CFL is: " << CFL << std::endl;

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    NDG_t<LegendrePolynomial_t> NDG(N_max, N_interpolation_points);
    Mesh_t mesh(N_elements, initial_N, x[0], x[1]);
    mesh.set_initial_conditions(NDG.nodes_);
    cudaDeviceSynchronize();

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "GPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    mesh.solve(CFL, output_times, NDG, viscosity);
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    return 0;
}