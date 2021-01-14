#include "float_types.h"
#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include <iostream>
#include <chrono>
#include <vector>

int main(void) {
    const size_t N_elements = 512;
    const int N_max = 8;
    const deviceFloat x[2] = {-1.0, 1.0};
    const deviceFloat CFL = 0.5;
    const deviceFloat u_max = 1.0;
    std::vector<deviceFloat> output_times{0.1, 0.2, 0.3, 0.4, 0.5};

    const int initial_N = N_max;
    const size_t N_interpolation_points = N_max * 8;
    const deviceFloat delta_x = (x[1] - x[0])/N_elements;
    const deviceFloat delta_t = CFL * delta_x/(u_max * N_max * N_max);

    std::cout << "CFL is: " << CFL << std::endl;
    std::cout << "Delta t is: " << delta_t << std::endl << std::endl;

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

    mesh.solve(delta_t, output_times, NDG);
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    // Printing debug values
    //NDG.print();
    //mesh.print();
    
    return 0;
}