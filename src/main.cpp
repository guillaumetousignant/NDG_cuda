#include "float_types.h"
#include "NDG_host_t.cuh"
#include "Mesh_host_t.cuh"
#include "ChebyshevPolynomial_host_t.cuh"
#include "LegendrePolynomial_host_t.cuh"
#include <iostream>
#include <chrono>
#include <vector>

int main(void) {
    const int N_elements = 128;
    const int N_max = 8;
    const hostFloat x[2] = {-1.0, 1.0};
    const hostFloat CFL = 0.2;
    const hostFloat u_max = 1.0;
    std::vector<hostFloat> output_times{0.1, 0.2, 0.3, 0.4, 0.5};

    const int initial_N = N_max;
    const int N_interpolation_points = N_max * 8;
    const hostFloat delta_x = (x[1] - x[0])/N_elements;
    const hostFloat delta_t = CFL * delta_x/(u_max * N_max * N_max);

    std::cout << "CFL is: " << CFL << std::endl;
    std::cout << "Delta t is: " << delta_t << std::endl << std::endl;

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    NDG_host_t<LegendrePolynomial_host_t> NDG_host(N_max, N_interpolation_points);
    Mesh_host_t mesh(N_elements, initial_N, x[0], x[1]);
    mesh.set_initial_conditions(NDG.nodes_);

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "GPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    mesh.solve(delta_t, output_times, NDG);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    // Printing debug values
    //NDG.print();
    //mesh.print();
    
    return 0;
}