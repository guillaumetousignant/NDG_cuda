#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include <iostream>
#include <chrono>
#include <vector>

int main(void) {
    const int N_elements = 4;
    const int N_max = 8;
    const float x[2] = {-1.0f, 1.0f};
    const float CFL = 0.5f;
    const float u_max = 1.0f;
    std::vector<float> output_times{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    const int initial_N = N_max;
    const int N_interpolation_points = N_max * 8;
    const float delta_x = (x[1] - x[0])/N_elements;
    const float delta_t = CFL * delta_x/(u_max * N_max * N_max);

    std::cout << "CFL is: " << CFL << std::endl;
    std::cout << "Delta t is: " << delta_t << std::endl << std::endl;

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    NDG_t NDG(N_max, N_interpolation_points);
    Mesh_t Mesh(N_elements, initial_N, x[0], x[1]);
    Mesh.set_initial_conditions(NDG.nodes_);
    cudaDeviceSynchronize();

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "GPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    Mesh.solve(delta_t, output_times, NDG);
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    // Printing debug values
    //NDG.print();
    //Mesh.print();
    
    return 0;
}