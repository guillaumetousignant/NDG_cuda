#include "float_types.h"
#include "NDG_host_t.h"
#include "Mesh_host_t.h"
#include "ChebyshevPolynomial_host_t.h"
#include "LegendrePolynomial_host_t.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <array>
#include <cmath>
#include <mpi.h>

constexpr hostFloat pi = 3.14159265358979323846;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    const size_t N_elements = 128;
    const int N_max = 16;
    const std::array<hostFloat, 2> x{-1.0, 1.0};
    const hostFloat max_splits = 2;
    const hostFloat delta_x_min = (x[1] - x[0])/(N_elements * std::pow(2, max_splits));
    const int adaptivity_interval = 100;
    const hostFloat CFL = 0.2f;
    const hostFloat viscosity = 0.1/pi;
    std::vector<hostFloat> output_times{0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00};

    const int initial_N = 6;
    const size_t N_interpolation_points = N_max * 8;

    // MPI ranks
    MPI_Comm node_communicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_communicator);
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    int node_rank;
    MPI_Comm_rank(node_communicator, &node_rank);
    int node_size;
    MPI_Comm_size(node_communicator, &node_size);

    std::cout << "Process with global id " << global_rank << "/" << global_size << " and local id " << node_rank << "/" << node_size << "." << std::endl;

    if (global_rank == 0) {
        std::cout << "CFL is: " << CFL << std::endl;
    }

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    SEM::NDG_host_t<SEM::LegendrePolynomial_host_t> NDG(N_max, N_interpolation_points);
    SEM::Mesh_host_t mesh(N_elements, initial_N, delta_x_min, x[0], x[1], adaptivity_interval);
    mesh.set_initial_conditions(NDG.nodes_);

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "CPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    mesh.solve(CFL, output_times, NDG, viscosity);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    MPI_Finalize();    
    return 0;
}