#include "float_types.h"
#include "NDG_host_t.h"
#include "Mesh_host_t.h"
#include "ChebyshevPolynomial_host_t.h"
#include "LegendrePolynomial_host_t.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <array>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    const size_t N_elements = 16;
    const int N_max = 16;
    const std::array<hostFloat, 2> x {-1.0, 1.0};
    const hostFloat CFL = 0.1;
    const hostFloat viscosity = 0.0f;
    std::vector<hostFloat> output_times{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

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
    SEM::Mesh_host_t mesh(N_elements, initial_N, x[0], x[1]);
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