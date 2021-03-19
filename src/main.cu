#include "float_types.h"
#include "NDG_t.cuh"
#include "Mesh_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include <iostream>
#include <chrono>
#include <vector>
#include <array>
#include <mpi.h>

constexpr hostFloat pi = 3.14159265358979323846;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    const size_t N_elements = 16;
    const int N_max = 16;
    const std::array<deviceFloat, 2> x {-1.0, 1.0};
    const deviceFloat CFL = 0.2f;
    const deviceFloat viscosity = 0.1;
    std::vector<deviceFloat> output_times{0.02f, 0.04f, 0.06f, 0.08f, 0.10f, 0.12f, 0.14f, 0.16f, 0.18f, 0.20f, 0.22f, 0.24f, 0.26f, 0.28f, 0.30f, 0.32f, 0.34f, 0.36f, 0.38f, 0.40f, 0.42f, 0.44f, 0.46f, 0.48f, 0.50f};

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

    // Device selection
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (node_rank == 0) {
        switch(deviceCount) {
            case 0:
                std::cout << "There are no Cuda devices." << std::endl;
                break;
            case 1:
                std::cout << "There is one Cuda device:" << std::endl;
                break;
            default:
                std::cout << "There are " << deviceCount << " Cuda devices:" << std::endl;
                break;
        }
        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            std::cout << '\t' << "Device #" << device << " (" << deviceProp.name << ") has compute capability " << deviceProp.major << "." << deviceProp.minor << "." << std::endl;
        }
    }

    int n_proc_per_gpu = (node_size + deviceCount - 1)/deviceCount;
    int device = node_rank/n_proc_per_gpu;
    int device_rank = node_rank%n_proc_per_gpu;
    int device_size = (device == deviceCount - 1) ? n_proc_per_gpu + node_size - n_proc_per_gpu * deviceCount : n_proc_per_gpu;
    cudaSetDevice(device);
    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    std::cout << "Process with global id " << global_rank << "/" << global_size << " and local id " << node_rank << "/" << node_size << " picked GPU " << device << "/" << deviceCount << " with stream " << device_rank << "/" << device_size << "." << std::endl;

    if (global_rank == 0) {
        std::cout << "CFL is: " << CFL << std::endl;
    }

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    SEM::NDG_t<SEM::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Mesh_t mesh(N_elements, initial_N, x[0], x[1], stream);
    mesh.set_initial_conditions(NDG.nodes_);
    cudaDeviceSynchronize();

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " GPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    mesh.solve(CFL, output_times, NDG, viscosity);
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    cudaStreamDestroy(stream);
    MPI_Finalize();
    return 0;
}