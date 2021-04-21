#include "helpers/float_types.h"
#include "helpers/InputParser_t.h"
#include "entities/NDG_t.cuh"
#include "meshes/Mesh_t.cuh"
#include "meshes/Mesh2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <vector>
#include <array>
#include <mpi.h>

namespace fs = std::filesystem;

constexpr deviceFloat pi = 3.14159265358979323846;

auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);
    MPI_Init(&argc, &argv);

    const std::string input_mesh_file = input_parser.getCmdOption("--mesh");
    const fs::path mesh_file = (input_mesh_file.empty()) ? fs::current_path() / "meshes" / "mesh.cgns" : input_mesh_file;

    const size_t N_elements = 128;
    const int N_max = 16;
    const std::array<deviceFloat, 2> x{-1.0, 1.0};
    const deviceFloat max_splits = 3;
    const deviceFloat delta_x_min = (x[1] - x[0])/(N_elements * std::pow(2, max_splits));
    const int adaptivity_interval = 100;
    const deviceFloat CFL = 0.2f;
    const deviceFloat viscosity = 0.1/pi;
    std::vector<deviceFloat> output_times{0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00};

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

    const int n_proc_per_gpu = (node_size + deviceCount - 1)/deviceCount;
    const int device = node_rank/n_proc_per_gpu;
    const int device_rank = node_rank%n_proc_per_gpu;
    const int device_size = (device == deviceCount - 1) ? n_proc_per_gpu + node_size - n_proc_per_gpu * deviceCount : n_proc_per_gpu;
    cudaSetDevice(device);
    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    std::cout << "Process with global id " << global_rank << "/" << global_size << " and local id " << node_rank << "/" << node_size << " picked GPU " << device << "/" << deviceCount << " with stream " << device_rank << "/" << device_size << "." << std::endl;

    if (global_rank == 0) {
        std::cout << "CFL is: " << CFL << std::endl;
    }

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Meshes::Mesh2D_t mesh(mesh_file, initial_N, stream);
    cudaDeviceSynchronize();

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " GPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    //mesh.solve(CFL, output_times, NDG, viscosity);
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