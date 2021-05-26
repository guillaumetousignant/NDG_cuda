#include "helpers/float_types.h"
#include "helpers/InputParser_t.h"
#include "helpers/DataWriter_t.h"
#include "entities/NDG_t.cuh"
#include "meshes/Mesh2D_t.cuh"
#include "solvers/Solver2D_t.cuh"
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

auto get_input_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string input_mesh_path = input_parser.getCmdOption("--mesh");
    if (!input_mesh_path.empty()) {
        const fs::path mesh_file = input_mesh_path;
        fs::create_directory(mesh_file.parent_path());
        return (mesh_file.extension().empty()) ? mesh_file / ".cgns" : mesh_file;
    }
    else {
        const std::string input_filename = input_parser.getCmdOption("--mesh_filename");
        const std::string mesh_filename = (input_filename.empty()) ? "mesh.cgns" : input_filename;

        const std::string input_mesh_dir = input_parser.getCmdOption("--mesh_directory");
        const fs::path mesh_dir = (input_mesh_dir.empty()) ? fs::current_path() / "meshes" : fs::path(input_mesh_dir);

        fs::create_directory(mesh_dir);
        const fs::path mesh_file = mesh_dir / mesh_filename;
        return (mesh_file.extension().empty()) ? mesh_file / ".cgns" : mesh_file;
    }
}

auto get_output_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string input_save_path = input_parser.getCmdOption("--output");
    if (!input_save_path.empty()) {
        const fs::path save_file = input_save_path;
        fs::create_directory(save_file.parent_path());
        return (save_file.extension().empty()) ? save_file / ".pvtu" : save_file;
    }
    else {
        const std::string input_filename = input_parser.getCmdOption("--output_filename");
        const std::string save_filename = (input_filename.empty()) ? "output.pvtu" : input_filename;

        const std::string input_save_dir = input_parser.getCmdOption("--output_directory");
        const fs::path save_dir = (input_save_dir.empty()) ? fs::current_path() / "data" : fs::path(input_save_dir);

        fs::create_directory(save_dir);
        const fs::path save_file = save_dir / save_filename;
        return (save_file.extension().empty()) ? save_file / ".pvtu" : save_file;
    }
}

auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);
    if (input_parser.cmdOptionExists("--help") || input_parser.cmdOptionExists("-h")) {
        std::cout << "Spectral element method 2D unstructured solver" << std::endl;
        std::cout << '\t' <<  "Solves the 2D wave equation on 2D unstructured meshes. The meshes use the CGNS HDF5 format, and output uses the VTK format." << std::endl << std::endl;
        std::cout << "Available options:" << std::endl;
        std::cout << '\t' <<  "--mesh"             <<  '\t' <<  "Full path of the input mesh file. Overrides mesh_filename and mesh_directory if set." << std::endl;
        std::cout << '\t' <<  "--mesh_filename"    <<  '\t' <<  "File name of the input mesh file. Defaults to [mesh.cgns]" << std::endl;
        std::cout << '\t' <<  "--mesh_directory"   <<  '\t' <<  "Directory of the input mesh file. Defaults to [./meshes/]" << std::endl;
        std::cout << '\t' <<  "--output"           <<  '\t' <<  "Full path of the output data file. Overrides output_filename and output_directory if set." << std::endl;
        std::cout << '\t' <<  "--output_filename"  <<  '\t' <<  "File name of the output data file. Defaults to [output.pvtu]" << std::endl;
        std::cout << '\t' <<  "--output_directory" <<  '\t' <<  "Directory of the output data file. Defaults to [./data/]" << std::endl;
        exit(0);
    }

    MPI_Init(&argc, &argv);

    const fs::path mesh_file = get_input_file(input_parser);
    const fs::path output_file = get_output_file(input_parser);

    const size_t N_elements = 128;
    const int N_max = 8;
    const std::array<deviceFloat, 2> x{-1.0, 1.0};
    const deviceFloat max_splits = 3;
    const deviceFloat delta_x_min = (x[1] - x[0])/(N_elements * std::pow(2, max_splits));
    const int adaptivity_interval = 100;
    const deviceFloat CFL = 0.2f;
    const deviceFloat viscosity = 0.1/pi;
    std::vector<deviceFloat> output_times{0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00};

    const int initial_N = 4;
    const size_t N_interpolation_points = std::pow(N_max, 2);

    // MPI ranks
    MPI_Comm node_communicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_communicator);
    int global_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    int local_rank = -1;
    MPI_Comm_rank(node_communicator, &local_rank);
    int local_size = -1;
    MPI_Comm_size(node_communicator, &local_size);

    std::vector<int> local_ranks(global_size);
    MPI_Allgather(&local_rank, 1, MPI_INT, local_ranks.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int node_rank = 0;
    int node_size = 0;
    for (int i = 0; i < global_size; ++i) {
        if (local_ranks[i] == 0) {
            ++node_size;
            if (i < global_rank) {
                ++node_rank;
            }
        }
    }

    // Device selection
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (local_rank == 0) {
        switch(deviceCount) {
            case 0:
                std::cout << "Node " << node_rank << " has no Cuda devices." << std::endl;
                break;
            case 1:
                std::cout << "Node " << node_rank << " has one Cuda device:" << std::endl;
                break;
            default:
                std::cout << "Node " << node_rank << " has " << deviceCount << " Cuda devices:" << std::endl;
                break;
        }
        for (int device = 0; device < deviceCount; ++device) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device);
            std::cout << '\t' << "Device #" << device << " (" << deviceProp.name << ") has compute capability " << deviceProp.major << "." << deviceProp.minor << "." << std::endl;
        }
    }

    const int n_proc_per_gpu = (local_size + deviceCount - 1)/deviceCount;
    const int device = local_rank/n_proc_per_gpu;
    const int device_rank = local_rank%n_proc_per_gpu;
    const int device_size = (device == deviceCount - 1) ? n_proc_per_gpu + local_size - n_proc_per_gpu * deviceCount : n_proc_per_gpu;

    cudaSetDevice(device);
    if (device_rank == 0) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        const cudaError_t code = cudaDeviceSetLimit(cudaLimitMallocHeapSize, deviceProp.totalGlobalMem/2);
        if (code != cudaSuccess) {
            std::cerr << "GPU memory request failed: " << cudaGetErrorString(code) << std::endl;
            exit(1);
        }
        size_t device_heap_limit = 0;
        cudaDeviceGetLimit(&device_heap_limit, cudaLimitMallocHeapSize);
        std::cout << "Node " << node_rank << ", device " << device << " requested " << deviceProp.totalGlobalMem << " bytes and got " << device_heap_limit << " bytes." << std::endl;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    std::cout << "Process with global id " << global_rank << "/" << global_size << " on node " << node_rank << "/" << node_size << " and local id " << local_rank << "/" << local_size << " picked GPU " << device << "/" << deviceCount << " with stream " << device_rank << "/" << device_size << "." << std::endl;

    if (global_rank == 0) {
        std::cout << "CFL is: " << CFL << std::endl;
    }

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Meshes::Mesh2D_t mesh(mesh_file, initial_N, NDG.nodes_, stream);
    SEM::Solvers::Solver2D_t solver(CFL, output_times, viscosity);
    SEM::Helpers::DataWriter_t data_writer(output_file);
    mesh.initial_conditions(NDG.nodes_.data());
    cudaDeviceSynchronize();

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " GPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    solver.solve(NDG, mesh, data_writer);
    //mesh.print();
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