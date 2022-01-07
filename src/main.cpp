#include "helpers/float_types.h"
#include "helpers/InputParser_t.h"
#include "helpers/DataWriter_t.h"
#include "entities/NDG_t.h"
#include "meshes/Mesh2D_t.h"
#include "solvers/Solver2D_t.h"
#include "polynomials/LegendrePolynomial_t.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <vector>
#include <array>
#include <cmath>
#include <mpi.h>

#include <chrono>
#include <thread>

namespace fs = std::filesystem;

constexpr hostFloat pi = 3.14159265358979323846;

auto get_input_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string input_mesh_path = input_parser.getCmdOption("--mesh");
    if (!input_mesh_path.empty()) {
        const fs::path mesh_file = input_mesh_path;
        fs::create_directory(mesh_file.parent_path());
        return (mesh_file.extension().empty()) ? mesh_file / ".cgns" : mesh_file;
    }
    else {
        std::string filename_default("mesh.cgns");
        const std::string mesh_filename = input_parser.getCmdOptionOr("--mesh_filename", filename_default);
        const fs::path mesh_dir = input_parser.getCmdOptionOr("--mesh_directory", fs::current_path() / "meshes");

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
        const std::string filename_default("output.pvtu");
        const std::string save_filename = input_parser.getCmdOptionOr("--output_filename", filename_default);
        const fs::path save_dir = input_parser.getCmdOptionOr("--output_directory", fs::current_path() / "data");

        fs::create_directory(save_dir);
        const fs::path save_file = save_dir / save_filename;
        return (save_file.extension().empty()) ? save_file / ".pvtu" : save_file;
    }
}

auto get_output_times(const SEM::Helpers::InputParser_t& input_parser) -> std::vector<hostFloat> {
    if (input_parser.cmdOptionExists("--times")) {
        const std::string default_times{"1"};
        const std::string times = input_parser.getCmdOptionOr("--times", default_times);
        std::vector<hostFloat> output_times;

        std::stringstream ss(times);
        while(ss.good()) {
            std::string time;
            std::getline(ss, time, ',');

            std::stringstream ss2(time); // Could use stod, but doesn't work for all hostFloat possible types
            hostFloat t = 0;
            ss2 >> t;

            output_times.push_back(t);
        }

        return output_times;
    }

    const hostFloat t_max = input_parser.getCmdOptionOr("--t", static_cast<hostFloat>(1));

    int n_t = 11;
    hostFloat t_interval = t_max/(n_t - 1);

    if (input_parser.cmdOptionExists("--t_interval")) {
        t_interval = input_parser.getCmdOptionOr("--t_interval", t_interval);
        n_t = std::lround(t_max/t_interval + 0.5);
    }
    else {
        if (input_parser.cmdOptionExists("--n_t")) {
            n_t = input_parser.getCmdOptionOr("--n_t", n_t);
            t_interval = t_max/(n_t - 1);
        }
    }

    std::vector<hostFloat> output_times(n_t);
    for (int i = 0; i < n_t - 1; ++i) {
        output_times[i] = i * t_interval;
    }

    output_times.back() = t_max;
    return output_times;
}

auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);
    if (input_parser.cmdOptionExists("--help") || input_parser.cmdOptionExists("-h")) {
        std::cout << "Spectral element method 2D unstructured solver" << std::endl;
        std::cout << '\t' <<  "Solves the 2D wave equation on 2D unstructured meshes. The meshes use the CGNS HDF5 format, and output uses the VTK format." << std::endl << std::endl;
        std::cout << "Available options:" << std::endl;
        std::cout << '\t' <<  "--mesh"                    <<  '\t' <<  "Full path of the input mesh file. Overrides mesh_filename and mesh_directory if set." << std::endl;
        std::cout << '\t' <<  "--mesh_filename"           <<  '\t' <<  "File name of the input mesh file. Defaults to [mesh.cgns]" << std::endl;
        std::cout << '\t' <<  "--mesh_directory"          <<  '\t' <<  "Directory of the input mesh file. Defaults to [./meshes/]" << std::endl;
        std::cout << '\t' <<  "--output"                  <<  '\t' <<  "Full path of the output data file. Overrides output_filename and output_directory if set." << std::endl;
        std::cout << '\t' <<  "--output_filename"         <<  '\t' <<  "File name of the output data file. Defaults to [output.pvtu]" << std::endl;
        std::cout << '\t' <<  "--output_directory"        <<  '\t' <<  "Directory of the output data file. Defaults to [./data/]" << std::endl;
        std::cout << '\t' <<  "--n"                       <<  '\t' <<  "Initial polynomial order in elements. Defaults to [8]" << std::endl;
        std::cout << '\t' <<  "--n_max"                   <<  '\t' <<  "Maximum polynomial order in elements. Defaults to [16]" << std::endl;
        std::cout << '\t' <<  "--max_splits"              <<  '\t' <<  "Maximum number of times an elements can split. Defaults to [3]" << std::endl;
        std::cout << '\t' <<  "--n_points"                <<  '\t' <<  "Number of interpolation points in elements. Defaults to [n_max²]. Minimum 2." << std::endl;
        std::cout << '\t' <<  "--adaptivity_interval"     <<  '\t' <<  "Number of iterations between adapting the mesh. Defaults to [100]" << std::endl;
        std::cout << '\t' <<  "--load_balancing_interval" <<  '\t' <<  "Number of iterations between load balancing the mesh. Defaults to [100]" << std::endl;
        std::cout << '\t' <<  "--cfl"                     <<  '\t' <<  "CFL used for the simulation. Defaults to [0.5]" << std::endl;
        std::cout << '\t' <<  "--viscosity"               <<  '\t' <<  "Viscosity used for the simulation. Defaults to [0.1/π]" << std::endl;
        std::cout << '\t' <<  "--times"                   <<  '\t' <<  "Comma separated list of times to output at. The last time determines the simulation length. Overrides t, n_t, and t_interval." << std::endl;
        std::cout << '\t' <<  "--t"                       <<  '\t' <<  "End time of the simulation. Defaults to [1]" << std::endl;
        std::cout << '\t' <<  "--n_t"                     <<  '\t' <<  "Number of times to output. Defaults to [11]" << std::endl;
        std::cout << '\t' <<  "--t_interval"              <<  '\t' <<  "Time interval between output. Overrides n_t if set." << std::endl;
        std::cout << '\t' <<  "--tolerance_min"           <<  '\t' <<  "Estimated error above which elements will refine. Defaults to [1e-6]" << std::endl;
        std::cout << '\t' <<  "--tolerance_max"           <<  '\t' <<  "Estimated error below which elements will coarsen. Defaults to [1e-14]" << std::endl;
        exit(0);
    }

    MPI_Init(&argc, &argv);

    /*volatile bool dodo = true;
    while (dodo) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }*/

    // Argument parsing
    const fs::path mesh_file = get_input_file(input_parser);
    const fs::path output_file = get_output_file(input_parser);
    const int N_max = input_parser.getCmdOptionOr("--n_max", 16);
    const int N_initial = input_parser.getCmdOptionOr("--n", 8);
    const int max_splits = input_parser.getCmdOptionOr("--max_splits", 3);
    const size_t n_interpolation_points = input_parser.getCmdOptionOr("--n_points", static_cast<size_t>(std::pow(N_max, 2)));
    const int adaptivity_interval = input_parser.getCmdOptionOr("--adaptivity_interval", std::size_t{100});
    const int load_balancing_interval = input_parser.getCmdOptionOr("--load_balancing_interval", std::size_t{100});
    const hostFloat CFL = input_parser.getCmdOptionOr("--cfl", static_cast<hostFloat>(0.5));
    const hostFloat viscosity = input_parser.getCmdOptionOr("--viscosity", static_cast<hostFloat>(0.1/pi));
    const std::vector<hostFloat> output_times = get_output_times(input_parser);
    const hostFloat tolerance_min = input_parser.getCmdOptionOr("--tolerance_min", static_cast<hostFloat>(1e-6));
    const hostFloat tolerance_max = input_parser.getCmdOptionOr("--tolerance_max", static_cast<hostFloat>(1e-14));

    // Error checking
    if (N_initial > N_max) {
        std::cerr << "Error: Initial N (" << N_initial << ") is greater than maximum N (" << N_max << "). Exiting." << std::endl;
        exit(49);
    }
    if (n_interpolation_points < 2) {
        std::cerr << "Error: Number of interpolation points (" << n_interpolation_points << ") is smaller than 2. Exiting." << std::endl;
        exit(59);
    }

    // MPI ranks
    int global_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    if (global_rank == 0) {
        std::cout << "CFL is: " << CFL << std::endl;
    }

    // Initialisation
    auto t_start_init = std::chrono::high_resolution_clock::now();

    SEM::Host::Entities::NDG_t<SEM::Host::Polynomials::LegendrePolynomial_t> NDG(N_max, n_interpolation_points);
    SEM::Host::Meshes::Mesh2D_t mesh(mesh_file, N_initial, N_max, n_interpolation_points, max_splits, adaptivity_interval, load_balancing_interval, tolerance_min, tolerance_max, NDG.nodes_);
    SEM::Host::Solvers::Solver2D_t solver(CFL, output_times, viscosity);
    SEM::Helpers::DataWriter_t data_writer(output_file);
    mesh.initial_conditions(NDG.nodes_);

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " CPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Computation
    auto t_start = std::chrono::high_resolution_clock::now();

    solver.solve(NDG, mesh, data_writer);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " CPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    MPI_Finalize();
    return 0;
}