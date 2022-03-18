#include "helpers/float_types.h"
#include "helpers/InputParser_t.h"
#include "helpers/DataWriter_t.h"
#include "entities/NDG_t.h"
#include "meshes/Mesh2D_t.h"
#include "solvers/Solver2D_t.h"
#include "polynomials/LegendrePolynomial_t.h"
#include "functions/Utilities.h"
#define NOMINMAX
#include "helpers/termcolor.hpp"
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

constexpr hostFloat pi{3.14159265358979323846};
enum PreConditionAlgorithm : int {Iterative, Sequential, None};

auto get_input_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string input_mesh_path = input_parser.getCmdOption("--mesh");
    if (!input_mesh_path.empty()) {
        const fs::path mesh_file = input_mesh_path;
        if (mesh_file.has_parent_path()) {
            fs::create_directory(mesh_file.parent_path());
        }
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
        if (save_file.has_parent_path()) {
            fs::create_directory(save_file.parent_path());
        }
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

auto get_pre_condition_algorithm(const SEM::Helpers::InputParser_t& input_parser) -> PreConditionAlgorithm {
    std::string algorithm_string = input_parser.getCmdOption("--pre_condition_algorithm");
    if (!algorithm_string.empty()) {
        SEM::to_lower(algorithm_string);
        if (algorithm_string == "iterative") {
            return PreConditionAlgorithm::Iterative;
        }
        else if (algorithm_string == "sequential") {
            return PreConditionAlgorithm::Sequential;
        }
        else if (algorithm_string == "none") {
            return PreConditionAlgorithm::None;
        }
        else {
            std::cerr << "Error: pre-condition algorithm '" << algorithm_string << "' unknown, valid options are: 'iterative', 'sequential', 'none'. Exiting." << std::endl;
            exit(79);
        }
    }
    else {
        return PreConditionAlgorithm::Sequential;
    }
}

auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);
    if (input_parser.cmdOptionExists("--help") || input_parser.cmdOptionExists("-h")) {
        std::cout << "usage: NDG_host [-h] [--mesh MESH] [--mesh_filename MESH_FILENAME] [--mesh_directory MESH_DIRECTORY] [--output OUTPUT] [--output_filename OUTPUT_FILENAME] [--output_directory OUTPUT_DIRECTORY] [--n N] [--n_max N_MAX] [--max_splits MAX_SPLITS] [--n_points N_POINTS] [--adaptivity_interval ADAPTIVITY_INTERVAL] [--load_balancing_interval LOAD_BALANCING_INTERVAL] [--cfl CFL] [--viscosity VISCOSITY] [--times T1,T2,T3,T4...] [--t T] [--n_t N_T] [--t_interval T_INTERVAL] [--memory MEMORY] [--tolerance_min TOLERANCE_MIN] [--tolerance_max TOLERANCE_MAX] [--load_balancing_threshold LOAD_BALANCING_THRESHOLD] [--pre_condition PRE_CONDITION] [-v]" << std::endl << std::endl;
        std::cout << "Discontinuous Galerkin spectral element method CPU solver for the 2D wave equation on 2D unstructured meshes. The meshes use the CGNS HDF5 format, and output uses the VTK format." << std::endl << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << '\t' << "-h, --help"                   << '\t' << "show this help message and exit" << std::endl;
        std::cout << '\t' << "--mesh"                       << '\t' << "full path of the input mesh file, overrides mesh_filename and mesh_directory if set" << std::endl;
        std::cout << '\t' << "--mesh_filename"              << '\t' << "file name of the input mesh file (default: mesh.cgns)" << std::endl;
        std::cout << '\t' << "--mesh_directory"             << '\t' << "directory of the input mesh file (default: ./meshes/)" << std::endl;
        std::cout << '\t' << "--output"                     << '\t' << "full path of the output data file, overrides output_filename and output_directory if set" << std::endl;
        std::cout << '\t' << "--output_filename"            << '\t' << "file name of the output data file (default: output.pvtu)" << std::endl;
        std::cout << '\t' << "--output_directory"           << '\t' << "directory of the output data file (default: ./data/)" << std::endl;
        std::cout << '\t' << "--n"                          << '\t' << "initial polynomial order in elements (default: 8)" << std::endl;
        std::cout << '\t' << "--n_max"                      << '\t' << "maximum polynomial order in elements (default: 16)" << std::endl;
        std::cout << '\t' << "--max_splits"                 << '\t' << "maximum number of times an elements can split (default: 3)" << std::endl;
        std::cout << '\t' << "--n_points"                   << '\t' << "number of interpolation points in elements, minimum 2 (default: n_max²)" << std::endl;
        std::cout << '\t' << "--adaptivity_interval"        << '\t' << "number of iterations between adapting the mesh, 0 to disable (default: 100)" << std::endl;
        std::cout << '\t' << "--load_balancing_interval"    << '\t' << "number of iterations between load balancing the mesh, 0 to disable (default: 100)" << std::endl;
        std::cout << '\t' << "--cfl"                        << '\t' << "CFL used for the simulation (default: 0.5)" << std::endl;
        std::cout << '\t' << "--viscosity"                  << '\t' << "viscosity used for the simulation (default: 0.1/π)" << std::endl;
        std::cout << '\t' << "--times"                      << '\t' << "comma separated list of times to output at, simulating up to the last one, and overrides t, n_t, and t_interval" << std::endl;
        std::cout << '\t' << "--t"                          << '\t' << "end time of the simulation (default: 1)" << std::endl;
        std::cout << '\t' << "--n_t"                        << '\t' << "number of times to output (default: 11)" << std::endl;
        std::cout << '\t' << "--t_interval"                 << '\t' << "time interval between output, overrides n_t if set" << std::endl;
        std::cout << '\t' << "--tolerance_min"              << '\t' << "estimated error above which elements will refine (default: 1e-6)" << std::endl;
        std::cout << '\t' << "--tolerance_max"              << '\t' << "estimated error below which elements will coarsen (default: 1e-14)" << std::endl;
        std::cout << '\t' << "--load_balancing_threshold"   << '\t' << "load imbalance ratio, n_elements_max/n_elements_min below which no load balancing occurs (default: 1.01)" << std::endl;
        std::cout << '\t' << "--pre_condition"              << '\t' << "number of adaptivity steps to run before starting the computation, 0 to disable (default: 0)" << std::endl;
        std::cout << '\t' << "--pre_condition_interval"     << '\t' << "number of iterations between adapting the mesh for pre-conditioning, 0 to disable (default: adaptivity_interval)" << std::endl;
        std::cout << '\t' << "--pre_condition_algorithm {iterative,sequential,none}" << '\t' << "algorithm to use for pre-conditioning (default: sequential)" << std::endl;
        std::cout << '\t' << "-v, --version"                << '\t' << "show program's version number and exit" << std::endl;
        exit(0);
    }
    else if (input_parser.cmdOptionExists("--version") || input_parser.cmdOptionExists("-v")) {
        std::cout << "NDG.exe 1.0.0" << std::endl;
        exit(0);
    }

    MPI_Init(&argc, &argv);


    // Argument parsing
    const fs::path mesh_file = get_input_file(input_parser);
    const fs::path output_file = get_output_file(input_parser);
    const int N_max = input_parser.getCmdOptionOr("--n_max", 16);
    const int N_initial = input_parser.getCmdOptionOr("--n", 8);
    const int max_splits = input_parser.getCmdOptionOr("--max_splits", 3);
    const size_t n_interpolation_points = input_parser.getCmdOptionOr("--n_points", static_cast<size_t>(std::pow(N_max, 2)));
    const size_t adaptivity_interval = input_parser.getCmdOptionOr("--adaptivity_interval", std::size_t{100});
    const size_t load_balancing_interval = input_parser.getCmdOptionOr("--load_balancing_interval", std::size_t{100});
    const hostFloat CFL = input_parser.getCmdOptionOr("--cfl", hostFloat{0.5});
    const hostFloat viscosity = input_parser.getCmdOptionOr("--viscosity", hostFloat{0.1}/pi);
    const std::vector<hostFloat> output_times = get_output_times(input_parser);
    const hostFloat tolerance_min = input_parser.getCmdOptionOr("--tolerance_min", hostFloat{1e-6});
    const hostFloat tolerance_max = input_parser.getCmdOptionOr("--tolerance_max", hostFloat{1e-14});
    const deviceFloat load_balancing_threshold = input_parser.getCmdOptionOr("--load_balancing_threshold", hostFloat{1.01});
    const size_t pre_condition_steps = input_parser.getCmdOptionOr("--pre_condition", std::size_t{0});
    const size_t pre_condition_interval = input_parser.getCmdOptionOr("--pre_condition_interval", adaptivity_interval);
    const PreConditionAlgorithm pre_condition_algorithm = get_pre_condition_algorithm(input_parser);

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
    if (global_rank == 0) {
        std::cout << termcolor::bold << termcolor::blue;
        std::cout << "Initialising" << std::endl;
        std::cout << termcolor::reset;
    }
    auto t_start_init = std::chrono::high_resolution_clock::now();

    SEM::Host::Entities::NDG_t<SEM::Host::Polynomials::LegendrePolynomial_t> NDG(N_max, n_interpolation_points);
    SEM::Host::Meshes::Mesh2D_t mesh(mesh_file, N_initial, N_max, n_interpolation_points, max_splits, adaptivity_interval, load_balancing_interval, tolerance_min, tolerance_max, load_balancing_threshold, NDG.nodes_);
    SEM::Host::Solvers::Solver2D_t solver(CFL, output_times, viscosity);
    SEM::Helpers::DataWriter_t data_writer(output_file);
    mesh.initial_conditions(NDG.nodes_);

    auto t_end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " CPU initialisation time: " 
            << std::chrono::duration<double, std::milli>(t_end_init - t_start_init).count()/1000.0 
            << "s." << std::endl;

    // Pre-condition
    if (pre_condition_steps > 0 && pre_condition_interval > 0 && pre_condition_algorithm != PreConditionAlgorithm::None) {
        if (global_rank == 0) {
            std::cout << termcolor::bold << termcolor::green;
            std::cout << "Pre-conditioning" << std::endl;
            std::cout << termcolor::reset;
        }
        auto t_start_pre = std::chrono::high_resolution_clock::now();

        if (pre_condition_algorithm == PreConditionAlgorithm::Iterative) {
            solver.pre_condition_iterative(NDG, mesh, pre_condition_steps, pre_condition_interval);
        }
        else if (pre_condition_algorithm == PreConditionAlgorithm::Sequential) {
            solver.pre_condition(NDG, mesh, pre_condition_steps, pre_condition_interval);
        }

        auto t_end_pre = std::chrono::high_resolution_clock::now();
        std::cout << "Process " << global_rank << " CPU pre-condition time: " 
                  << std::chrono::duration<double, std::milli>(t_end_pre - t_start_pre).count()/1000.0 
                  << "s." << std::endl;
    }

    // Computation
    if (global_rank == 0) {
        std::cout << termcolor::bold << termcolor::yellow;
        std::cout << "Solving" << std::endl;
        std::cout << termcolor::reset;
    }
    auto t_start = std::chrono::high_resolution_clock::now();

    solver.solve(NDG, mesh, data_writer);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Process " << global_rank << " CPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end - t_start).count()/1000.0 
            << "s." << std::endl;

    MPI_Finalize();
    return 0;
}