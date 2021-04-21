#include "helpers/InputParser_t.h"
#include "cgnslib.h"
#include <string>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

auto main(int argc, char* argv[]) -> int {
    fs::path in_file;
    fs::path out_file;

    const SEM::Helpers::InputParser_t input_parser(argc, argv);

    const std::string input_save_path = input_parser.getCmdOption("--in_path");
    if (!input_save_path.empty()) {
        in_file = input_save_path;
        fs::create_directory(in_file.parent_path());
    }
    else {
        const std::string input_filename = input_parser.getCmdOption("--in_filename");
        const std::string save_filename = (input_filename.empty()) ? "mesh.cgns" : input_filename;

        const std::string input_save_dir = input_parser.getCmdOption("--in_directory");
        const fs::path save_dir = (input_save_dir.empty()) ? fs::current_path() / "meshes" : input_save_dir;

        fs::create_directory(save_dir);
        in_file = save_dir / save_filename;
    }

    const std::string output_save_path = input_parser.getCmdOption("--out_path");
    if (!output_save_path.empty()) {
        out_file = output_save_path;
        fs::create_directory(out_file.parent_path());
    }
    else {
        const std::string output_filename = input_parser.getCmdOption("--out_filename");
        const std::string save_filename = (output_filename.empty()) ? "mesh_partitioned.cgns" : output_filename;

        const std::string output_save_dir = input_parser.getCmdOption("--out_directory");
        const fs::path save_dir = (output_save_dir.empty()) ? fs::current_path() / "meshes" : output_save_dir;

        fs::create_directory(save_dir);
        out_file = save_dir / save_filename;
    }
    
    return 0;
}