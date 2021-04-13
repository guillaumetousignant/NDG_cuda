#include "helpers/InputParser_t.h"
#include "cgnslib.h"
#include <string>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);

    const std::string input_filename = input_parser.getCmdOption("--filename");
    const std::string filename = (input_filename.empty()) ? "mesh.cgns" : input_filename;

    const std::string input_save_dir = input_parser.getCmdOption("--directory");
    const fs::path save_dir = (input_save_dir.empty()) ? fs::current_path() / "meshes" : input_save_dir;

    fs::create_directory(save_dir);
    const fs::path save_file = save_dir / filename;

    const std::string input_x_res = input_parser.getCmdOption("--xres");
    const int x_res = (input_x_res.empty()) ? 4 : std::stoi(input_x_res);

    const std::string input_y_res = input_parser.getCmdOption("--yres");
    const int y_res = (input_y_res.empty()) ? 4 : std::stoi(input_y_res);

    /* create gridpoints for simple example: */
    std::vector<double> x(x_res * y_res);
    std::vector<double> y(x_res * y_res);

    for (int i = 0; i < x_res; ++i)
    {
        for (int j = 0; j < y_res; ++j)
        {
            x[i * y_res + j] = i;
            y[i * y_res + j] = j;
        }
    }
    std::cout << "Created simple 2D grid points" << std::endl;

    /* WRITE X, Y, Z GRID POINTS TO CGNS FILE */
    /* open CGNS file for write */
    int index_file;
    if (cg_open(save_file.string().c_str(), CG_MODE_WRITE, &index_file)) cg_error_exit();

    /* create base (user can give any name) */
    std::string base_name("Base");
    const int icelldim = 2;
    const int iphysdim = 2;
    int index_base;
    cg_base_write(index_file, base_name.c_str(), icelldim, iphysdim, &index_base);

    /* define zone name (user can give any name) */
    std::string zone_name("Zone  1");

    /* vertex size */
    cgsize_t isize[3][2];
    isize[0][0] = 21;
    isize[0][1] = 17;

    /* cell size */
    isize[1][0] = isize[0][0]-1;
    isize[1][1] = isize[0][1]-1;

    /* boundary vertex size (always zero for structured grids) */
    isize[2][0] = 0;
    isize[2][1] = 0;

    /* create zone */
    int index_zone;
    cg_zone_write(index_file, index_base, zone_name.c_str(), *isize,Structured, &index_zone);

    /* write grid coordinates (user must use SIDS-standard names here) */
    int index_coord;
    cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateX", x.data(), &index_coord);
    cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateY", y.data(), &index_coord);

    /* close CGNS file */
    cg_close(index_file);
    std::cout << "Successfully wrote grid to file grid_c.cgns" << std::endl;
    
    return 0;
}