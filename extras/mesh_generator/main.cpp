#include "helpers/InputParser_t.h"
#include "functions/Hilbert.h"
#include "cgnslib.h"
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
#include <array>

namespace fs = std::filesystem;

/* Function to check if x is power of 2*/
auto isPowerOfTwo (int x) -> bool {
    /* First x in the below expression is for the case when x is 0 */
    return x && (!(x&(x-1)));
}
 
auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);

    const std::string input_filename = input_parser.getCmdOption("--filename");
    const std::string filename = (input_filename.empty()) ? "mesh.cgns" : input_filename;

    const std::string input_save_dir = input_parser.getCmdOption("--directory");
    const fs::path save_dir = (input_save_dir.empty()) ? fs::current_path() / "meshes" : input_save_dir;

    fs::create_directory(save_dir);
    const fs::path save_file = save_dir / filename;

    const std::string input_res = input_parser.getCmdOption("--resolution");
    const int res = (input_res.empty()) ? 4 : std::stoi(input_res);

    if (!isPowerOfTwo(res)) {
        std::cerr << "Error, grid resolution should be a power of two for the Hillbert curve to work. Input resolution: " << res << "'. Exiting." << std::endl;
        exit(3);
    }

    const int x_res = res;
    const int y_res = res;

    const int n_elements = x_res * y_res;
    const int n_nodes = (x_res + 1) * (y_res + 1);

    /* create gridpoints for simple example: */
    std::vector<double> x(n_nodes);
    std::vector<double> y(n_nodes);

    for (int i = 0; i < x_res + 1; ++i)
    {
        for (int j = 0; j < y_res + 1; ++j)
        {
            x[i * (y_res + 1) + j] = i;
            y[i * (y_res + 1) + j] = j;
        }
    }
    std::cout << "Created simple 2D grid points" << std::endl;

    /* WRITE X, Y, Z GRID POINTS TO CGNS FILE */
    /* open CGNS file for write */
    int index_file;
    if (cg_open(save_file.string().c_str(), CG_MODE_WRITE, &index_file)) cg_error_exit();

    /* create base (user can give any name) */
    const std::string base_name("Base");
    const int icelldim = 2;
    const int iphysdim = 2;
    int index_base;
    cg_base_write(index_file, base_name.c_str(), icelldim, iphysdim, &index_base);

    /* define zone name (user can give any name) */
    const std::string zone_name("Zone  1");

    /* vertex size */
    cgsize_t isize[3];
    isize[0] = n_nodes;

    /* cell size */
    isize[1] = n_elements;

    /* boundary vertex size (always zero for structured grids) */
    isize[2] = 0;

    /* create zone */
    int index_zone;
    cg_zone_write(index_file, index_base, zone_name.c_str(), isize, Unstructured, &index_zone);

    /* write grid coordinates (user must use SIDS-standard names here) */
    int index_coord;
    cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateX", x.data(), &index_coord);
    cg_coord_write(index_file, index_base, index_zone, RealDouble, "CoordinateY", y.data(), &index_coord);

    /* set element connectivity */
    constexpr int n_sides = 4;
    std::vector<int> elements(n_sides * n_elements);

    for (int i = 0; i < n_elements; ++i) {
        const std::array<int, 2> xy = SEM::Hilbert::d2xy(res, i);
        elements[i * n_sides]     = (x_res + 1) * (xy[0] + 1) + xy[1];
        elements[i * n_sides + 1] = (x_res + 1) * (xy[0] + 1) + xy[1] + 1;
        elements[i * n_sides + 2] = (x_res + 1) * xy[0] + xy[1] + 1;
        elements[i * n_sides + 3] = (x_res + 1) * xy[0] + xy[1];
    }

    /* write HEX_8 element connectivity (user can give any name) */
    const std::string elements_name("Elements");
    int index_section;
    const int nelem_start = 0;
    const int nelem_end = n_elements - 1;
    const int n_boundary_elem = 0; // No boundaries yet
    cg_section_write(index_file, index_base, index_zone, elements_name.c_str(), QUAD_4, nelem_start, nelem_end, n_boundary_elem, elements.data(), &index_section);

    /* close CGNS file */
    cg_close(index_file);
    std::cout << "Successfully wrote grid to file " <<  save_file << std::endl;
    
    return 0;
}