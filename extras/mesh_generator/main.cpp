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
auto is_power_of_two(int x) -> bool {
    /* First x in the below expression is for the case when x is 0 */
    return x && (!(x&(x-1)));
}

auto get_save_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string input_save_path = input_parser.getCmdOption("--path");
    if (!input_save_path.empty()) {
        const fs::path save_file = input_save_path;
        fs::create_directory(save_file.parent_path());
        return save_file;
    }
    else {
        const std::string input_filename = input_parser.getCmdOption("--filename");
        const std::string save_filename = (input_filename.empty()) ? "mesh.cgns" : input_filename;

        const std::string input_save_dir = input_parser.getCmdOption("--directory");
        const fs::path save_dir = (input_save_dir.empty()) ? fs::current_path() / "meshes" : input_save_dir;

        fs::create_directory(save_dir);
        return save_dir / save_filename;
    }
}
 
auto main(int argc, char* argv[]) -> int {
    
    const SEM::Helpers::InputParser_t input_parser(argc, argv);

    const fs::path save_file = get_save_file(input_parser);

    const std::string input_res = input_parser.getCmdOption("--resolution");
    const int res = (input_res.empty()) ? 4 : std::stoi(input_res);

    if (!is_power_of_two(res)) {
        std::cerr << "Error, grid resolution should be a power of two for the Hillbert curve to work. Input resolution: " << res << "'. Exiting." << std::endl;
        exit(3);
    }

    const int x_res = res;
    const int y_res = res;

    const int x_node_res = x_res + 1;
    const int y_node_res = y_res + 1;

    const int n_elements = x_res * y_res;
    const int n_nodes = x_node_res * y_node_res;

    /* create gridpoints for simple example: */
    std::vector<double> x(n_nodes);
    std::vector<double> y(n_nodes);

    for (int i = 0; i < x_node_res; ++i)
    {
        for (int j = 0; j < y_node_res; ++j)
        {
            x[i * y_node_res + j] = i;
            y[i * y_node_res + j] = j;
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
    cg_zone_write(index_file, index_base, zone_name.c_str(), isize, ZoneType_t::Unstructured, &index_zone);

    /* write grid coordinates (user must use SIDS-standard names here) */
    int index_coord;
    cg_coord_write(index_file, index_base, index_zone, DataType_t::RealDouble, "CoordinateX", x.data(), &index_coord);
    cg_coord_write(index_file, index_base, index_zone, DataType_t::RealDouble, "CoordinateY", y.data(), &index_coord);

    /* set element connectivity */
    constexpr int n_sides = 4;
    std::vector<int> elements(n_sides * n_elements);

    for (int i = 0; i < n_elements; ++i) {
        const std::array<int, 2> xy = SEM::Hilbert::d2xy(res, i);
        elements[i * n_sides]     = y_node_res * (xy[0] + 1) + xy[1] + 1;
        elements[i * n_sides + 1] = y_node_res * (xy[0] + 1) + xy[1] + 2;
        elements[i * n_sides + 2] = y_node_res * xy[0] + xy[1] + 2;
        elements[i * n_sides + 3] = y_node_res * xy[0] + xy[1] + 1;
    }

    /* write HEX_8 element connectivity (user can give any name) */
    const std::string elements_name("Elements");
    int index_section;
    const int nelem_start = 1;
    const int nelem_end = n_elements;
    const int n_boundary_elem = 0; // No boundaries yet
    cg_section_write(index_file, index_base, index_zone, elements_name.c_str(), ElementType_t::QUAD_4, nelem_start, nelem_end, n_boundary_elem, elements.data(), &index_section);

    /* create boundary (BAR) elements */
    constexpr int boundary_n_sides = 2;
    int bottom_index_section;
    const std::string bottom_boundary_name("BottomElements");
    const int bottom_start_index = n_elements + 1;
    const int bottom_end_index   = n_elements + x_res;
    std::vector<int> bottom_elements(boundary_n_sides * x_res);

    for (int i = 0; i < x_res; ++i) {
        bottom_elements[i * boundary_n_sides]     = i * y_node_res + 1;
        bottom_elements[i * boundary_n_sides + 1] = (i + 1) * y_node_res + 1;
    }

    cg_section_write(index_file, index_base, index_zone, bottom_boundary_name.c_str(), ElementType_t::BAR_2, bottom_start_index, bottom_end_index, n_boundary_elem, bottom_elements.data(), &bottom_index_section);

    int right_index_section;
    const std::string right_boundary_name("RightElements");
    const int right_start_index = bottom_end_index + 1;
    const int right_end_index   = bottom_end_index + y_res;
    std::vector<int> right_elements(boundary_n_sides * y_res);

    for (int j = 0; j < y_res; ++j) {
        right_elements[j * boundary_n_sides]     = y_node_res * (x_node_res - 1) + j + 1;
        right_elements[j * boundary_n_sides + 1] = y_node_res * (x_node_res - 1) + j + 2;
    }

    cg_section_write(index_file, index_base, index_zone, right_boundary_name.c_str(), ElementType_t::BAR_2, right_start_index, right_end_index, n_boundary_elem, right_elements.data(), &right_index_section);

    int top_index_section;
    const std::string top_boundary_name("TopElements");
    const int top_start_index = right_end_index + 1;
    const int top_end_index   = right_end_index + x_res;
    std::vector<int> top_elements(boundary_n_sides * x_res);

    for (int i = 0; i < x_res; ++i) {
        top_elements[i * boundary_n_sides]     = (x_res - i + 1) * y_node_res;
        top_elements[i * boundary_n_sides + 1] = (x_res - i) * y_node_res;
    }

    cg_section_write(index_file, index_base, index_zone, top_boundary_name.c_str(), ElementType_t::BAR_2, top_start_index, top_end_index, n_boundary_elem, top_elements.data(), &top_index_section);

    int left_index_section;
    const std::string left_boundary_name("LeftElements");
    const int left_start_index = top_end_index + 1;
    const int left_end_index   = top_end_index + y_res;
    std::vector<int> left_elements(boundary_n_sides * y_res);

    for (int j = 0; j < y_res; ++j) {
        left_elements[j * boundary_n_sides]     = y_res - j + 1;
        left_elements[j * boundary_n_sides + 1] = y_res - j;
    }

    cg_section_write(index_file, index_base, index_zone, left_boundary_name.c_str(), ElementType_t::BAR_2, left_start_index, left_end_index, n_boundary_elem, left_elements.data(), &left_index_section);

    /* write boundary conditions */
    int bottom_index_boundary;
    std::vector<int> bottom_boundary(x_res);

    for (int i = 0; i < x_res; ++i) {
        bottom_boundary[i] = bottom_start_index + i;
    }

    cg_boco_write(index_file, index_base, index_zone, "BottomBoundary", BCType_t::BCWall, PointSetType_t::PointList, x_res, bottom_boundary.data(), &bottom_index_boundary);

    int right_index_boundary;
    std::vector<int> right_boundary(y_res);

    for (int i = 0; i < y_res; ++i) {
        right_boundary[i] = right_start_index + i;
    }

    cg_boco_write(index_file, index_base, index_zone, "RightBoundary", BCType_t::BCWall, PointSetType_t::PointList, y_res, right_boundary.data(), &right_index_boundary);

    int top_index_boundary;
    std::vector<int> top_boundary(x_res);

    for (int i = 0; i < x_res; ++i) {
        top_boundary[i] = top_start_index + i;
    }

    cg_boco_write(index_file, index_base, index_zone, "TopBoundary", BCType_t::BCWall, PointSetType_t::PointList, x_res, top_boundary.data(), &top_index_boundary);

    int left_index_boundary;
    std::vector<int> left_boundary(y_res);

    for (int i = 0; i < y_res; ++i) {
        left_boundary[i] = left_start_index + i;
    }

    cg_boco_write(index_file, index_base, index_zone, "LeftBoundary", BCType_t::BCWall, PointSetType_t::PointList, y_res, left_boundary.data(), &left_index_boundary);

    /* the above are all face-center locations for the BCs - must indicate this, otherwise Vertices will be assumed! */
    cg_boco_gridlocation_write(index_file, index_base, index_zone, bottom_index_boundary, GridLocation_t::EdgeCenter);

    cg_boco_gridlocation_write(index_file, index_base, index_zone, right_index_boundary, GridLocation_t::EdgeCenter);

    cg_boco_gridlocation_write(index_file, index_base, index_zone, top_index_boundary, GridLocation_t::EdgeCenter);

    cg_boco_gridlocation_write(index_file, index_base, index_zone, left_index_boundary, GridLocation_t::EdgeCenter);

    /* close CGNS file */
    cg_close(index_file);
    std::cout << "Successfully wrote grid to file " <<  save_file << std::endl;
    
    return 0;
}