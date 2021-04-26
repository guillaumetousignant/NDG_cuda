#include "helpers/InputParser_t.h"
#include "functions/Hilbert.h"
#include "functions/Utilities.h"
#include "cgnslib.h"
#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
#include <array>
#include <unordered_map>

namespace fs = std::filesystem;

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

auto get_boundary_conditions(const SEM::Helpers::InputParser_t& input_parser) -> std::array<BCType_t, 4> {
    BCType_t bottom_boundary_type = BCType_t::BCTypeNull;
    BCType_t right_boundary_type  = BCType_t::BCTypeNull;
    BCType_t top_boundary_type    = BCType_t::BCTypeNull;
    BCType_t left_boundary_type   = BCType_t::BCTypeNull;

    const std::unordered_map<std::string, BCType_t> values {
        {"wall",     BCType_t::BCWall},
        {"symmetry", BCType_t::BCSymmetryPlane}, 
        {"null",     BCType_t::BCTypeNull}
    };

    std::string input_boundaries = input_parser.getCmdOption("--boundaries");
    if (!input_boundaries.empty()) {
        SEM::to_lower(input_boundaries);

        const auto it = values.find(input_boundaries);
        if (it == values.end()) {
            std::cerr << "Error, unknown boundary type '" << input_boundaries << "'. Implemented boundary types are: 'wall', 'symmetry'. Exiting." << std::endl;
            exit(4);
        }
        else {
            bottom_boundary_type = it->second;
            right_boundary_type = it->second;
            top_boundary_type = it->second;
            left_boundary_type = it->second;
        }
    }
    else {
        bottom_boundary_type = BCType_t::BCWall;
        right_boundary_type  = BCType_t::BCWall;
        top_boundary_type    = BCType_t::BCWall;
        left_boundary_type   = BCType_t::BCWall;
    }
    
    std::string input_bottom_boundary = input_parser.getCmdOption("--bottom_boundary");
    if (!input_bottom_boundary.empty()) {
        SEM::to_lower(input_bottom_boundary);

        const auto it = values.find(input_bottom_boundary);
        if (it == values.end()) {
            std::cerr << "Error, unknown bottom boundary type '" << input_bottom_boundary << "'. Implemented boundary types are: 'wall', 'symmetry'. Exiting." << std::endl;
            exit(5);
        }
        else {
            bottom_boundary_type = it->second;
        }
    }

    std::string input_right_boundary = input_parser.getCmdOption("--right_boundary");
    if (!input_right_boundary.empty()) {
        SEM::to_lower(input_right_boundary);

        const auto it = values.find(input_right_boundary);
        if (it == values.end()) {
            std::cerr << "Error, unknown right boundary type '" << input_right_boundary << "'. Implemented boundary types are: 'wall', 'symmetry'. Exiting." << std::endl;
            exit(6);
        }
        else {
            right_boundary_type = it->second;
        }
    }

    std::string input_top_boundary = input_parser.getCmdOption("--top_boundary");
    if (!input_top_boundary.empty()) {
        SEM::to_lower(input_top_boundary);

        const auto it = values.find(input_top_boundary);
        if (it == values.end()) {
            std::cerr << "Error, unknown top boundary type '" << input_top_boundary << "'. Implemented boundary types are: 'wall', 'symmetry'. Exiting." << std::endl;
            exit(7);
        }
        else {
            top_boundary_type = it->second;
        }
    }

    std::string input_left_boundary = input_parser.getCmdOption("--left_boundary");
    if (!input_left_boundary.empty()) {
        SEM::to_lower(input_left_boundary);

        const auto it = values.find(input_left_boundary);
        if (it == values.end()) {
            std::cerr << "Error, unknown left boundary type '" << input_left_boundary << "'. Implemented boundary types are: 'wall', 'symmetry'. Exiting." << std::endl;
            exit(8);
        }
        else {
            left_boundary_type = it->second;
        }
    }

    return {bottom_boundary_type, right_boundary_type, top_boundary_type, left_boundary_type};
}
 
auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);
    if (input_parser.cmdOptionExists("--help") || input_parser.cmdOptionExists("-h")) {
        std::cout << "Square unstructured mesh generator" << std::endl;
        std::cout << '\t' << "Available options:" << std::endl;
        std::cout << '\t' << '\t' <<  "--path"            <<  '\t' <<  "Full path to the output mesh file. Overrides filename and directory if set." << std::endl;
        std::cout << '\t' << '\t' <<  "--filename"        <<  '\t' <<  "File name to the output mesh file. Defaults to [mesh.cgns]" << std::endl;
        std::cout << '\t' << '\t' <<  "--directory"       <<  '\t' <<  "Directory in which to save the output mesh file. Defaults to [./meshes/]" << std::endl;
        std::cout << '\t' << '\t' <<  "--resolution"      <<  '\t' <<  "Number of elements in the x and y directions. Must be a power of two. Defaults to [4]" << std::endl;
        std::cout << '\t' << '\t' <<  "--x_periodic"      <<  '\t' <<  "Sets the mesh to be periodic in the x direction. Overrides left and right boundaries." << std::endl;
        std::cout << '\t' << '\t' <<  "--y_periodic"      <<  '\t' <<  "Sets the mesh to be periodic in the y direction. Overrides top and bottom boundaries." << std::endl;
        std::cout << '\t' << '\t' <<  "--boundaries"      <<  '\t' <<  "Sets all four boundary conditions. Acceptable values are \"wall\", \"symmetry\" and \"null\". Defaults to [wall]" << std::endl;
        std::cout << '\t' << '\t' <<  "--bottom_boundary" <<  '\t' <<  "Sets the bottom boundary condition. Acceptable values are \"wall\", \"symmetry\" and \"null\". Overrides boundaries." << std::endl;
        std::cout << '\t' << '\t' <<  "--right_boundary"  <<  '\t' <<  "Sets the right boundary condition. Acceptable values are \"wall\", \"symmetry\" and \"null\". Overrides boundaries." << std::endl;
        std::cout << '\t' << '\t' <<  "--top_boundary"    <<  '\t' <<  "Sets the top boundary condition. Acceptable values are \"wall\", \"symmetry\" and \"null\". Overrides boundaries." << std::endl;
        std::cout << '\t' << '\t' <<  "--left_boundary"   <<  '\t' <<  "Sets the left boundary condition. Acceptable values are \"wall\", \"symmetry\" and \"null\". Overrides boundaries." << std::endl;
        exit(0);
    }

    const fs::path save_file = get_save_file(input_parser);

    // Mesh resolution input
    const std::string input_res = input_parser.getCmdOption("--resolution");
    const int res = (input_res.empty()) ? 4 : std::stoi(input_res);

    if (!SEM::is_power_of_two(res)) {
        std::cerr << "Error, grid resolution should be a power of two for the Hillbert curve to work. Input resolution: " << res << "'. Exiting." << std::endl;
        exit(3);
    }

    const int x_res = res;
    const int y_res = res;

    const int x_node_res = x_res + 1;
    const int y_node_res = y_res + 1;

    const int n_elements = x_res * y_res;
    const int n_elements_total = n_elements + 2 * x_res + 2 * y_res;
    const int n_nodes = x_node_res * y_node_res;

    // Boundary conditions input
    const auto [bottom_boundary_type, right_boundary_type, top_boundary_type, left_boundary_type] = get_boundary_conditions(input_parser);

    // Symmetry input
    const bool x_periodic = input_parser.cmdOptionExists("--x_periodic");
    const bool y_periodic = input_parser.cmdOptionExists("--y_periodic");

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
    int index_file = 0;
    if (cg_open(save_file.string().c_str(), CG_MODE_WRITE, &index_file)) cg_error_exit();

    /* create base (user can give any name) */
    const std::string base_name("Base");
    const int icelldim = 2;
    const int iphysdim = 2;
    int index_base = 0;
    cg_base_write(index_file, base_name.c_str(), icelldim, iphysdim, &index_base);

    /* define zone name (user can give any name) */
    const std::string zone_name("Zone 1");

    /* vertex size, cell size, boundary vertex size (always zero for structured grids) */
    std::array<cgsize_t, 3> isize {n_nodes,
                                   n_elements_total,
                                   0};

    /* create zone */
    int index_zone = 0;
    cg_zone_write(index_file, index_base, zone_name.c_str(), isize.data(), ZoneType_t::Unstructured, &index_zone);

    /* write grid coordinates (user must use SIDS-standard names here) */
    int index_coord = 0;
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
    int index_section = 0;
    const int nelem_start = 1;
    const int nelem_end = n_elements;
    const int n_boundary_elem = 0; // No boundaries yet
    cg_section_write(index_file, index_base, index_zone, elements_name.c_str(), ElementType_t::QUAD_4, nelem_start, nelem_end, n_boundary_elem, elements.data(), &index_section);

    /* create boundary (BAR) elements */
    constexpr int boundary_n_sides = 2;
    const int bottom_start_index = n_elements + 1;
    const int bottom_end_index   = n_elements + x_res;
    const int right_start_index  = n_elements + x_res + 1;
    const int right_end_index    = n_elements + x_res + y_res;
    const int top_start_index    = n_elements + x_res + y_res + 1;
    const int top_end_index      = n_elements + 2 * x_res + y_res;
    const int left_start_index   = n_elements + 2 * x_res + y_res + 1;
    const int left_end_index     = n_elements + 2 * x_res + 2 * y_res;

    int bottom_index_section = 0;
    const std::string bottom_boundary_name("BottomElements");
    std::vector<int> bottom_elements(boundary_n_sides * x_res);

    for (int i = 0; i < x_res; ++i) {
        bottom_elements[i * boundary_n_sides]     = i * y_node_res + 1;
        bottom_elements[i * boundary_n_sides + 1] = (i + 1) * y_node_res + 1;
    }

    cg_section_write(index_file, index_base, index_zone, bottom_boundary_name.c_str(), ElementType_t::BAR_2, bottom_start_index, bottom_end_index, n_boundary_elem, bottom_elements.data(), &bottom_index_section);

    int right_index_section = 0;
    const std::string right_boundary_name("RightElements");
    std::vector<int> right_elements(boundary_n_sides * y_res);

    for (int j = 0; j < y_res; ++j) {
        right_elements[j * boundary_n_sides]     = y_node_res * (x_node_res - 1) + j + 1;
        right_elements[j * boundary_n_sides + 1] = y_node_res * (x_node_res - 1) + j + 2;
    }

    cg_section_write(index_file, index_base, index_zone, right_boundary_name.c_str(), ElementType_t::BAR_2, right_start_index, right_end_index, n_boundary_elem, right_elements.data(), &right_index_section);

    int top_index_section = 0;
    const std::string top_boundary_name("TopElements");
    std::vector<int> top_elements(boundary_n_sides * x_res);

    for (int i = 0; i < x_res; ++i) {
        top_elements[i * boundary_n_sides]     = (x_res - i + 1) * y_node_res;
        top_elements[i * boundary_n_sides + 1] = (x_res - i) * y_node_res;
    }

    cg_section_write(index_file, index_base, index_zone, top_boundary_name.c_str(), ElementType_t::BAR_2, top_start_index, top_end_index, n_boundary_elem, top_elements.data(), &top_index_section);

    int left_index_section = 0;
    const std::string left_boundary_name("LeftElements");
    std::vector<int> left_elements(boundary_n_sides * y_res);

    for (int j = 0; j < y_res; ++j) {
        left_elements[j * boundary_n_sides]     = y_res - j + 1;
        left_elements[j * boundary_n_sides + 1] = y_res - j;
    }

    cg_section_write(index_file, index_base, index_zone, left_boundary_name.c_str(), ElementType_t::BAR_2, left_start_index, left_end_index, n_boundary_elem, left_elements.data(), &left_index_section);

    /* write boundary conditions */
    /* the above are all face-center locations for the BCs - must indicate this, otherwise Vertices will be assumed! */
    if (!y_periodic) {
        int bottom_index_boundary = 0;
        std::vector<int> bottom_boundary(x_res);

        for (int i = 0; i < x_res; ++i) {
            bottom_boundary[i] = bottom_start_index + i;
        }

        cg_boco_write(index_file, index_base, index_zone, "BottomBoundary", bottom_boundary_type, PointSetType_t::PointList, x_res, bottom_boundary.data(), &bottom_index_boundary);
        cg_boco_gridlocation_write(index_file, index_base, index_zone, bottom_index_boundary, GridLocation_t::EdgeCenter);

        int top_index_boundary = 0;
        std::vector<int> top_boundary(x_res);

        for (int i = 0; i < x_res; ++i) {
            top_boundary[i] = top_start_index + i;
        }

        cg_boco_write(index_file, index_base, index_zone, "TopBoundary", top_boundary_type, PointSetType_t::PointList, x_res, top_boundary.data(), &top_index_boundary);
        cg_boco_gridlocation_write(index_file, index_base, index_zone, top_index_boundary, GridLocation_t::EdgeCenter);
    }

    if (!x_periodic) {
        int right_index_boundary = 0;
        std::vector<int> right_boundary(y_res);

        for (int i = 0; i < y_res; ++i) {
            right_boundary[i] = right_start_index + i;
        }

        cg_boco_write(index_file, index_base, index_zone, "RightBoundary", right_boundary_type, PointSetType_t::PointList, y_res, right_boundary.data(), &right_index_boundary);
        cg_boco_gridlocation_write(index_file, index_base, index_zone, right_index_boundary, GridLocation_t::EdgeCenter);
        
        int left_index_boundary = 0;
        std::vector<int> left_boundary(y_res);

        for (int i = 0; i < y_res; ++i) {
            left_boundary[i] = left_start_index + i;
        }

        cg_boco_write(index_file, index_base, index_zone, "LeftBoundary", left_boundary_type, PointSetType_t::PointList, y_res, left_boundary.data(), &left_index_boundary);
        cg_boco_gridlocation_write(index_file, index_base, index_zone, left_index_boundary, GridLocation_t::EdgeCenter);
    }

    /* write integer connectivity info (user can give any name) */
    if (y_periodic) {
        int y_periodic_bottom_index = 0;
        int y_periodic_top_index = 0;
        std::vector<int> elements_bottom(x_res);
        std::vector<int> elements_top(x_res);

        for (int i = 0; i < x_res; ++i) {
            elements_bottom[i] = bottom_start_index + i;
            elements_top[i] = top_end_index - i;
        }

        cg_conn_write(index_file, index_base, index_zone, "YPeriodicBottom", GridLocation_t::FaceCenter, GridConnectivityType_t::Abutting1to1, PointSetType_t::PointList, x_res, elements_bottom.data(), zone_name.c_str(), ZoneType_t::Unstructured, PointSetType_t::PointListDonor, DataType_t::Integer, x_res, elements_top.data(), &y_periodic_bottom_index);
        cg_conn_write(index_file, index_base, index_zone, "YPeriodicTop", GridLocation_t::FaceCenter, GridConnectivityType_t::Abutting1to1, PointSetType_t::PointList, x_res, elements_top.data(), zone_name.c_str(), ZoneType_t::Unstructured, PointSetType_t::PointListDonor, DataType_t::Integer, x_res, elements_bottom.data(), &y_periodic_top_index);
    }

    if (x_periodic) {
        int x_periodic_right_index = 0;
        int x_periodic_left_index = 0;
        std::vector<int> elements_right(y_res);
        std::vector<int> elements_left(y_res);

        for (int j = 0; j < y_res; ++j) {
            elements_right[j] = right_start_index + j;
            elements_left[j] = left_end_index - j;
        }

        cg_conn_write(index_file, index_base, index_zone, "XPeriodicRight", GridLocation_t::FaceCenter, GridConnectivityType_t::Abutting1to1, PointSetType_t::PointList, y_res, elements_right.data(), zone_name.c_str(), ZoneType_t::Unstructured, PointSetType_t::PointListDonor, DataType_t::Integer, y_res, elements_left.data(), &x_periodic_right_index);
        cg_conn_write(index_file, index_base, index_zone, "XPeriodicLeft", GridLocation_t::FaceCenter, GridConnectivityType_t::Abutting1to1, PointSetType_t::PointList, y_res, elements_left.data(), zone_name.c_str(), ZoneType_t::Unstructured, PointSetType_t::PointListDonor, DataType_t::Integer, y_res, elements_right.data(), &x_periodic_left_index);
    }

    /* close CGNS file */
    cg_close(index_file);
    std::cout << "Successfully wrote grid to file " <<  save_file << std::endl;
    
    return 0;
}