#include "meshes/Mesh2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "helpers/ProgressBar_t.h"
#include "functions/Utilities.h"
#include "cgnslib.h"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;
using SEM::Entities::device_vector;
using SEM::Entities::Vec2;

constexpr int CGIO_MAX_NAME_LENGTH = 33; // Includes the null terminator

SEM::Meshes::Mesh2D_t::Mesh2D_t(std::filesystem::path filename, int initial_N, cudaStream_t &stream) :       
        stream_(stream) {

    std::string extension = filename.extension().string();
    SEM::to_lower(extension);

    if (extension == ".cgns") {
        read_cgns(filename);
    }
    else if (extension == ".su2") {
        read_su2(filename);
    }
    else {
        std::cerr << "Error: extension '" << extension << "' not recognized. Exiting." << std::endl;
        exit(14);
    }
}

void SEM::Meshes::Mesh2D_t::read_su2(std::filesystem::path filename) {
    std::cerr << "Error: SU2 meshes not implemented yet. Exiting." << std::endl;
    exit(15);

    /*std::string line;
    std::string token;
    size_t value;

    std::ifstream mesh_file(filename);
    if (!mesh_file.is_open()) {
        std::cerr << "Error: file '" << filename << "' could not be opened. Exiting." << std::endl;
        exit(7);
    }

    do {
        std::getline(mesh_file, line);  
    }
    while (line.empty());
    
    std::istringstream liness(line);
    liness >> token;
    liness >> value;
    if (token != "NDIME=") {
        std::cerr << "Error: first token should be 'NDIME=', found '" << token << "'. Exiting." << std::endl;
        exit(8);
    }

    if (value != 2) {
        std::cerr << "Error: program only works for 2 dimensions, found '" << value << "'. Exiting." << std::endl;
        exit(9);
    }

    //std::vector<Cell_t> farfield;
    //std::vector<Cell_t> wall;
    //std::vector<Cell_t> inlet;
    std::vector<Vec2<deviceFloat>> host_nodes;
    std::vector<std::array<size_t, 4>> host_element_to_nodes;

    while (!mesh_file.eof()) {
        do {
            std::getline(mesh_file, line);  
        }
        while (line.empty() && !mesh_file.eof());

        std::istringstream liness(line);
        liness >> token;
        std::transform(token.begin(), token.end(), token.begin(),
            [](unsigned char c){ return std::toupper(c); });

        if (token == "NPOIN=") {
            liness >> value;
            host_nodes = std::vector<Vec2<deviceFloat>>(value);

            for (size_t i = 0; i < host_nodes.size(); ++i) {
                std::getline(mesh_file, line);
                std::istringstream liness2(line);
                liness2 >> host_nodes[i].pos_[0] >> host_nodes[i].pos_[1];
            }
        }
        else if (token == "NELEM=") {
            liness >> value;
            host_element_to_nodes = std::vector<std::array<size_t, 4>>(value);
            n_elements_ = value;

            for (size_t i = 0; i < host_element_to_nodes.size(); ++i) {
                std::getline(mesh_file, line);
                std::istringstream liness2(line);
                liness2 >> token;

                if (token == "9") {
                    constexpr int n_sides = 4;

                    for (int j = 0; j < n_sides; ++j) {
                        liness2 >> host_element_to_nodes[i][j];
                    }
                }
                /*else if (token == "5") {
                    constexpr int n_sides = 3;

                    for (int j = 0; j < n_sides; ++j) {
                        liness2 >> host_element_to_nodes[i][j];
                    }
                }*/
                /*else {
                    std::cerr << "Error: expected token '9', found '" << token << "'. Exiting." << std::endl;
                    exit(10);
                }
            }
        }
        else if (token == "NMARK=") {
            int n_markers;
            liness >> n_markers;

            n_farfield_ = 0;
            n_wall_ = 0;
            n_inlet_ = 0;

            for (int i = 0; i < n_markers; ++i) {
                std::string type;
                do {
                    std::getline(mesh_file, line);
                    if (!line.empty()) {
                        std::istringstream liness(line);
                        liness >> token;
                        liness >> type;
                    }   
                }
                while (token != "MARKER_TAG=");
                std::transform(type.begin(), type.end(), type.begin(),
                    [](unsigned char c){ return std::tolower(c); });

                if (type == "farfield") {
                    do {
                        std::getline(mesh_file, line);
                        if (!line.empty()) {
                            std::istringstream liness(line);
                            liness >> token;
                            liness >> value;
                        }   
                    }
                    while (token != "MARKER_ELEMS=");

                    n_farfield_ += value;
                    farfield.reserve(n_farfield_);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(mesh_file, line);
                        std::istringstream liness6(line);

                        liness6 >> token;
                        if (token != "3") {
                            std::cerr << "Error: expected token '3', found '" << token << "'. Exiting." << std::endl;
                            exit(11);
                        }

                        size_t val0, val1;
                        liness6 >> val0 >> val1;
                        farfield.push_back(Cell_t(2));
                        farfield[farfield.size() - 1].nodes_[0] = val0;
                        farfield[farfield.size() - 1].nodes_[1] = val1;
                    }
                }
                else if (type == "wall") {
                    do {
                        std::getline(mesh_file, line);
                        if (!line.empty()) {
                            std::istringstream liness(line);
                            liness >> token;
                            liness >> value;
                        }   
                    }
                    while (token != "MARKER_ELEMS=");

                    n_wall_ += value;
                    wall.reserve(n_wall_);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(mesh_file, line);
                        std::istringstream liness6(line);

                        liness6 >> token;
                        if (token != "3") {
                            std::cerr << "Error: expected token '3', found '" << token << "'. Exiting." << std::endl;
                            exit(12);
                        }

                        size_t val0, val1;
                        liness6 >> val0 >> val1;
                        wall.push_back(Cell_t(2));
                        wall[wall.size() - 1].nodes_[0] = val0;
                        wall[wall.size() - 1].nodes_[1] = val1;
                    }
                }
                else if (type == "inlet") {
                    do {
                        std::getline(mesh_file, line);
                        if (!line.empty()) {
                            std::istringstream liness(line);
                            liness >> token;
                            liness >> value;
                        }   
                    }
                    while (token != "MARKER_ELEMS=");

                    n_inlet_ += value;
                    inlet.reserve(n_inlet_);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(mesh_file, line);
                        std::istringstream liness6(line);

                        liness6 >> token;
                        if (token != "3") {
                            std::cerr << "Error: expected token '3', found '" << token << "'. Exiting." << std::endl;
                            exit(12);
                        }

                        size_t val0, val1;
                        liness6 >> val0 >> val1;
                        inlet.push_back(Cell_t(2));
                        inlet[inlet.size() - 1].nodes_[0] = val0;
                        inlet[inlet.size() - 1].nodes_[1] = val1;
                    }
                }
                else {
                    std::cerr << "Error: expected marker tag 'farfield', 'wall' or 'inlet', found '" << type << "'. Exiting." << std::endl;
                    exit(6);
                }
            }
        }
        else {
            if (!mesh_file.eof()) {
                std::cerr << "Error: expected marker 'NPOIN=', 'NELEM=' or 'NMARK=', found '" << token << "'. Exiting." << std::endl;
                exit(13);
            }
        }
    }

    mesh_file.close();

    cells_.insert(std::end(cells_), std::begin(farfield), std::end(farfield));
    cells_.insert(std::end(cells_), std::begin(wall), std::end(wall));
    cells_.insert(std::end(cells_), std::begin(inlet), std::end(inlet));*/
}

void SEM::Meshes::Mesh2D_t::read_cgns(std::filesystem::path filename) {
    int index_file;
    const int open_error = cg_open(filename.string().c_str(), CG_MODE_READ, &index_file);
    if (open_error != CG_OK) {
        std::cerr << "Error: file '" << filename << "' could not be opened with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(16);
    }

    // Getting base information
    int n_bases = 0;
    cg_nbases(index_file, &n_bases);
    if (n_bases != 1) {
        std::cerr << "Error: CGNS mesh has " << n_bases << " base(s), but for now only a single base is supported. Exiting." << std::endl;
        exit(17);
    }
    const int index_base = 1;

    std::array<char, CGIO_MAX_NAME_LENGTH> base_name; // Oh yeah cause it's the 80s still
    int dim = 0;
    int physDim = 0;
    cg_base_read(index_file, index_base, base_name.data(), &dim, &physDim);
    if (dim != 2) {
        std::cerr << "Error: CGNS mesh, base " << index_base << " has " << dim << " dimensions, but the program only supports 2 dimensions. Exiting." << std::endl;
        exit(18);
    }
    if (physDim != 2) {
        std::cerr << "Error: CGNS mesh, base " << index_base << " has " << physDim << " physical dimensions, but the program only supports 2 physical dimensions. Exiting." << std::endl;
        exit(19);
    }

    // Getting zone information
    int n_zones = 0;
    cg_nzones(index_file, index_base, &n_zones);
    if (n_bases != 1) {
        std::cerr << "Error: CGNS mesh, base " << index_base << " has " << n_zones << " zone(s), but for now only a single zone is supported. Exiting." << std::endl;
        exit(20);
    }
    const int index_zone = 1;

    ZoneType_t zone_type = ZoneType_t::ZoneTypeNull;
    cg_zone_type(index_file, index_base, index_zone, &zone_type);
    if (zone_type != ZoneType_t::Unstructured) {
        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << " is not an unstructured zone. For now only unstructured zones are supported. Exiting." << std::endl;
        exit(21);
    }

    std::array<char, CGIO_MAX_NAME_LENGTH> zone_name; // Oh yeah cause it's the 80s still
    std::array<int, 3> isize{0, 0, 0};
    cg_zone_read(index_file, index_base, index_zone, zone_name.data(), isize.data());
    if (isize[2] != 0) {
        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << " has " << isize[2] << " boundary vertices, but to be honest I'm not sure how to deal with them. Exiting." << std::endl;
        exit(22);
    }
    const int n_nodes = isize[0];
    const int n_elements = isize[1];

    // Getting nodes
    int n_coords = 0;
    cg_ncoords(index_file, index_base, index_zone, &n_coords);
    if (n_coords != 2) {
        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << " has " << n_coords << " sets of coordinates, but for now only two are supported. Exiting." << std::endl;
        exit(23);
    }

    std::array<std::array<char, CGIO_MAX_NAME_LENGTH>, 2> coord_names; // Oh yeah cause it's the 80s still
    std::array<DataType_t, 2> coord_data_types {DataType_t::DataTypeNull, DataType_t::DataTypeNull};
    for (int index_coord = 1; index_coord <= n_coords; ++index_coord) {
        cg_coord_info(index_file, index_base, index_zone, index_coord, &coord_data_types[index_coord - 1], coord_names[index_coord - 1].data());
    }

    std::array<std::vector<double>, 2> xy{std::vector<double>(n_nodes), std::vector<double>(n_nodes)};

    for (int index_coord = 1; index_coord <= n_coords; ++index_coord) {
        const int index_coord_start = 1;
        cg_coord_read(index_file, index_base, index_zone, coord_names[index_coord - 1].data(), DataType_t::RealDouble, &index_coord_start, &n_nodes, xy[index_coord - 1].data());
    }
    
    // Getting connectivity
    int n_sections = 0;
    cg_nsections(index_file, index_base, index_zone, &n_sections);

    std::vector<int> section_data_size(n_sections);
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> section_names(n_sections); // Oh yeah cause it's the 80s still
    std::vector<ElementType_t> section_type(n_sections);
    std::vector<std::array<int, 2>> section_ranges(n_sections);
    std::vector<int> section_n_boundaries(n_sections);
    std::vector<int> section_parent_flags(n_sections);
    for (int index_section = 1; index_section <= n_sections; ++index_section) {
        cg_ElementDataSize(index_file, index_base, index_zone, index_section, &section_data_size[index_section - 1]);
        cg_section_read(index_file, index_base, index_zone, index_section, section_names[index_section - 1].data(), &section_type[index_section - 1], &section_ranges[index_section - 1][0], &section_ranges[index_section - 1][1], &section_n_boundaries[index_section - 1], &section_parent_flags[index_section - 1]);
        if (section_n_boundaries[index_section - 1] != 0) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", section " << index_section << " has " << section_n_boundaries[index_section - 1] << " boundary elements, but to be honest I'm not sure how to deal with them. Exiting." << std::endl;
            exit(24);
        }
    }

    std::vector<std::vector<int>> connectivity(n_sections);
    std::vector<std::vector<int>> parent_data(n_sections);
    for (int index_section = 1; index_section <= n_sections; ++index_section) {
        connectivity[index_section - 1] = std::vector<int>(section_data_size[index_section - 1]);
        parent_data[index_section - 1] = std::vector<int>(section_ranges[index_section - 1][1] - section_ranges[index_section - 1][0]);

        cg_elements_read(index_file, index_base, index_zone, index_section, connectivity[index_section - 1].data(), parent_data[index_section - 1].data());
    }

    // Interfaces
    int n_connectivity = 0;
    cg_nconns(index_file, index_base, index_zone, &n_connectivity);

    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> connectivity_names(n_connectivity); // Oh yeah cause it's the 80s still
    std::vector<GridLocation_t> connectivity_grid_locations(n_connectivity);
    std::vector<GridConnectivityType_t> connectivity_types(n_connectivity);
    std::vector<PointSetType_t> connectivity_point_set_types(n_connectivity);
    std::vector<int> connectivity_sizes(n_connectivity);
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> connectivity_donor_names(n_connectivity); // Oh yeah cause it's the 80s still
    std::vector<ZoneType_t> connectivity_donor_zone_types(n_connectivity);
    std::vector<PointSetType_t> connectivity_donor_point_set_types(n_connectivity);
    std::vector<DataType_t> connectivity_donor_data_types(n_connectivity);
    std::vector<int> connectivity_donor_sizes(n_connectivity);
    for (int index_connectivity = 1; index_connectivity <= n_connectivity; ++index_connectivity) {
        cg_conn_info(index_file, index_base, index_zone, index_connectivity, connectivity_names[index_connectivity - 1].data(),
            &connectivity_grid_locations[index_connectivity - 1], &connectivity_types[index_connectivity - 1],
            &connectivity_point_set_types[index_connectivity - 1], &connectivity_sizes[index_connectivity - 1], connectivity_donor_names[index_connectivity - 1].data(),
            &connectivity_donor_zone_types[index_connectivity - 1], &connectivity_donor_point_set_types[index_connectivity - 1],
            &connectivity_donor_data_types[index_connectivity - 1], &connectivity_donor_sizes[index_connectivity - 1]);

        if (connectivity_donor_zone_types[index_connectivity - 1] != ZoneType_t::Unstructured) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << index_connectivity << " has a donor zone type that isn't unstructured. For now only unstructured zones are supported. Exiting." << std::endl;
            exit(25);
        }
        if (connectivity_point_set_types[index_connectivity - 1] != PointSetType_t::PointList) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << index_connectivity << " has a point set type that isn't PointList. For now only PointList point set types are supported. Exiting." << std::endl;
            exit(26);
        }
        if (connectivity_donor_point_set_types[index_connectivity - 1] != PointSetType_t::PointListDonor) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << index_connectivity << " has a donor point set type that isn't PointListDonor. For now only PointListDonor point set types are supported. Exiting." << std::endl;
            exit(27);
        }

        if (connectivity_grid_locations[index_connectivity - 1] != GridLocation_t::FaceCenter) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << index_connectivity << " has a grid location that isn't FaceCenter. For now only FaceCenter grid locations are supported. Exiting." << std::endl;
            exit(28);
        }

        if (connectivity_types[index_connectivity - 1] != GridConnectivityType_t::Abutting1to1) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << index_connectivity << " has a grid connectivity type that isn't Abutting1to1. For now only Abutting1to1 grid connectivity types are supported. Exiting." << std::endl;
            exit(29);
        }
    }

    std::vector<std::vector<int>> interface_elements(n_connectivity);
    std::vector<std::vector<int>> interface_donor_elements(n_connectivity);
    for (int index_connectivity = 1; index_connectivity <= n_connectivity; ++index_connectivity) {
        interface_elements[index_connectivity - 1] = std::vector<int>(connectivity_sizes[index_connectivity - 1]);
        interface_donor_elements[index_connectivity - 1] = std::vector<int>(connectivity_donor_sizes[index_connectivity - 1]);
        cg_conn_read(index_file, index_base, index_zone, index_connectivity, interface_elements[index_connectivity - 1].data(),
            DataType_t::Integer, interface_donor_elements[index_connectivity - 1].data());
    }

    // Boundary conditions
    int n_boundaries = 0;
    cg_nbocos(index_file, index_base, index_zone, &n_boundaries);

    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> boundary_names(n_boundaries); // Oh yeah cause it's the 80s still
    std::vector<BCType_t> boundary_types(n_boundaries);
    std::vector<PointSetType_t> boundary_point_set_types(n_boundaries);
    std::vector<int> boundary_sizes(n_boundaries);
    std::vector<int> boundary_normal_indices(n_boundaries);
    std::vector<int> boundary_normal_list_sizes(n_boundaries);
    std::vector<DataType_t> boundary_normal_data_types(n_boundaries);
    std::vector<int> boundary_n_datasets(n_boundaries);
    std::vector<GridLocation_t> boundary_grid_locations(n_boundaries);
    for (int index_boundary = 1; index_boundary <= n_boundaries; ++index_boundary) {
        cg_boco_info(index_file, index_base, index_zone, index_boundary, boundary_names[index_boundary - 1].data(),
            &boundary_types[index_boundary - 1], &boundary_point_set_types[index_boundary - 1], &boundary_sizes[index_boundary - 1],
            &boundary_normal_indices[index_boundary - 1], &boundary_normal_list_sizes[index_boundary - 1],
            &boundary_normal_data_types[index_boundary - 1], &boundary_n_datasets[index_boundary - 1]);

        if (boundary_point_set_types[index_boundary - 1] != PointSetType_t::PointList) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << index_boundary << " has a point set type that isn't PointList. For now only PointList point set types are supported. Exiting." << std::endl;
            exit(30);
        }

        cg_boco_gridlocation_read(index_file, index_base, index_zone, index_boundary, &boundary_grid_locations[index_boundary - 1]);

        if (boundary_grid_locations[index_boundary - 1] != GridLocation_t::EdgeCenter) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << index_boundary << " has a grid location that isn't EdgeCenter. For now only EdgeCenter grid locations are supported. Exiting." << std::endl;
            exit(31);
        }
    }

    std::vector<std::vector<int>> boundary_elements(n_boundaries);
    std::vector<std::vector<int>> boundary_normals(n_boundaries);
    for (int index_boundary = 1; index_boundary <= n_boundaries; ++index_boundary) {
        boundary_elements[index_boundary - 1] = std::vector<int>(boundary_sizes[index_boundary - 1]);
        boundary_normals[index_boundary - 1] = std::vector<int>(boundary_normal_list_sizes[index_boundary - 1]);
        cg_boco_read(index_file, index_base, index_zone, index_boundary, boundary_elements[index_boundary - 1].data(), boundary_normals[index_boundary - 1].data());
    }

    cg_close(index_file);

    // Putting everything in the format used by the mesh
    std::vector<Vec2<deviceFloat>> host_nodes(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        host_nodes[i].x() = xy[0][i];
        host_nodes[i].y() = xy[1][i];
    }

    std::vector<bool> section_is_domain(n_sections);
    int n_elements_domain = 0;
    int n_elements_ghost = 0;
    for (int i = 0; i < n_sections; ++i) {
        switch (section_type[n_sections]) {
            case ElementType_t::BAR_2:
                section_is_domain[i] = false;
                n_elements_ghost += section_ranges[i][1] - section_ranges[i][0];
                break;

            case ElementType_t::QUAD_4:
                section_is_domain[i] = true;
                n_elements_domain += section_ranges[i][1] - section_ranges[i][0];
                break;

            default:
                std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", section " << i << " has an unknown element type. For now only BAR_2 and QUAD_4 are implemented, for boundaries and domain respectively. Exiting." << std::endl;
                exit(32);
        }
    }

    if (n_elements_domain + n_elements_ghost != n_elements) {
        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << " has " << n_elements << " elements but the sum of its sections is " << n_elements_domain + n_elements_ghost << " elements. Exiting." << std::endl;
        exit(33);
    }


}

void SEM::Meshes::Mesh2D_t::set_initial_conditions(const deviceFloat* nodes) {

}

void SEM::Meshes::Mesh2D_t::print() {
    
}

void SEM::Meshes::Mesh2D_t::write_file_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& coordinates, const std::vector<deviceFloat>& velocity, const std::vector<deviceFloat>& du_dx, const std::vector<deviceFloat>& intermediate, const std::vector<deviceFloat>& x_L, const std::vector<deviceFloat>& x_R, const std::vector<int>& N, const std::vector<deviceFloat>& sigma, const bool* refine, const bool* coarsen, const std::vector<deviceFloat>& error) {
    
}

void SEM::Meshes::Mesh2D_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices) {

}

template void SEM::Meshes::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const SEM::Entities::NDG_t<SEM::Polynomials::ChebyshevPolynomial_t> &NDG, deviceFloat viscosity); // Get with the times c++, it's crazy I have to do this
template void SEM::Meshes::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> &NDG, deviceFloat viscosity);

template<typename Polynomial>
void SEM::Meshes::Mesh2D_t::solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const SEM::Entities::NDG_t<Polynomial> &NDG, deviceFloat viscosity) {
    
}

deviceFloat SEM::Meshes::Mesh2D_t::get_delta_t(const deviceFloat CFL) {   
    return 0.0;
}

void SEM::Meshes::Mesh2D_t::adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) {
    
}

void SEM::Meshes::Mesh2D_t::boundary_conditions() {
    
}
