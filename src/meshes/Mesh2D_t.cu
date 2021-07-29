#include "meshes/Mesh2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "helpers/constants.h"
#include "functions/Utilities.h"
#include "functions/quad_map.cuh"
#include "cgnslib.h"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <limits>

namespace fs = std::filesystem;

using SEM::Entities::device_vector;
using SEM::Entities::host_vector;
using SEM::Entities::Vec2;
using SEM::Entities::Element2D_t;
using SEM::Entities::Face2D_t;

constexpr int CGIO_MAX_NAME_LENGTH = 33; // Includes the null terminator

SEM::Meshes::Mesh2D_t::Mesh2D_t(std::filesystem::path filename, int initial_N, int maximum_N, size_t n_interpolation_points, int max_split_level, int adaptivity_interval, deviceFloat tolerance_min, deviceFloat tolerance_max, const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, const cudaStream_t &stream) :       
        initial_N_{initial_N},  
        maximum_N_{maximum_N},
        n_interpolation_points_{n_interpolation_points},
        max_split_level_{max_split_level},
        adaptivity_interval_{adaptivity_interval},
        tolerance_min_{tolerance_min},
        tolerance_max_{tolerance_max},
        stream_{stream} {

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

    compute_element_geometry<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, elements_.data(), nodes_.data(), polynomial_nodes.data());
    compute_boundary_geometry<<<ghosts_numBlocks_, boundaries_blockSize_, 0, stream_>>>(n_elements_, elements_.size(), elements_.data(), nodes_.data(), polynomial_nodes.data());
    compute_face_geometry<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data(), elements_.data(), nodes_.data());
}

auto SEM::Meshes::Mesh2D_t::read_su2(std::filesystem::path filename) -> void {
    std::cerr << "Error: SU2 meshes not implemented yet. Exiting." << std::endl;
    exit(15);
}

auto SEM::Meshes::Mesh2D_t::read_cgns(std::filesystem::path filename) -> void {
    int index_file = 0;
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
    constexpr int index_base = 1;

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

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    if (n_zones != global_size) {
        std::cerr << "Error: CGNS mesh, base " << index_base << " has " << n_zones << " zone(s), but the program has been run with " << global_size << " process(es). For now only a single zone per process is supported. Exiting." << std::endl;
        exit(48);
    }

    // Getting all zone names
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> zone_names(n_zones);
    for (int i = 0; i < n_zones; ++i) {
        std::array<cgsize_t, 3> temp{0, 0, 0};
        cg_zone_read(index_file, index_base, i + 1, zone_names[i].data(), temp.data());
    }

    const int index_zone = global_rank + 1;

    ZoneType_t zone_type = ZoneType_t::ZoneTypeNull;
    cg_zone_type(index_file, index_base, index_zone, &zone_type);
    if (zone_type != ZoneType_t::Unstructured) {
        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << " is not an unstructured zone. For now only unstructured zones are supported. Exiting." << std::endl;
        exit(21);
    }

    std::array<char, CGIO_MAX_NAME_LENGTH> zone_name; // Oh yeah cause it's the 80s still
    std::array<cgsize_t, 3> isize{0, 0, 0};
    cg_zone_read(index_file, index_base, index_zone, zone_name.data(), isize.data());
    if (isize[2] != 0) {
        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << " has " << isize[2] << " boundary vertices, but to be honest I'm not sure how to deal with them. Exiting." << std::endl;
        exit(22);
    }
    const cgsize_t n_nodes = isize[0];
    const cgsize_t n_elements = isize[1];

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
        const cgsize_t index_coord_start = 1;
        cg_coord_read(index_file, index_base, index_zone, coord_names[index_coord - 1].data(), DataType_t::RealDouble, &index_coord_start, &n_nodes, xy[index_coord - 1].data());
    }
    
    // Getting connectivity
    int n_sections = 0;
    cg_nsections(index_file, index_base, index_zone, &n_sections);

    std::vector<cgsize_t> section_data_size(n_sections);
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> section_names(n_sections); // Oh yeah cause it's the 80s still
    std::vector<ElementType_t> section_type(n_sections);
    std::vector<std::array<cgsize_t, 2>> section_ranges(n_sections);
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

    std::vector<std::vector<cgsize_t>> connectivity(n_sections);
    std::vector<std::vector<cgsize_t>> parent_data(n_sections);
    for (int index_section = 1; index_section <= n_sections; ++index_section) {
        connectivity[index_section - 1] = std::vector<cgsize_t>(section_data_size[index_section - 1]);
        parent_data[index_section - 1] = std::vector<cgsize_t>(section_ranges[index_section - 1][1] - section_ranges[index_section - 1][0] + 1);

        cg_elements_read(index_file, index_base, index_zone, index_section, connectivity[index_section - 1].data(), parent_data[index_section - 1].data());
    }

    // Interfaces
    int n_connectivity = 0;
    cg_nconns(index_file, index_base, index_zone, &n_connectivity);

    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> connectivity_names(n_connectivity); // Oh yeah cause it's the 80s still
    std::vector<GridLocation_t> connectivity_grid_locations(n_connectivity);
    std::vector<GridConnectivityType_t> connectivity_types(n_connectivity);
    std::vector<PointSetType_t> connectivity_point_set_types(n_connectivity);
    std::vector<cgsize_t> connectivity_sizes(n_connectivity);
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> connectivity_donor_names(n_connectivity); // Oh yeah cause it's the 80s still
    std::vector<ZoneType_t> connectivity_donor_zone_types(n_connectivity);
    std::vector<PointSetType_t> connectivity_donor_point_set_types(n_connectivity);
    std::vector<DataType_t> connectivity_donor_data_types(n_connectivity);
    std::vector<cgsize_t> connectivity_donor_sizes(n_connectivity);
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

        if (connectivity_sizes[index_connectivity - 1] != connectivity_donor_sizes[index_connectivity - 1]) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << index_connectivity << " has a different number of elements in the origin and destination zones. Exiting." << std::endl;
            exit(30);
        }
    }

    std::vector<std::vector<cgsize_t>> interface_elements(n_connectivity);
    std::vector<std::vector<cgsize_t>> interface_donor_elements(n_connectivity);
    for (int index_connectivity = 1; index_connectivity <= n_connectivity; ++index_connectivity) {
        interface_elements[index_connectivity - 1] = std::vector<cgsize_t>(connectivity_sizes[index_connectivity - 1]);
        interface_donor_elements[index_connectivity - 1] = std::vector<cgsize_t>(connectivity_donor_sizes[index_connectivity - 1]);
        cg_conn_read(index_file, index_base, index_zone, index_connectivity, interface_elements[index_connectivity - 1].data(),
            DataType_t::Integer, interface_donor_elements[index_connectivity - 1].data());
    }

    // Boundary conditions
    int n_boundaries = 0;
    cg_nbocos(index_file, index_base, index_zone, &n_boundaries);

    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> boundary_names(n_boundaries); // Oh yeah cause it's the 80s still
    std::vector<BCType_t> boundary_types(n_boundaries);
    std::vector<PointSetType_t> boundary_point_set_types(n_boundaries);
    std::vector<cgsize_t> boundary_sizes(n_boundaries);
    std::vector<int> boundary_normal_indices(n_boundaries);
    std::vector<cgsize_t> boundary_normal_list_sizes(n_boundaries);
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
            exit(31);
        }

        cg_boco_gridlocation_read(index_file, index_base, index_zone, index_boundary, &boundary_grid_locations[index_boundary - 1]);

        if (boundary_grid_locations[index_boundary - 1] != GridLocation_t::EdgeCenter) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << index_boundary << " has a grid location that isn't EdgeCenter. For now only EdgeCenter grid locations are supported. Exiting." << std::endl;
            exit(32);
        }
    }

    std::vector<std::vector<cgsize_t>> boundary_elements(n_boundaries);
    std::vector<std::vector<cgsize_t>> boundary_normals(n_boundaries);
    for (int index_boundary = 1; index_boundary <= n_boundaries; ++index_boundary) {
        boundary_elements[index_boundary - 1] = std::vector<cgsize_t>(boundary_sizes[index_boundary - 1]);
        boundary_normals[index_boundary - 1] = std::vector<cgsize_t>(boundary_normal_list_sizes[index_boundary - 1]);
        cg_boco_read(index_file, index_base, index_zone, index_boundary, boundary_elements[index_boundary - 1].data(), boundary_normals[index_boundary - 1].data());
    }

    const int close_error = cg_close(index_file);
    if (close_error != CG_OK) {
        std::cerr << "Error: file '" << filename << "' could not be closed with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(44);
    }

    // Putting nodes in the format used by the mesh
    std::vector<Vec2<deviceFloat>> host_nodes(n_nodes);
    for (cgsize_t i = 0; i < n_nodes; ++i) {
        host_nodes[i].x() = xy[0][i];
        host_nodes[i].y() = xy[1][i];
    }

    // Transferring onto the GPU
    nodes_ = device_vector<Vec2<deviceFloat>>(host_nodes, stream_);

    // Figuring out which sections are the domain and which are ghost cells
    std::vector<bool> section_is_domain(n_sections);
    cgsize_t n_elements_domain = 0;
    cgsize_t n_elements_ghost = 0;
    for (int i = 0; i < n_sections; ++i) {
        switch (section_type[i]) {
            case ElementType_t::BAR_2:
                section_is_domain[i] = false;
                n_elements_ghost += section_ranges[i][1] - section_ranges[i][0] + 1;
                break;

            case ElementType_t::QUAD_4:
                section_is_domain[i] = true;
                n_elements_domain += section_ranges[i][1] - section_ranges[i][0] + 1;
                break;

            default:
                std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", section " << i << " has an unknown element type. For now only BAR_2 and QUAD_4 are implemented, for boundaries and domain respectively. Exiting." << std::endl;
                exit(33);
        }
    }

    if (n_elements_domain + n_elements_ghost != n_elements) {
        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << " has " << n_elements << " elements but the sum of its sections is " << n_elements_domain + n_elements_ghost << " elements. Exiting." << std::endl;
        exit(34);
    }

    // Putting connectivity data in the format used by the mesh
    std::vector<Element2D_t> host_elements(n_elements);
    std::vector<size_t> section_start_indices(n_sections);
    size_t element_domain_index = 0;
    size_t element_ghost_index = n_elements_domain;
    for (int i = 0; i < n_sections; ++i) {
        if (section_is_domain[i]) {
            section_start_indices[i] = element_domain_index;
            for (cgsize_t j = 0; j < section_ranges[i][1] - section_ranges[i][0] + 1; ++j) {
                Element2D_t& element = host_elements[section_start_indices[i] + j];
                element.N_ = initial_N_;
                element.nodes_ = {static_cast<size_t>(connectivity[i][4 * j] - 1),
                                  static_cast<size_t>(connectivity[i][4 * j + 1] - 1),
                                  static_cast<size_t>(connectivity[i][4 * j + 2] - 1),
                                  static_cast<size_t>(connectivity[i][4 * j + 3] - 1)};       
            }
            element_domain_index += section_ranges[i][1] - section_ranges[i][0] + 1;
        }
        else {
            section_start_indices[i] = element_ghost_index;
            for (cgsize_t j = 0; j < section_ranges[i][1] - section_ranges[i][0] + 1; ++j) {
                Element2D_t& element = host_elements[section_start_indices[i] + j];
                element.N_ = initial_N_;
                element.nodes_ = {static_cast<size_t>(connectivity[i][2 * j] - 1),
                                  static_cast<size_t>(connectivity[i][2 * j + 1] - 1),
                                  static_cast<size_t>(connectivity[i][2 * j + 1] - 1),
                                  static_cast<size_t>(connectivity[i][2 * j] - 1)};
            }
            element_ghost_index += section_ranges[i][1] - section_ranges[i][0] + 1;
        }
    }

    // Computing nodes to elements
    const std::vector<std::vector<size_t>> node_to_element = build_node_to_element(n_nodes, host_elements);

    // Computing element to elements
    const std::vector<std::vector<size_t>> element_to_element = build_element_to_element(host_elements, node_to_element);

    // Computing faces and filling element faces
    auto [host_faces, node_to_face, element_to_face] = build_faces(n_elements_domain, n_nodes, initial_N_, host_elements);

    // Transferring onto the GPU
    elements_ = device_vector<Element2D_t>(host_elements, stream_);
    faces_ = device_vector<Face2D_t>(host_faces, stream_);

    // Building boundaries
    std::vector<size_t> wall_boundaries;
    std::vector<size_t> symmetry_boundaries;
    std::vector<size_t> inflow_boundaries;
    std::vector<size_t> outflow_boundaries;

    for (int i = 0; i < n_boundaries; ++i) {
        switch (boundary_types[i]) {
            case BCType_t::BCWall:
                wall_boundaries.reserve(wall_boundaries.size() + boundary_sizes[i]);
                for (cgsize_t j = 0; j < boundary_sizes[i]; ++j) {
                    int section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((boundary_elements[i][j] >= section_ranges[k][0]) && (boundary_elements[i][j] <= section_ranges[k][1])) {
                            section_index = k;
                            break;
                        }
                    }

                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << i << " is a wall boundary and contains element " << boundary_elements[i][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(36);
                    }

                    wall_boundaries.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
                }
                break;

            case BCType_t::BCSymmetryPlane:
                symmetry_boundaries.reserve(symmetry_boundaries.size() + boundary_sizes[i]);
                for (cgsize_t j = 0; j < boundary_sizes[i]; ++j) {
                    int section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((boundary_elements[i][j] >= section_ranges[k][0]) && (boundary_elements[i][j] <= section_ranges[k][1])) {
                            section_index = k;
                            break;
                        }
                    }

                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << i << " is a symmetry boundary and contains element " << boundary_elements[i][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(37);
                    }

                    symmetry_boundaries.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
                }
                
                break;

            case BCType_t::BCInflow:
                inflow_boundaries.reserve(inflow_boundaries.size() + boundary_sizes[i]);
                for (cgsize_t j = 0; j < boundary_sizes[i]; ++j) {
                    int section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((boundary_elements[i][j] >= section_ranges[k][0]) && (boundary_elements[i][j] <= section_ranges[k][1])) {
                            section_index = k;
                            break;
                        }
                    }

                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << i << " is an inflow boundary and contains element " << boundary_elements[i][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(36);
                    }

                    inflow_boundaries.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
                }

                break;
            
            case BCType_t::BCOutflow:
                outflow_boundaries.reserve(outflow_boundaries.size() + boundary_sizes[i]);
                for (cgsize_t j = 0; j < boundary_sizes[i]; ++j) {
                    int section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((boundary_elements[i][j] >= section_ranges[k][0]) && (boundary_elements[i][j] <= section_ranges[k][1])) {
                            section_index = k;
                            break;
                        }
                    }

                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << i << " is an outflow boundary and contains element " << boundary_elements[i][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(36);
                    }

                    outflow_boundaries.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
                }
            
                break;

            case BCType_t::BCTypeNull:
                break;

            default:
                std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << i << " has an unknown boundary type. For now only BCWall, BCSymmetryPlane and BCTypeNull are implemented. Exiting." << std::endl;
                exit(35);
        }
    }

    // Transferring onto the GPU
    if (!wall_boundaries.empty()) {
        wall_boundaries_ = device_vector<size_t>(wall_boundaries, stream_);
    }
    if (!symmetry_boundaries.empty()) {
        symmetry_boundaries_ = device_vector<size_t>(symmetry_boundaries, stream_);
    }
    if (!inflow_boundaries.empty()) {
        inflow_boundaries_ = device_vector<size_t>(inflow_boundaries, stream_);
    }
    if (!outflow_boundaries.empty()) {
        outflow_boundaries_ = device_vector<size_t>(outflow_boundaries, stream_);
    }

    // Building self interfaces
    size_t n_interface_elements = 0;
    std::vector<size_t> interface_start_index(n_connectivity);
    for (int i = 0; i < n_connectivity; ++i) {
        if (strncmp(zone_name.data(), connectivity_donor_names[i].data(), CGIO_MAX_NAME_LENGTH) == 0) {
            interface_start_index[i] = n_interface_elements;
            n_interface_elements += connectivity_sizes[i];
        }
    }

    if (n_interface_elements > 0) {
        std::vector<size_t> interfaces_origin(n_interface_elements);
        std::vector<size_t> interfaces_origin_side(n_interface_elements);
        std::vector<size_t> interfaces_destination(n_interface_elements);

        for (int i = 0; i < n_connectivity; ++i) {
            if (strncmp(zone_name.data(), connectivity_donor_names[i].data(), CGIO_MAX_NAME_LENGTH) == 0) {
                for (cgsize_t j = 0; j < connectivity_sizes[i]; ++j) {
                    int origin_section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((interface_elements[i][j] >= section_ranges[k][0]) && (interface_elements[i][j] <= section_ranges[k][1])) {
                            origin_section_index = k;
                            break;
                        }
                    }

                    if (origin_section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << i << " contains element " << interface_elements[i][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(38);
                    }

                    int donor_section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((interface_donor_elements[i][j] >= section_ranges[k][0]) && (interface_donor_elements[i][j] <= section_ranges[k][1])) {
                            donor_section_index = k;
                            break;
                        }
                    }

                    if (donor_section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << i << " contains donor element " << interface_donor_elements[i][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(39);
                    }

                    const size_t donor_boundary_element_index = section_start_indices[donor_section_index] + interface_donor_elements[i][j] - section_ranges[donor_section_index][0];
                    const size_t face_index = element_to_face[donor_boundary_element_index][0];
                    const size_t face_side_index = host_faces[face_index].elements_[0] == donor_boundary_element_index;
                    const size_t donor_domain_element_index = host_faces[face_index].elements_[face_side_index];
                    
                    interfaces_origin[interface_start_index[i] + j] = donor_domain_element_index;
                    interfaces_origin_side[interface_start_index[i] + j] = host_faces[face_index].elements_side_[face_side_index];
                    interfaces_destination[interface_start_index[i] + j] = section_start_indices[origin_section_index] + interface_elements[i][j] - section_ranges[origin_section_index][0];
                }
            }
        }

        // Transferring onto the GPU
        interfaces_origin_ = device_vector<size_t>(interfaces_origin, stream_);
        interfaces_origin_side_ = device_vector<size_t>(interfaces_origin_side, stream_);
        interfaces_destination_ = device_vector<size_t>(interfaces_destination, stream_);
    }

    // Building MPI interfaces
    // These will be backwards due to how I did the element_side thing. Shouldn't affect much. If it does, just MPI transmit 
    std::vector<size_t> mpi_interface_process(n_connectivity, global_rank);
    std::vector<bool> process_used_in_interface(n_zones);
    size_t n_mpi_interface_elements = 0;
    for (int i = 0; i < n_connectivity; ++i) {
        if (strncmp(zone_name.data(), connectivity_donor_names[i].data(), CGIO_MAX_NAME_LENGTH) != 0) {
            mpi_interface_process[i] = global_rank;
            for (int j = 0; j < n_zones; ++j) {
                if (strncmp(connectivity_donor_names[i].data(), zone_names[j].data(), CGIO_MAX_NAME_LENGTH) == 0) {
                    mpi_interface_process[i] = j;
                    process_used_in_interface[j] = true;
                    break;
                }
            }
            if (mpi_interface_process[i] == global_rank) {
                std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << i << " links to zone \"" << connectivity_donor_names[i].data() << "\" but it is not found in any mesh section. Exiting." << std::endl;
                exit(50);
            }
            n_mpi_interface_elements += connectivity_sizes[i];
        }
    }

    if (n_mpi_interface_elements > 0) {
        size_t n_mpi_interfaces = 0;
        for (int j = 0; j < n_zones; ++j) {
            n_mpi_interfaces += process_used_in_interface[j];
        }

        mpi_interfaces_size_ = std::vector<size_t>(n_mpi_interfaces);
        mpi_interfaces_offset_ = std::vector<size_t>(n_mpi_interfaces);
        mpi_interfaces_process_ = std::vector<size_t>(n_mpi_interfaces);
        std::vector<size_t> mpi_interfaces_origin(n_mpi_interface_elements);
        std::vector<size_t> mpi_interfaces_origin_side(n_mpi_interface_elements);
        std::vector<size_t> mpi_interfaces_destination(n_mpi_interface_elements);
        std::vector<size_t> mpi_interfaces_destination_in_this_proc(n_mpi_interface_elements);

        size_t mpi_interface_offset = 0;
        size_t mpi_interface_index = 0;
        for (int j = 0; j < n_zones; ++j) {
            if (process_used_in_interface[j]) {
                mpi_interfaces_offset_[mpi_interface_index] = mpi_interface_offset;
                mpi_interfaces_process_[mpi_interface_index] = j;
                for (int i = 0; i < n_connectivity; ++i) {
                    if (mpi_interface_process[i] == j) {
                        mpi_interfaces_size_[mpi_interface_index] += connectivity_sizes[i];
                        for (size_t k = 0; k < connectivity_sizes[i]; ++k) {
                            int origin_section_index = -1;
                            for (int m = 0; m < n_sections; ++m) {
                                if ((interface_elements[i][k] >= section_ranges[m][0]) && (interface_elements[i][k] <= section_ranges[m][1])) {
                                    origin_section_index = m;
                                    break;
                                }
                            }

                            if (origin_section_index == -1) {
                                std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << i << " contains element " << interface_elements[i][k] << " but it is not found in any mesh section. Exiting." << std::endl;
                                exit(38);
                            }

                            // Starts to be backwards here
                            const size_t boundary_element_index = section_start_indices[origin_section_index] + interface_elements[i][k] - section_ranges[origin_section_index][0];
                            const size_t face_index = element_to_face[boundary_element_index][0];
                            const size_t face_side_index = host_faces[face_index].elements_[0] == boundary_element_index;

                            mpi_interfaces_origin[mpi_interface_offset + k]      = host_faces[face_index].elements_[face_side_index];;
                            mpi_interfaces_origin_side[mpi_interface_offset + k] = host_faces[face_index].elements_side_[face_side_index];
                            mpi_interfaces_destination[mpi_interface_offset + k] = interface_donor_elements[i][k]; // Still in local referential, will have to exchange info to know.
                        }

                        mpi_interface_offset += connectivity_sizes[i];
                    }
                }

                ++mpi_interface_index;
            }
        }

        // Exchanging mpi interfaces destination
        std::vector<MPI_Request> adaptivity_requests(2 * n_mpi_interfaces);
        std::vector<MPI_Status> adaptivity_statuses(2 * n_mpi_interfaces);
        constexpr MPI_Datatype data_type = (sizeof(size_t) == sizeof(unsigned long long)) ? MPI_UNSIGNED_LONG_LONG : (sizeof(size_t) == sizeof(unsigned long)) ? MPI_UNSIGNED_LONG : MPI_UNSIGNED; // CHECK this is a bad way of doing this

        for (size_t i = 0; i < n_mpi_interfaces; ++i) {
            MPI_Isend(mpi_interfaces_destination.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i], data_type, mpi_interfaces_process_[i], global_size * global_rank + mpi_interfaces_process_[i], MPI_COMM_WORLD, &adaptivity_requests[n_mpi_interfaces + i]);
            MPI_Irecv(mpi_interfaces_destination_in_this_proc.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i], data_type, mpi_interfaces_process_[i],  global_size * mpi_interfaces_process_[i] + global_rank, MPI_COMM_WORLD, &adaptivity_requests[i]);
        }

        MPI_Waitall(n_mpi_interfaces, adaptivity_requests.data(), adaptivity_statuses.data());

        for (size_t i = 0; i < n_mpi_interfaces; ++i) {
            for (size_t j = 0; j < mpi_interfaces_size_[i]; ++j) {
                int donor_section_index = -1;
                for (int k = 0; k < n_sections; ++k) {
                    if ((mpi_interfaces_destination_in_this_proc[mpi_interfaces_offset_[i] + j] >= section_ranges[k][0]) && (mpi_interfaces_destination_in_this_proc[mpi_interfaces_offset_[i] + j] <= section_ranges[k][1])) {
                        donor_section_index = k;
                        break;
                    }
                }

                if (donor_section_index == -1) {
                    std::cerr << "Error: Process " << mpi_interfaces_process_[i] << " sent element " << mpi_interfaces_destination_in_this_proc[mpi_interfaces_offset_[i] + j] << " to process " << global_rank << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(51);
                }

                mpi_interfaces_destination_in_this_proc[mpi_interfaces_offset_[i] + j] = section_start_indices[donor_section_index] + mpi_interfaces_destination_in_this_proc[mpi_interfaces_offset_[i] + j] - section_ranges[donor_section_index][0];
            }
        }

        MPI_Waitall(n_mpi_interfaces, adaptivity_requests.data() + n_mpi_interfaces, adaptivity_statuses.data() + n_mpi_interfaces);

        // Transferring onto the GPU
        mpi_interfaces_origin_ = device_vector<size_t>(mpi_interfaces_origin, stream_);
        mpi_interfaces_origin_side_ = device_vector<size_t>(mpi_interfaces_origin_side, stream_);
        mpi_interfaces_destination_ = device_vector<size_t>(mpi_interfaces_destination_in_this_proc, stream_);

        device_interfaces_p_ = device_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1), stream_);
        device_interfaces_u_ = device_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1), stream_);
        device_interfaces_v_ = device_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1), stream_);
        device_interfaces_N_ = device_vector<int>(mpi_interfaces_origin.size(), stream_);
        host_interfaces_p_ = host_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
        host_interfaces_u_ = host_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
        host_interfaces_v_ = host_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
        host_interfaces_N_ = std::vector<int>(mpi_interfaces_origin.size());
        host_receiving_interfaces_p_ = host_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
        host_receiving_interfaces_u_ = host_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
        host_receiving_interfaces_v_ = host_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
        host_receiving_interfaces_N_ = std::vector<int>(mpi_interfaces_origin.size());

        requests_ = std::vector<MPI_Request>(n_mpi_interfaces * 6);
        statuses_ = std::vector<MPI_Status>(n_mpi_interfaces * 6);
        requests_N_ = std::vector<MPI_Request>(n_mpi_interfaces * 2);
        statuses_N_ = std::vector<MPI_Status>(n_mpi_interfaces * 2);
    }

    // Setting sizes
    n_elements_ = n_elements_domain;
    elements_numBlocks_ = (n_elements_ + elements_blockSize_ - 1) / elements_blockSize_;
    faces_numBlocks_ = (faces_.size() + faces_blockSize_ - 1) / faces_blockSize_;
    wall_boundaries_numBlocks_ = (wall_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    symmetry_boundaries_numBlocks_ = (symmetry_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    inflow_boundaries_numBlocks_ = (inflow_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    outflow_boundaries_numBlocks_ = (outflow_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    ghosts_numBlocks_ = (n_elements_ghost + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    interfaces_numBlocks_ = (interfaces_origin_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    mpi_interfaces_numBlocks_ = (mpi_interfaces_origin_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;

    // Sharing number of elements to calculate offset
    std::vector<size_t> n_elements_per_process(global_size);
    constexpr MPI_Datatype size_t_data_type = (sizeof(size_t) == sizeof(unsigned long long)) ? MPI_UNSIGNED_LONG_LONG : (sizeof(size_t) == sizeof(unsigned long)) ? MPI_UNSIGNED_LONG : MPI_UNSIGNED; // CHECK this is a bad way of doing this
    MPI_Allgather(&n_elements_, 1, size_t_data_type, n_elements_per_process.data(), 1, size_t_data_type, MPI_COMM_WORLD);

    size_t n_elements_global = 0;
    for (int i = 0; i < global_rank; ++i) {
        n_elements_global += n_elements_per_process[i];
    }
    global_element_offset_ = n_elements_global;
    for (int i = global_rank; i < global_size; ++i) {
        n_elements_global += n_elements_per_process[i];
    }
    n_elements_global_ = n_elements_global;

    // Transfer arrays
    host_delta_t_array_ = host_vector<deviceFloat>(elements_numBlocks_);
    device_delta_t_array_ = device_vector<deviceFloat>(elements_numBlocks_, stream_);
    host_refine_array_ = std::vector<size_t>(elements_numBlocks_);
    device_refine_array_ = device_vector<size_t>(elements_numBlocks_, stream_);
    host_faces_refine_array_ = std::vector<size_t>(faces_numBlocks_);
    device_faces_refine_array_ = device_vector<size_t>(faces_numBlocks_, stream_);

    allocate_element_storage<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, elements_.data());
    allocate_boundary_storage<<<ghosts_numBlocks_, boundaries_blockSize_, 0, stream_>>>(n_elements_, elements_.size(), elements_.data());
    allocate_face_storage<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data());

    device_vector<std::array<size_t, 4>> device_element_to_face(element_to_face, stream_);
    fill_element_faces<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(elements_.size(), elements_.data(), device_element_to_face.data());
    device_element_to_face.clear(stream_);

    // Allocating output arrays
    x_output_host_ = std::vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    y_output_host_ = std::vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    p_output_host_ = std::vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    u_output_host_ = std::vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    v_output_host_ = std::vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    x_output_device_ = device_vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2), stream_);
    y_output_device_ = device_vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2), stream_);
    p_output_device_ = device_vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2), stream_);
    u_output_device_ = device_vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2), stream_);
    v_output_device_ = device_vector<deviceFloat>(n_elements_ * std::pow(n_interpolation_points_, 2), stream_);
}

auto SEM::Meshes::Mesh2D_t::build_node_to_element(size_t n_nodes, const std::vector<Element2D_t>& elements) -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> node_to_element(n_nodes);

    for (size_t j = 0; j < elements.size(); ++j) {
        for (auto node_index: elements[j].nodes_) {
            if (std::find(node_to_element[node_index].begin(), node_to_element[node_index].end(), j) == node_to_element[node_index].end()) { // This will be slower, but is needed because boundaries have 4 sides and not 2. Remove when variable geometry elements are added.
                node_to_element[node_index].push_back(j);
            }
        }
    }

    return node_to_element;
}

auto SEM::Meshes::Mesh2D_t::build_element_to_element(const std::vector<Element2D_t>& elements, const std::vector<std::vector<size_t>>& node_to_element) -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> element_to_element(elements.size());

    for (size_t i = 0; i < elements.size(); ++i) {
        const Element2D_t& element = elements[i];
        element_to_element[i] = std::vector<size_t>(element.nodes_.size());

        for (size_t j = 0; j < element.nodes_.size(); ++j) {
            const size_t node_index = element.nodes_[j];
            const size_t node_index_next = (j + 1 < element.nodes_.size()) ? element.nodes_[j + 1] : element.nodes_[0];

            for (auto element_index : node_to_element[node_index]) {
                if (element_index != i) {
                    const Element2D_t& element_neighbor = elements[element_index];

                    auto it = std::find(element_neighbor.nodes_.begin(), element_neighbor.nodes_.end(), node_index);
                    if (it != element_neighbor.nodes_.end()) {
                        const size_t node_element_index = it - element_neighbor.nodes_.begin();
                        
                        for (size_t node_element_index_next = 0; node_element_index_next < node_element_index; ++node_element_index_next) {
                            if (element_neighbor.nodes_[node_element_index_next] == node_index_next) {
                                element_to_element[i][j] = element_index;
                                goto endloop; // I hate this too don't worry
                            }
                        }

                        for (size_t node_element_index_next = node_element_index + 1; node_element_index_next < element_neighbor.nodes_.size(); ++node_element_index_next) {
                            if (element_neighbor.nodes_[node_element_index_next] == node_index_next) {
                                element_to_element[i][j] = element_index;
                                goto endloop; // I hate this too don't worry
                            }
                        }
                    }
                }
            }
            endloop: ;
        }
    }
 
    return element_to_element;
}

auto SEM::Meshes::Mesh2D_t::build_faces(size_t n_elements_domain, size_t n_nodes, int initial_N, const std::vector<Element2D_t>& elements) -> std::tuple<std::vector<Face2D_t>, std::vector<std::vector<size_t>>, std::vector<std::array<size_t, 4>>> {
    size_t total_edges = 0;
    for (const auto& element: elements) {
        total_edges += element.nodes_.size();
    }

    std::vector<Face2D_t> faces;
    faces.reserve(total_edges/2); // This is not exact

    std::vector<std::vector<size_t>> node_to_face(n_nodes);
    std::vector<std::array<size_t, 4>> element_to_face(elements.size());

    for (size_t i = 0; i < n_elements_domain; ++i) {
        for (size_t j = 0; j < elements[i].nodes_.size(); ++j) {
            const std::array<size_t, 2> nodes{elements[i].nodes_[j], (j + 1 < elements[i].nodes_.size()) ? elements[i].nodes_[j + 1] : elements[i].nodes_[0]};

            bool found = false;
            for (auto face_index: node_to_face[nodes[0]]) {
                if ((faces[face_index].nodes_[0] == nodes[1]) && (faces[face_index].nodes_[1] == nodes[0])) {
                    found = true;
                    faces[face_index].elements_[1] = i;
                    faces[face_index].elements_side_[1] = j;
                    element_to_face[i][j] = face_index;
                    break;
                }
            }

            if (!found) {
                element_to_face[i][j] = faces.size();
                node_to_face[nodes[0]].push_back(faces.size());
                if (nodes[1] != nodes[0]) {
                    node_to_face[nodes[1]].push_back(faces.size());
                }
                faces.emplace_back();
                faces.back().N_ = initial_N;
                faces.back().nodes_ = {nodes[0], nodes[1]};
                faces.back().elements_ = {i, static_cast<size_t>(-1)};
                faces.back().elements_side_ = {j, static_cast<size_t>(-1)};
            }
        }
    }

    for (size_t i = n_elements_domain; i < elements.size(); ++i) {
        const std::array<size_t, 2> nodes{elements[i].nodes_[0], elements[i].nodes_[1]};

        for (auto face_index: node_to_face[nodes[0]]) {
            if ((faces[face_index].nodes_[0] == nodes[1]) && (faces[face_index].nodes_[1] == nodes[0])) {
                faces[face_index].elements_[1] = i;
                faces[face_index].elements_side_[1] = 0;
                element_to_face[i][0] = face_index;
                for (size_t j = 1; j < element_to_face[i].size(); ++j) {
                    element_to_face[i][j] = static_cast<size_t>(-1);
                }
                break;
            }
        }
    }

    // Faces have to be moved, or else this copies the vector, and the device (???) vector copy for face vectors is used, which bad allocs for some reason.
    // 1) Why doesn't this move the vector, as it would be if it was plain returned?
    // 2) Why is the device copy used, it shouldn't be able to be called from that's like the whole point.
    // 3) Why does it bad alloc, the copied face should have its size default-constructed to 0.
    return {std::move(faces), std::move(node_to_face), std::move(element_to_face)};
}

auto SEM::Meshes::Mesh2D_t::initial_conditions(const device_vector<deviceFloat>& polynomial_nodes) -> void {
    initial_conditions_2D<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, elements_.data(), nodes_.data(), polynomial_nodes.data());
}

auto SEM::Meshes::Mesh2D_t::print() const -> void {
    std::vector<Face2D_t> host_faces(faces_.size());
    std::vector<Element2D_t> host_elements(elements_.size());
    std::vector<Vec2<deviceFloat>> host_nodes(nodes_.size());
    std::vector<size_t> host_wall_boundaries(wall_boundaries_.size());
    std::vector<size_t> host_symmetry_boundaries(symmetry_boundaries_.size());
    std::vector<size_t> host_inflow_boundaries(inflow_boundaries_.size());
    std::vector<size_t> host_outflow_boundaries(outflow_boundaries_.size());
    std::vector<size_t> host_interfaces_origin(interfaces_origin_.size());
    std::vector<size_t> host_interfaces_origin_side(interfaces_origin_side_.size());
    std::vector<size_t> host_interfaces_destination(interfaces_destination_.size());
    
    faces_.copy_to(host_faces, stream_);
    elements_.copy_to(host_elements, stream_);
    nodes_.copy_to(host_nodes, stream_);
    wall_boundaries_.copy_to(host_wall_boundaries, stream_);
    symmetry_boundaries_.copy_to(host_symmetry_boundaries, stream_);
    inflow_boundaries_.copy_to(host_inflow_boundaries, stream_);
    outflow_boundaries_.copy_to(host_outflow_boundaries, stream_);
    interfaces_origin_.copy_to(host_interfaces_origin, stream_);
    interfaces_origin_side_.copy_to(host_interfaces_origin_side, stream_);
    interfaces_destination_.copy_to(host_interfaces_destination, stream_);
    cudaStreamSynchronize(stream_);
    
    std::cout << "N elements: " << n_elements_ << std::endl;
    std::cout << "N elements and ghosts: " << elements_.size() << std::endl;
    std::cout << "N faces: " << faces_.size() << std::endl;
    std::cout << "N nodes: " << nodes_.size() << std::endl;
    std::cout << "N wall boundaries: " << wall_boundaries_.size() << std::endl;
    std::cout << "N symmetry boundaries: " << symmetry_boundaries_.size() << std::endl;
    std::cout << "N inflow boundaries: " << inflow_boundaries_.size() << std::endl;
    std::cout << "N outflow boundaries: " << outflow_boundaries_.size() << std::endl;
    std::cout << "N interfaces: " << interfaces_origin_.size() << std::endl;
    std::cout << "Initial N: " << initial_N_ << std::endl;

    std::cout << std::endl <<  "Connectivity" << std::endl;
    std::cout << '\t' <<  "Nodes:" << std::endl;
    for (size_t i = 0; i < host_nodes.size(); ++i) {
        std::cout << '\t' << '\t' << "node " << i << " : " << host_nodes[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Element nodes:" << std::endl;
    for (size_t i = 0; i < host_elements.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << i << " : ";
        for (auto node_index : host_elements[i].nodes_) {
            std::cout << node_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face nodes:" << std::endl;
    for (size_t i = 0; i < host_faces.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << i << " : ";
        for (auto node_index : host_faces[i].nodes_) {
            std::cout << node_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face elements:" << std::endl;
    for (size_t i = 0; i < host_faces.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << i << " : ";
        for (auto element_index : host_faces[i].elements_) {
            std::cout << element_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face elements side:" << std::endl;
    for (size_t i = 0; i < host_faces.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << i << " : ";
        for (auto side_index : host_faces[i].elements_side_) {
            std::cout << side_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl <<  "Geometry" << std::endl;
    std::cout << '\t' <<  "Element min length:" << std::endl;
    for (size_t i = 0; i < host_elements.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << i << " : " << host_elements[i].delta_xy_min_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Element N:" << std::endl;
    for (size_t i = 0; i < host_elements.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << i << " : " << host_elements[i].N_ << std::endl;
    }
    
    std::cout << std::endl << '\t' <<  "Face N:" << std::endl;
    for (size_t i = 0; i < host_faces.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << i << " : " << host_faces[i].N_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face length:" << std::endl;
    for (size_t i = 0; i < host_faces.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << i << " : " << host_faces[i].length_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face normal:" << std::endl;
    for (size_t i = 0; i < host_faces.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << i << " : " << host_faces[i].normal_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face tangent:" << std::endl;
    for (size_t i = 0; i < host_faces.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << i << " : " << host_faces[i].tangent_ << std::endl;
    }

    std::cout << std::endl <<  "Interfaces" << std::endl;
    std::cout << '\t' <<  "Interface destination, origin and origin side:" << std::endl;
    for (size_t i = 0; i < host_interfaces_origin.size(); ++i) {
        std::cout << '\t' << '\t' << "interface " << i << " : " << host_interfaces_destination[i] << " " << host_interfaces_origin[i] << " " << host_interfaces_origin_side[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Wall boundaries:" << std::endl;
    for (size_t i = 0; i < host_wall_boundaries.size(); ++i) {
        std::cout << '\t' << '\t' << "wall " << i << " : " << host_wall_boundaries[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Symmetry boundaries:" << std::endl;
    for (size_t i = 0; i < host_symmetry_boundaries.size(); ++i) {
        std::cout << '\t' << '\t' << "symmetry " << i << " : " << host_symmetry_boundaries[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Inflow boundaries:" << std::endl;
    for (size_t i = 0; i < host_inflow_boundaries.size(); ++i) {
        std::cout << '\t' << '\t' << "inflow " << i << " : " << host_inflow_boundaries[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Outflow boundaries:" << std::endl;
    for (size_t i = 0; i < host_outflow_boundaries.size(); ++i) {
        std::cout << '\t' << '\t' << "outflow " << i << " : " << host_outflow_boundaries[i] << std::endl;
    }

    std::cout << std::endl;
}

auto SEM::Meshes::Mesh2D_t::write_data(deviceFloat time, const device_vector<deviceFloat>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void {
    SEM::Meshes::get_solution<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, n_interpolation_points_, elements_.data(), nodes_.data(), interpolation_matrices.data(), x_output_device_.data(), y_output_device_.data(), p_output_device_.data(), u_output_device_.data(), v_output_device_.data());
    
    x_output_device_.copy_to(x_output_host_, stream_);
    y_output_device_.copy_to(y_output_host_, stream_);
    p_output_device_.copy_to(p_output_host_, stream_);
    u_output_device_.copy_to(u_output_host_, stream_);
    v_output_device_.copy_to(v_output_host_, stream_);
    cudaStreamSynchronize(stream_);

    data_writer.write_data(n_interpolation_points_, n_elements_, time, x_output_host_, y_output_host_, p_output_host_, u_output_host_, v_output_host_);
}

auto SEM::Meshes::Mesh2D_t::write_complete_data(deviceFloat time, const device_vector<deviceFloat>& interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void {
    device_vector<deviceFloat> dp_dt(n_elements_ * n_interpolation_points_ * n_interpolation_points_, stream_);
    device_vector<deviceFloat> du_dt(n_elements_ * n_interpolation_points_ * n_interpolation_points_, stream_);
    device_vector<deviceFloat> dv_dt(n_elements_ * n_interpolation_points_ * n_interpolation_points_, stream_);
    device_vector<int> N(n_elements_, stream_);
    device_vector<deviceFloat> p_error(n_elements_, stream_);
    device_vector<deviceFloat> u_error(n_elements_, stream_);
    device_vector<deviceFloat> v_error(n_elements_, stream_);
    device_vector<deviceFloat> p_sigma(n_elements_, stream_);
    device_vector<deviceFloat> u_sigma(n_elements_, stream_);
    device_vector<deviceFloat> v_sigma(n_elements_, stream_);
    device_vector<int> refine(n_elements_, stream_);
    device_vector<int> coarsen(n_elements_, stream_);
    device_vector<int> split_level(n_elements_, stream_);

    SEM::Meshes::get_complete_solution<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, n_interpolation_points_, elements_.data(), nodes_.data(), interpolation_matrices.data(), x_output_device_.data(), y_output_device_.data(), p_output_device_.data(), u_output_device_.data(), v_output_device_.data(), N.data(), dp_dt.data(), du_dt.data(), dv_dt.data(), p_error.data(), u_error.data(), v_error.data(), p_sigma.data(), u_sigma.data(), v_sigma.data(), refine.data(), coarsen.data(), split_level.data());
    
    std::vector<deviceFloat> dp_dt_host(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<deviceFloat> du_dt_host(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<deviceFloat> dv_dt_host(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<int> N_host(n_elements_);
    std::vector<deviceFloat> p_error_host(n_elements_);
    std::vector<deviceFloat> u_error_host(n_elements_);
    std::vector<deviceFloat> v_error_host(n_elements_);
    std::vector<deviceFloat> p_sigma_host(n_elements_);
    std::vector<deviceFloat> u_sigma_host(n_elements_);
    std::vector<deviceFloat> v_sigma_host(n_elements_);
    std::vector<int> refine_host(n_elements_);
    std::vector<int> coarsen_host(n_elements_);
    std::vector<int> split_level_host(n_elements_);

    x_output_device_.copy_to(x_output_host_, stream_);
    y_output_device_.copy_to(y_output_host_, stream_);
    p_output_device_.copy_to(p_output_host_, stream_);
    u_output_device_.copy_to(u_output_host_, stream_);
    v_output_device_.copy_to(v_output_host_, stream_);
    dp_dt.copy_to(dp_dt_host, stream_);
    du_dt.copy_to(du_dt_host, stream_);
    dv_dt.copy_to(dv_dt_host, stream_);
    N.copy_to(N_host, stream_);
    p_error.copy_to(p_error_host, stream_);
    u_error.copy_to(u_error_host, stream_);
    v_error.copy_to(v_error_host, stream_);
    p_sigma.copy_to(p_sigma_host, stream_);
    u_sigma.copy_to(u_sigma_host, stream_);
    v_sigma.copy_to(v_sigma_host, stream_);
    refine.copy_to(refine_host, stream_);
    coarsen.copy_to(coarsen_host, stream_);
    split_level.copy_to(split_level_host, stream_);
    cudaStreamSynchronize(stream_);

    data_writer.write_complete_data(n_interpolation_points_, n_elements_, time, x_output_host_, y_output_host_, p_output_host_, u_output_host_, v_output_host_, N_host, dp_dt_host, du_dt_host, dv_dt_host, p_error_host, u_error_host, v_error_host, p_sigma_host, u_sigma_host, v_sigma_host, refine_host, coarsen_host, split_level_host);

    dp_dt.clear(stream_);
    du_dt.clear(stream_);
    dv_dt.clear(stream_);
    N.clear(stream_);
    p_error.clear(stream_);
    u_error.clear(stream_);
    v_error.clear(stream_);
    p_sigma.clear(stream_);
    u_sigma.clear(stream_);
    v_sigma.clear(stream_);
    refine.clear(stream_);
    coarsen.clear(stream_);
    split_level.clear(stream_);
}

__host__ __device__
auto SEM::Meshes::Mesh2D_t::g(Vec2<deviceFloat> xy, deviceFloat t) -> std::array<deviceFloat, 3> {    
    const deviceFloat p = std::exp(-(SEM::Constants::k.x() * (xy.x() - SEM::Constants::xy0.x()) + SEM::Constants::k.y() * (xy.y() - SEM::Constants::xy0.y()) - SEM::Constants::c * t) * (SEM::Constants::k.x() * (xy.x() - SEM::Constants::xy0.x()) + SEM::Constants::k.y() * (xy.y() - SEM::Constants::xy0.y()) - SEM::Constants::c * t) / (SEM::Constants::d * SEM::Constants::d));

    return {p,
            p * SEM::Constants::k.x() / SEM::Constants::c,
            p * SEM::Constants::k.y() / SEM::Constants::c};
}

auto SEM::Meshes::Mesh2D_t::adapt(int N_max, const device_vector<deviceFloat>& polynomial_nodes, const device_vector<deviceFloat>& barycentric_weights) -> void {
    SEM::Meshes::reduce_refine_2D<elements_blockSize_/2><<<elements_numBlocks_, elements_blockSize_/2, 0, stream_>>>(n_elements_, max_split_level_, elements_.data(), device_refine_array_.data());
    device_refine_array_.copy_to(host_refine_array_, stream_);
    cudaStreamSynchronize(stream_);

    size_t n_splitting_elements = 0;
    for (int i = 0; i < elements_numBlocks_; ++i) {
        host_refine_array_[i] = n_splitting_elements; // Current block offset
        n_splitting_elements += host_refine_array_[i];
    }

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    std::vector<size_t> splitting_elements_global(global_size);
    constexpr MPI_Datatype data_type = (sizeof(size_t) == sizeof(unsigned long long)) ? MPI_UNSIGNED_LONG_LONG : (sizeof(size_t) == sizeof(unsigned long)) ? MPI_UNSIGNED_LONG : MPI_UNSIGNED; // CHECK this is a bad way of doing this

    MPI_Allgather(&n_splitting_elements, 1, data_type, splitting_elements_global.data(), 1, data_type, MPI_COMM_WORLD);

    size_t n_splitting_elements_previous = 0;
    for (int i = 0; i < global_rank; ++i) {
        n_splitting_elements_previous += splitting_elements_global[i];
    }
    const size_t global_element_offset_current = global_element_offset_ + 3 * n_splitting_elements_previous;
    size_t n_splitting_elements_global = 0;
    for (int i = 0; i < global_size; ++i) {
        n_splitting_elements_global += splitting_elements_global[i];
    }
    n_elements_global_ += 3 * n_splitting_elements_global;
    const size_t global_element_offset_end_current = global_element_offset_current + n_elements_ + 3 * n_splitting_elements - 1;

    const size_t n_elements_per_process = (n_elements_global_ + global_size - 1)/global_size;
    global_element_offset_ = global_rank * n_elements_per_process;
    const size_t global_element_offset_end = std::min(global_element_offset_ + n_elements_per_process - 1, n_elements_global_ - 1);

    if ((n_splitting_elements == 0) && (global_element_offset_ == global_element_offset_current) && (global_element_offset_end == global_element_offset_end_current)) {
        SEM::Meshes::p_adapt<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, elements_.data(), N_max, nodes_.data(), polynomial_nodes.data(), barycentric_weights.data());

        // We need to adjust the boundaries in all cases, or check if of our neighbours have to change
        if (!wall_boundaries_.empty()) {
            SEM::Meshes::adjust_boundaries<<<wall_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(wall_boundaries_.size(), elements_.data(), wall_boundaries_.data(), faces_.data());
        }
        if (!symmetry_boundaries_.empty()) {
            SEM::Meshes::adjust_boundaries<<<symmetry_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(symmetry_boundaries_.size(), elements_.data(), symmetry_boundaries_.data(), faces_.data());
        }
        if (!inflow_boundaries_.empty()) {
            SEM::Meshes::adjust_boundaries<<<inflow_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(inflow_boundaries_.size(), elements_.data(), inflow_boundaries_.data(), faces_.data());
        }
        if (!outflow_boundaries_.empty()) {
            SEM::Meshes::adjust_boundaries<<<outflow_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(outflow_boundaries_.size(), elements_.data(), outflow_boundaries_.data(), faces_.data());
        }
        if (!interfaces_origin_.empty()) {
            SEM::Meshes::adjust_interfaces<<<interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(interfaces_origin_.size(), elements_.data(), interfaces_origin_.data(), interfaces_destination_.data());
        }
        if (!mpi_interfaces_origin_.empty()) {
            SEM::Meshes::get_MPI_interfaces_N<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(mpi_interfaces_origin_.size(), elements_.data(), mpi_interfaces_origin_.data(), device_interfaces_N_.data());

            device_interfaces_N_.copy_to(host_interfaces_N_, stream_);
            cudaStreamSynchronize(stream_);

            for (size_t i = 0; i < mpi_interfaces_size_.size(); ++i) {
                MPI_Isend(host_interfaces_N_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i], MPI_INT, mpi_interfaces_process_[i], 3 * global_size * global_size + global_size * global_rank + mpi_interfaces_process_[i], MPI_COMM_WORLD, &requests_N_[mpi_interfaces_size_.size() + i]);
                MPI_Irecv(host_receiving_interfaces_N_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i], MPI_INT, mpi_interfaces_process_[i], 3 * global_size * global_size + global_size * mpi_interfaces_process_[i] + global_rank, MPI_COMM_WORLD, &requests_N_[i]);
            }

            MPI_Waitall(mpi_interfaces_size_.size(), requests_N_.data(), statuses_N_.data());

            device_interfaces_N_.copy_from(host_receiving_interfaces_N_, stream_);

            SEM::Meshes::put_MPI_interfaces_N<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(mpi_interfaces_destination_.size(), elements_.data(), mpi_interfaces_destination_.data(), device_interfaces_N_.data());

            MPI_Waitall(mpi_interfaces_size_.size(), requests_N_.data() + mpi_interfaces_size_.size(), statuses_N_.data() + mpi_interfaces_size_.size());
        }

        SEM::Meshes::adjust_faces<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data(), elements_.data());

        return;
    }

    device_refine_array_.copy_from(host_refine_array_, stream_);

    device_vector<Element2D_t> new_elements(elements_.size() + 3 * n_splitting_elements, stream_);

    SEM::Meshes::find_nodes<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, elements_.data(), faces_.data(), nodes.data(), max_split_level_);
    
    SEM::Meshes::reduce_faces_refine_2D<faces_blockSize_/2><<<faces_numBlocks_, faces_blockSize_/2, 0, stream_>>>(n_faces_, max_split_level_, faces_.data(), elements_.data(), device_faces_refine_array_.data());
    device_faces_refine_array_.copy_to(host_faces_refine_array_, stream_);
    cudaStreamSynchronize(stream_);

    size_t n_splitting_faces = 0;
    for (int i = 0; i < faces_numBlocks_; ++i) {
        host_faces_refine_array_[i] = n_splitting_faces; // Current block offset
        n_splitting_faces += host_faces_refine_array_[i];
    }

    device_faces_refine_array_.copy_from(host_faces_refine_array_, stream_);

    device_vector<Vec2<deviceFloat>> new_nodes(nodes_.size() + n_splitting_elements + n_splitting_faces, stream_);
    device_vector<Face2D_t> new_faces(faces_.size() + 4 * n_splitting_elements + n_splitting_faces, stream_);

    cudaMemcpyAsync(new_nodes.data(), nodes_.data(), nodes_.size() * sizeof(Vec2<deviceFloat>), cudaMemcpyDeviceToDevice, stream_); // Apparently slower than using a kernel

    SEM::Meshes::hp_adapt<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, faces_.size(), nodes_.size(), n_splitting_elements, elements_.data(), new_elements.data(), faces_.data(), new_faces.data(), device_refine_array_.data(), device_faces_refine_array_.data(), max_split_level_, N_max, new_nodes.data(), polynomial_nodes.data(), barycentric_weights.data(), faces_blockSize_);
    
    SEM::Meshes::split_faces<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data(), new_faces.data(), new_nodes.data());
    






    if (!wall_boundaries_.empty()) {
        SEM::Meshes::rebuild_boundaries<<<wall_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(wall_boundaries_.size(), elements_.data(), elements_.data(), wall_boundaries_.data(), faces_.data());
    }
    if (!symmetry_boundaries_.empty()) {
        SEM::Meshes::rebuild_boundaries<<<symmetry_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(symmetry_boundaries_.size(), elements_.data(), elements_.data(), symmetry_boundaries_.data(), faces_.data());
    }
    if (!inflow_boundaries_.empty()) {
        SEM::Meshes::rebuild_boundaries<<<inflow_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(inflow_boundaries_.size(), elements_.data(), elements_.data(), inflow_boundaries_.data(), faces_.data());
    }
    if (!outflow_boundaries_.empty()) {
        SEM::Meshes::rebuild_boundaries<<<outflow_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(outflow_boundaries_.size(), elements_.data(), elements_.data(), outflow_boundaries_.data(), faces_.data());
    }
    if (!interfaces_origin_.empty()) {
        SEM::Meshes::rebuild_interfaces<<<interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(interfaces_origin_.size(), elements_.data(), new_elements.data(), interfaces_origin_.data(), interfaces_destination_.data());
    }
    if (!mpi_interfaces_origin_.empty()) {
        SEM::Meshes::get_MPI_interfaces_N<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(mpi_interfaces_origin_.size(), new_elements.data(), mpi_interfaces_origin_.data(), device_interfaces_N_.data());

        device_interfaces_N_.copy_to(host_interfaces_N_, stream_);
        cudaStreamSynchronize(stream_);

        for (size_t i = 0; i < mpi_interfaces_size_.size(); ++i) {
            MPI_Isend(host_interfaces_N_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i], MPI_INT, mpi_interfaces_process_[i], 3 * global_size * global_size + global_size * global_rank + mpi_interfaces_process_[i], MPI_COMM_WORLD, &requests_N_[mpi_interfaces_size_.size() + i]);
            MPI_Irecv(host_receiving_interfaces_N_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i], MPI_INT, mpi_interfaces_process_[i], 3 * global_size * global_size + global_size * mpi_interfaces_process_[i] + global_rank, MPI_COMM_WORLD, &requests_N_[i]);
        }

        MPI_Waitall(mpi_interfaces_size_.size(), requests_N_.data(), statuses_N_.data());

        device_interfaces_N_.copy_from(host_receiving_interfaces_N_, stream_);

        SEM::Meshes::put_MPI_interfaces_N_and_rebuild<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(mpi_interfaces_destination_.size(), elements_.data(), new_elements.data(), mpi_interfaces_destination_.data(), device_interfaces_N_.data());

        MPI_Waitall(mpi_interfaces_size_.size(), requests_N_.data() + mpi_interfaces_size_.size(), statuses_N_.data() + mpi_interfaces_size_.size());
    }

    elements_ = std::move(new_elements);
    faces_ = std::move(new_faces);
    nodes_ = std::move(new_nodes);



    SEM::Meshes::adjust_faces<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data(), elements_.data());
}

auto SEM::Meshes::Mesh2D_t::boundary_conditions(deviceFloat t, const device_vector<deviceFloat>& polynomial_nodes, const device_vector<deviceFloat>& weights, const device_vector<deviceFloat>& barycentric_weights) -> void {
    // Boundary conditions
    if (!wall_boundaries_.empty()) {
        SEM::Meshes::compute_wall_boundaries<<<wall_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(wall_boundaries_.size(), elements_.data(), wall_boundaries_.data(), faces_.data(), polynomial_nodes.data(), weights.data(), barycentric_weights.data());
    }
    if (!symmetry_boundaries_.empty()) {
        SEM::Meshes::compute_symmetry_boundaries<<<symmetry_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(symmetry_boundaries_.size(), elements_.data(), symmetry_boundaries_.data(), faces_.data(), polynomial_nodes.data(), weights.data(), barycentric_weights.data());
    }
    if (!inflow_boundaries_.empty()) {
        SEM::Meshes::compute_inflow_boundaries<<<inflow_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(inflow_boundaries_.size(), elements_.data(), inflow_boundaries_.data(), faces_.data(), t, nodes_.data(), polynomial_nodes.data());
    }
    if (!outflow_boundaries_.empty()) {
        SEM::Meshes::compute_outflow_boundaries<<<outflow_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(outflow_boundaries_.size(), elements_.data(), outflow_boundaries_.data(), faces_.data(), polynomial_nodes.data(), weights.data(), barycentric_weights.data());
    }

    // Interfaces
    if (!interfaces_origin_.empty()) {
        SEM::Meshes::local_interfaces<<<interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(interfaces_origin_.size(), elements_.data(), interfaces_origin_.data(), interfaces_origin_side_.data(), interfaces_destination_.data());
    }

    if (!mpi_interfaces_origin_.empty()) {
        int global_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
        int global_size;
        MPI_Comm_size(MPI_COMM_WORLD, &global_size);

        SEM::Meshes::get_MPI_interfaces<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(mpi_interfaces_origin_.size(), elements_.data(), mpi_interfaces_origin_.data(), mpi_interfaces_origin_side_.data(), maximum_N_, device_interfaces_p_.data(), device_interfaces_u_.data(), device_interfaces_v_.data());

        device_interfaces_p_.copy_to(host_interfaces_p_, stream_);
        device_interfaces_u_.copy_to(host_interfaces_u_, stream_);
        device_interfaces_v_.copy_to(host_interfaces_v_, stream_);
        cudaStreamSynchronize(stream_);
        
        constexpr MPI_Datatype data_type = (sizeof(deviceFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
        for (size_t i = 0; i < mpi_interfaces_size_.size(); ++i) {
            MPI_Isend(host_interfaces_p_.data() + mpi_interfaces_offset_[i] * (maximum_N_ + 1), mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * global_rank + mpi_interfaces_process_[i]), MPI_COMM_WORLD, &requests_[3 * (mpi_interfaces_size_.size() + i)]);
            MPI_Irecv(host_receiving_interfaces_p_.data() + mpi_interfaces_offset_[i] * (maximum_N_ + 1), mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * mpi_interfaces_process_[i] + global_rank), MPI_COMM_WORLD, &requests_[3 * i]);

            MPI_Isend(host_interfaces_u_.data() + mpi_interfaces_offset_[i] * (maximum_N_ + 1), mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * global_rank + mpi_interfaces_process_[i]) + 1, MPI_COMM_WORLD, &requests_[3 * (mpi_interfaces_size_.size() + i) + 1]);
            MPI_Irecv(host_receiving_interfaces_u_.data() + mpi_interfaces_offset_[i] * (maximum_N_ + 1), mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * mpi_interfaces_process_[i] + global_rank) + 1, MPI_COMM_WORLD, &requests_[3 * i + 1]);

            MPI_Isend(host_interfaces_v_.data() + mpi_interfaces_offset_[i] * (maximum_N_ + 1), mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * global_rank + mpi_interfaces_process_[i]) + 2, MPI_COMM_WORLD, &requests_[3 * (mpi_interfaces_size_.size() + i) + 2]);
            MPI_Irecv(host_receiving_interfaces_v_.data() + mpi_interfaces_offset_[i] * (maximum_N_ + 1), mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * mpi_interfaces_process_[i] + global_rank) + 2, MPI_COMM_WORLD, &requests_[3 * i + 2]);
        }

        MPI_Waitall(3 * mpi_interfaces_size_.size(), requests_.data(), statuses_.data());

        device_interfaces_p_.copy_from(host_receiving_interfaces_p_, stream_);
        device_interfaces_u_.copy_from(host_receiving_interfaces_u_, stream_);
        device_interfaces_v_.copy_from(host_receiving_interfaces_v_, stream_);

        SEM::Meshes::put_MPI_interfaces<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(mpi_interfaces_destination_.size(), elements_.data(), mpi_interfaces_destination_.data(), maximum_N_, device_interfaces_p_.data(), device_interfaces_u_.data(), device_interfaces_v_.data());

        MPI_Waitall(3 * mpi_interfaces_size_.size(), requests_.data() + 3 * mpi_interfaces_size_.size(), statuses_.data() + 3 * mpi_interfaces_size_.size());
    }
}

// From cppreference.com
__device__
auto SEM::Meshes::Mesh2D_t::almost_equal(deviceFloat x, deviceFloat y) -> bool {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<deviceFloat>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<deviceFloat>::min();
}

auto SEM::Meshes::Mesh2D_t::interpolate_to_boundaries(const device_vector<deviceFloat>& lagrange_interpolant_left, const device_vector<deviceFloat>& lagrange_interpolant_right) -> void {
    SEM::Meshes::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, elements_.data(), lagrange_interpolant_left.data(), lagrange_interpolant_right.data());
}

auto SEM::Meshes::Mesh2D_t::project_to_faces(const device_vector<deviceFloat>& polynomial_nodes, const device_vector<deviceFloat>& barycentric_weights) -> void {
    SEM::Meshes::project_to_faces<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data(), elements_.data(), polynomial_nodes.data(), barycentric_weights.data());
}

auto SEM::Meshes::Mesh2D_t::project_to_elements(const device_vector<deviceFloat>& polynomial_nodes, const device_vector<deviceFloat>& weights, const device_vector<deviceFloat>& barycentric_weights) -> void {
    SEM::Meshes::project_to_elements<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, faces_.data(), elements_.data(), polynomial_nodes.data(), weights.data(), barycentric_weights.data());
}

template auto SEM::Meshes::Mesh2D_t::estimate_error<SEM::Polynomials::ChebyshevPolynomial_t>(const device_vector<deviceFloat>& polynomial_nodes, const device_vector<deviceFloat>& weights) -> void;
template auto SEM::Meshes::Mesh2D_t::estimate_error<SEM::Polynomials::LegendrePolynomial_t>(const device_vector<deviceFloat>& polynomial_nodes, const device_vector<deviceFloat>& weights) -> void;

template<typename Polynomial>
auto SEM::Meshes::Mesh2D_t::estimate_error<Polynomial>(const device_vector<deviceFloat>& polynomial_nodes, const device_vector<deviceFloat>& weights) -> void {
    SEM::Meshes::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(n_elements_, elements_.data(), tolerance_min_, tolerance_max_, polynomial_nodes.data(), weights.data());
}

__global__
auto SEM::Meshes::allocate_element_storage(size_t n_elements, Element2D_t* elements) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        elements[element_index].allocate_storage();
    }
}

__global__
auto SEM::Meshes::allocate_boundary_storage(size_t n_domain_elements, size_t n_total_elements, Element2D_t* elements) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index + n_domain_elements; element_index < n_total_elements; element_index += stride) {
        elements[element_index].allocate_boundary_storage();
    }
}

__global__
auto SEM::Meshes::compute_element_geometry(size_t n_elements, Element2D_t* elements, const Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        const std::array<Vec2<deviceFloat>, 4> points {nodes[elements[element_index].nodes_[0]],
                                                       nodes[elements[element_index].nodes_[1]],
                                                       nodes[elements[element_index].nodes_[2]],
                                                       nodes[elements[element_index].nodes_[3]]};
        elements[element_index].compute_geometry(points, polynomial_nodes);  
    }
}

__global__
auto SEM::Meshes::compute_boundary_geometry(size_t n_domain_elements, size_t n_total_elements, Element2D_t* elements, const Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index + n_domain_elements; element_index < n_total_elements; element_index += stride) {
        const std::array<Vec2<deviceFloat>, 4> points {nodes[elements[element_index].nodes_[0]],
                                                       nodes[elements[element_index].nodes_[1]],
                                                       nodes[elements[element_index].nodes_[2]],
                                                       nodes[elements[element_index].nodes_[3]]};
        elements[element_index].compute_boundary_geometry(points, polynomial_nodes);  
    }
}

__global__
auto SEM::Meshes::allocate_face_storage(size_t n_faces, Face2D_t* faces) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < n_faces; face_index += stride) {
        faces[face_index].allocate_storage();
    }
}

__global__
auto SEM::Meshes::fill_element_faces(size_t n_elements, Element2D_t* elements, const std::array<size_t, 4>* element_to_face) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        for (size_t j = 0; j < elements[element_index].faces_.size(); ++j) {
            elements[element_index].faces_[j][0] = element_to_face[element_index][j];
        }
    }
}

__global__
auto SEM::Meshes::compute_face_geometry(size_t n_faces, Face2D_t* faces, const Element2D_t* elements, const Vec2<deviceFloat>* nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < n_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];
        const std::array<Vec2<deviceFloat>, 2> face_nodes {nodes[face.nodes_[0]], nodes[face.nodes_[1]]};

        const size_t element_L_index = face.elements_[0];
        const size_t element_R_index = face.elements_[1];
        const size_t element_L_side = face.elements_side_[0];
        const size_t element_R_side = face.elements_side_[1];
        const size_t element_L_next_side = (element_L_side + 1 < elements[element_L_index].nodes_.size()) ? element_L_side + 1 : 0;
        const size_t element_R_next_side = (element_R_side + 1 < elements[element_R_index].nodes_.size()) ? element_R_side + 1 : 0;
        const Element2D_t& element_L = elements[element_L_index];
        const Element2D_t& element_R = elements[element_R_index];
        const std::array<std::array<Vec2<deviceFloat>, 2>, 2> elements_nodes {
            {nodes[element_L.nodes_[element_L_side]], nodes[element_L.nodes_[element_L_next_side]]},
            {nodes[element_R.nodes_[element_R_side]], nodes[element_R.nodes_[element_R_next_side]]}
        };

        faces[face_index].compute_geometry(elements, face_nodes, elements_nodes);
    }
}

__global__
auto SEM::Meshes::initial_conditions_2D(size_t n_elements, Element2D_t* elements, const Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        Element2D_t& element = elements[element_index];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;
        const std::array<Vec2<deviceFloat>, 4> points {nodes[element.nodes_[0]],
                                                       nodes[element.nodes_[1]],
                                                       nodes[element.nodes_[2]],
                                                       nodes[element.nodes_[3]]};
        
        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
                const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points);

                const std::array<deviceFloat, 3> state = SEM::Meshes::Mesh2D_t::g(global_coordinates, 0);
                element.p_[i * (element.N_ + 1) + j] = state[0];
                element.u_[i * (element.N_ + 1) + j] = state[1];
                element.v_[i * (element.N_ + 1) + j] = state[2];
            }
        }
    }
}

__global__
auto SEM::Meshes::get_solution(size_t n_elements, size_t n_interpolation_points, const Element2D_t* elements, const Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        const Element2D_t& element = elements[element_index];
        const size_t offset_interp_2D = element_index * n_interpolation_points * n_interpolation_points;
        const size_t offset_interp = element.N_ * (element.N_ + 1) * n_interpolation_points/2;

        const std::array<Vec2<deviceFloat>, 4> points {nodes[element.nodes_[0]],
                                                       nodes[element.nodes_[1]],
                                                       nodes[element.nodes_[2]],
                                                       nodes[element.nodes_[3]]};

        element.interpolate_solution(n_interpolation_points, points, interpolation_matrices + offset_interp, x + offset_interp_2D, y + offset_interp_2D, p + offset_interp_2D, u + offset_interp_2D, v + offset_interp_2D);
    }
}

__global__
auto SEM::Meshes::get_complete_solution(size_t n_elements, size_t n_interpolation_points, const Element2D_t* elements, const Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, int* N, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt, deviceFloat* p_error, deviceFloat* u_error, deviceFloat* v_error, deviceFloat* p_sigma, deviceFloat* u_sigma, deviceFloat* v_sigma, int* refine, int* coarsen, int* split_level) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        const Element2D_t& element = elements[element_index];
        const size_t offset_interp_2D = element_index * n_interpolation_points * n_interpolation_points;
        const size_t offset_interp = element.N_ * (element.N_ + 1) * n_interpolation_points/2;

        N[element_index] = element.N_;
        p_error[element_index] = element.p_error_;
        u_error[element_index] = element.u_error_;
        v_error[element_index] = element.v_error_;
        p_sigma[element_index] = element.p_sigma_;
        u_sigma[element_index] = element.u_sigma_;
        v_sigma[element_index] = element.v_sigma_;
        refine[element_index] = element.refine_;
        coarsen[element_index] = element.coarsen_;
        split_level[element_index] = element.split_level_;
        const std::array<Vec2<deviceFloat>, 4> points {nodes[element.nodes_[0]],
                                                       nodes[element.nodes_[1]],
                                                       nodes[element.nodes_[2]],
                                                       nodes[element.nodes_[3]]};

        element.interpolate_complete_solution(n_interpolation_points, points, interpolation_matrices + offset_interp, x + offset_interp_2D, y + offset_interp_2D, p + offset_interp_2D, u + offset_interp_2D, v + offset_interp_2D, dp_dt + offset_interp_2D, du_dt + offset_interp_2D, dv_dt + offset_interp_2D);
    }
}

template __global__ auto SEM::Meshes::estimate_error<SEM::Polynomials::ChebyshevPolynomial_t>(size_t n_elements, Element2D_t* elements, deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;
template __global__ auto SEM::Meshes::estimate_error<SEM::Polynomials::LegendrePolynomial_t>(size_t n_elements, Element2D_t* elements, deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

template<typename Polynomial>
__global__
auto SEM::Meshes::estimate_error<Polynomial>(size_t n_elements, Element2D_t* elements, deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        elements[element_index].estimate_error<Polynomial>(tolerance_min, tolerance_max, polynomial_nodes, weights);
    }
}

__global__
auto SEM::Meshes::interpolate_to_boundaries(size_t n_elements, Element2D_t* elements, const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        elements[element_index].interpolate_to_boundaries(lagrange_interpolant_minus, lagrange_interpolant_plus);
    }
}

__global__
auto SEM::Meshes::project_to_faces(size_t n_faces, Face2D_t* faces, const Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < n_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];

        // Getting element solution
        const Element2D_t& element_L = elements[face.elements_[0]];
        // Conforming
        if ((face.N_ == element_L.N_) 
                && (face.nodes_[0] == element_L.nodes_[face.elements_side_[0]]) 
                && (face.nodes_[1] == element_L.nodes_[(face.elements_side_[0] + 1) * (!(face.elements_side_[0] + 1 == (element_L.nodes_.size())))])) {
            for (int i = 0; i <= face.N_; ++i) {
                face.p_[0][i] = element_L.p_extrapolated_[face.elements_side_[0]][i];
                face.u_[0][i] = element_L.u_extrapolated_[face.elements_side_[0]][i];
                face.v_[0][i] = element_L.v_extrapolated_[face.elements_side_[0]][i];
            }
        }
        else { // We need to interpolate
            const size_t offset_1D = face.N_ * (face.N_ + 1) /2;
            const size_t offset_1D_other = element_L.N_ * (element_L.N_ + 1) /2;

            for (int i = 0; i <= face.N_; ++i) {
                const deviceFloat coordinate = (polynomial_nodes[offset_1D + i] + 1) * face.scale_[0] + 2 * face.offset_[0] - 1;

                deviceFloat p_numerator = 0.0;
                deviceFloat u_numerator = 0.0;
                deviceFloat v_numerator = 0.0;
                deviceFloat denominator = 0.0;

                for (int j = 0; j <= element_L.N_; ++j) {
                    if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D_other + j])) {
                        p_numerator = element_L.p_extrapolated_[face.elements_side_[0]][j];
                        u_numerator = element_L.u_extrapolated_[face.elements_side_[0]][j];
                        v_numerator = element_L.v_extrapolated_[face.elements_side_[0]][j];
                        denominator = 1.0;
                        break;
                    }

                    const deviceFloat t = barycentric_weights[offset_1D_other + j]/(coordinate - polynomial_nodes[offset_1D_other + j]);
                    p_numerator += t * element_L.p_extrapolated_[face.elements_side_[0]][j];
                    u_numerator += t * element_L.u_extrapolated_[face.elements_side_[0]][j];
                    v_numerator += t * element_L.v_extrapolated_[face.elements_side_[0]][j];
                    denominator += t;
                }
                face.p_[0][i] = p_numerator/denominator;
                face.u_[0][i] = u_numerator/denominator;
                face.v_[0][i] = v_numerator/denominator;
            }
        }

        const Element2D_t& element_R = elements[face.elements_[1]];
        // Conforming, but reversed
        if ((face.N_ == element_R.N_) 
                && (face.nodes_[1] == element_R.nodes_[face.elements_side_[1]]) 
                && (face.nodes_[0] == element_R.nodes_[(face.elements_side_[1] + 1) * (!(face.elements_side_[1] + 1 == (element_R.nodes_.size())))])) {
            for (int i = 0; i <= face.N_; ++i) {
                face.p_[1][face.N_ - i] = element_R.p_extrapolated_[face.elements_side_[1]][i];
                face.u_[1][face.N_ - i] = element_R.u_extrapolated_[face.elements_side_[1]][i];
                face.v_[1][face.N_ - i] = element_R.v_extrapolated_[face.elements_side_[1]][i];
            }
        }
        else { // We need to interpolate
            const size_t offset_1D = face.N_ * (face.N_ + 1) /2;
            const size_t offset_1D_other = element_R.N_ * (element_R.N_ + 1) /2;

            for (int i = 0; i <= face.N_; ++i) {
                const deviceFloat coordinate = (polynomial_nodes[offset_1D + face.N_ - i] + 1) * face.scale_[1] + 2 * face.offset_[1] - 1;

                deviceFloat p_numerator = 0.0;
                deviceFloat u_numerator = 0.0;
                deviceFloat v_numerator = 0.0;
                deviceFloat denominator = 0.0;

                for (int j = 0; j <= element_R.N_; ++j) {
                    if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D_other + j])) {
                        p_numerator = element_R.p_extrapolated_[face.elements_side_[1]][j];
                        u_numerator = element_R.u_extrapolated_[face.elements_side_[1]][j];
                        v_numerator = element_R.v_extrapolated_[face.elements_side_[1]][j];
                        denominator = 1.0;
                        break;
                    }

                    const deviceFloat t = barycentric_weights[offset_1D_other + j]/(coordinate - polynomial_nodes[offset_1D_other + j]);
                    p_numerator += t * element_R.p_extrapolated_[face.elements_side_[1]][j];
                    u_numerator += t * element_R.u_extrapolated_[face.elements_side_[1]][j];
                    v_numerator += t * element_R.v_extrapolated_[face.elements_side_[1]][j];
                    denominator += t;
                }
                face.p_[1][i] = p_numerator/denominator;
                face.u_[1][i] = u_numerator/denominator;
                face.v_[1][i] = v_numerator/denominator;
            }
        }
    }
}

__global__
auto SEM::Meshes::project_to_elements(size_t n_elements, const Face2D_t* faces, Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        Element2D_t& element = elements[element_index];

        for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
            // Conforming, forward
            if ((element.faces_[side_index].size() == 1)
                    && (faces[element.faces_[side_index][0]].N_ == element.N_)  
                    && (faces[element.faces_[side_index][0]].nodes_[0] == element.nodes_[side_index]) 
                    && (faces[element.faces_[side_index][0]].nodes_[1] == element.nodes_[(side_index + 1) * !(side_index + 1 == (element.faces_.size()))])) {

                const Face2D_t& face = faces[element.faces_[side_index][0]];
                for (int j = 0; j <= faces[element.faces_[side_index][0]].N_; ++j) {
                    element.p_flux_extrapolated_[side_index][j] = face.p_flux_[j] * element.scaling_factor_[side_index][j];
                    element.u_flux_extrapolated_[side_index][j] = face.u_flux_[j] * element.scaling_factor_[side_index][j];
                    element.v_flux_extrapolated_[side_index][j] = face.v_flux_[j] * element.scaling_factor_[side_index][j];
                }
            }
            // Conforming, backwards
            else if ((element.faces_[side_index].size() == 1)
                    && (faces[element.faces_[side_index][0]].N_ == element.N_) 
                    && (faces[element.faces_[side_index][0]].nodes_[1] == element.nodes_[side_index]) 
                    && (faces[element.faces_[side_index][0]].nodes_[0] == element.nodes_[(side_index + 1) * !(side_index + 1 == (element.faces_.size()))])) {

                const Face2D_t& face = faces[element.faces_[side_index][0]];
                for (int j = 0; j <= face.N_; ++j) {
                    element.p_flux_extrapolated_[side_index][face.N_ - j] = -face.p_flux_[j] * element.scaling_factor_[side_index][j];
                    element.u_flux_extrapolated_[side_index][face.N_ - j] = -face.u_flux_[j] * element.scaling_factor_[side_index][j];
                    element.v_flux_extrapolated_[side_index][face.N_ - j] = -face.v_flux_[j] * element.scaling_factor_[side_index][j];
                }
            }
            else { // We need to interpolate
                const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

                for (int i = 0; i <= element.N_; ++i) {
                    element.p_flux_extrapolated_[side_index][i] = 0.0;
                    element.u_flux_extrapolated_[side_index][i] = 0.0;
                    element.v_flux_extrapolated_[side_index][i] = 0.0;
                }

                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    const Face2D_t& face = faces[element.faces_[side_index][face_index]];
                    const size_t offset_1D_other = face.N_ * (face.N_ + 1) /2;

                    // Non-conforming, forward
                    if (element_index == face.elements_[0]) {
                        for (int j = 0; j <= face.N_; ++j) {
                            const deviceFloat coordinate = (polynomial_nodes[offset_1D_other + j] + 1) * face.scale_[0] + 2 * face.offset_[0] - 1;
                            bool found_row = false;
                            
                            for (int i = 0; i <= element.N_; ++i) {
                                if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                                    element.p_flux_extrapolated_[side_index][i] += weights[offset_1D_other + j]/weights[offset_1D + i] * face.p_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.u_flux_extrapolated_[side_index][i] += weights[offset_1D_other + j]/weights[offset_1D + i] * face.u_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.v_flux_extrapolated_[side_index][i] += weights[offset_1D_other + j]/weights[offset_1D + i] * face.v_flux_[j] * element.scaling_factor_[side_index][i];
                                    found_row = true;
                                    break;
                                }
                            }

                            if (!found_row) {
                                double s = 0.0;
                                for (int i = 0; i <= element.N_; ++i) {
                                    s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                                }
                                for (int i = 0; i <= element.N_; ++i) {
                                    const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                                    element.p_flux_extrapolated_[side_index][i] += T * weights[offset_1D_other + j]/weights[offset_1D + i] * face.p_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.u_flux_extrapolated_[side_index][i] += T * weights[offset_1D_other + j]/weights[offset_1D + i] * face.u_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.v_flux_extrapolated_[side_index][i] += T * weights[offset_1D_other + j]/weights[offset_1D + i] * face.v_flux_[j] * element.scaling_factor_[side_index][i];
                                }
                            }
                        }
                    }
                    // Non-conforming, backwards
                    else {
                        for (int j = 0; j <= face.N_; ++j) {
                            const deviceFloat coordinate = (polynomial_nodes[offset_1D_other + face.N_ - j] + 1) * face.scale_[1] + 2 * face.offset_[1] - 1;
                            bool found_row = false;
                            
                            for (int i = 0; i <= element.N_; ++i) {
                                if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                                    element.p_flux_extrapolated_[side_index][i] += -weights[offset_1D_other + j]/weights[offset_1D + i] * face.p_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.u_flux_extrapolated_[side_index][i] += -weights[offset_1D_other + j]/weights[offset_1D + i] * face.u_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.v_flux_extrapolated_[side_index][i] += -weights[offset_1D_other + j]/weights[offset_1D + i] * face.v_flux_[j] * element.scaling_factor_[side_index][i];
                                    found_row = true;
                                    break;
                                }
                            }

                            if (!found_row) {
                                double s = 0.0;
                                for (int i = 0; i <= element.N_; ++i) {
                                    s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                                }
                                for (int i = 0; i <= element.N_; ++i) {
                                    const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                                    element.p_flux_extrapolated_[side_index][i] += -T * weights[offset_1D_other + j]/weights[offset_1D + i] * face.p_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.u_flux_extrapolated_[side_index][i] += -T * weights[offset_1D_other + j]/weights[offset_1D + i] * face.u_flux_[j] * element.scaling_factor_[side_index][i];
                                    element.v_flux_extrapolated_[side_index][i] += -T * weights[offset_1D_other + j]/weights[offset_1D + i] * face.v_flux_[j] * element.scaling_factor_[side_index][i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__
auto SEM::Meshes::compute_wall_boundaries(size_t n_wall_boundaries, Element2D_t* elements, const size_t* wall_boundaries, const Face2D_t* faces, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t boundary_index = index; boundary_index < n_wall_boundaries; boundary_index += stride) {
        const size_t element_index = wall_boundaries[boundary_index];
        Element2D_t& element = elements[element_index];

        if (element.faces_[0].size() == 0) { // Only one neighbour
            const Face2D_t& face = faces[element.faces_[0][0]];
            const int face_side = face.elements_[0] == element_index;
            const Element2D_t& neighbour = elements[face.elements_[face_side]];
            const size_t neighbour_side = face.elements_side_[face_side];
            const Vec2<deviceFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<deviceFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};

            if (element.N_ == neighbour.N_) { // Conforming
                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<deviceFloat> neighbour_velocity {neighbour.u_extrapolated_[neighbour_side][neighbour.N_ - k], neighbour.v_extrapolated_[neighbour_side][neighbour.N_ - k]};
                    Vec2<deviceFloat> local_velocity {neighbour_velocity.dot(face.normal_), neighbour_velocity.dot(face.tangent_)};
                    local_velocity.x() = (2 * neighbour.p_extrapolated_[neighbour_side][neighbour.N_ - k] + SEM::Constants::c * local_velocity.x()) / SEM::Constants::c;
                
                    element.p_extrapolated_[0][k] = neighbour.p_extrapolated_[neighbour_side][neighbour.N_ - k];
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
            else {
                const size_t offset_1D = element.N_ * (element.N_ + 1) /2;
                const size_t offset_1D_neighbour = neighbour.N_ * (neighbour.N_ + 1) /2;

                for (int i = 0; i <= element.N_; ++i) {
                    element.p_extrapolated_[0][i] = 0.0;
                    element.u_extrapolated_[0][i] = 0.0;
                    element.v_extrapolated_[0][i] = 0.0;
                }

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const deviceFloat coordinate = (polynomial_nodes[offset_1D_neighbour + neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                            element.p_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }

                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<deviceFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                    Vec2<deviceFloat> local_velocity {neighbour_velocity.dot(face.normal_), neighbour_velocity.dot(face.tangent_)};
                    local_velocity.x() = (2 * element.p_extrapolated_[0][k] + SEM::Constants::c * local_velocity.x()) / SEM::Constants::c;
                
                    //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
        }
        else {
            const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

            for (int i = 0; i <= element.N_; ++i) {
                element.p_extrapolated_[0][i] = 0.0;
                element.u_extrapolated_[0][i] = 0.0;
                element.v_extrapolated_[0][i] = 0.0;
            }

            for (size_t face_index = 0; face_index < element.faces_[0].size(); ++face_index) {
                const Face2D_t& face = faces[element.faces_[0][face_index]];
                const int face_side = face.elements_[0] == element_index;
                const Element2D_t& neighbour = elements[face.elements_[face_side]];
                const size_t neighbour_side = face.elements_side_[face_side];
                const Vec2<deviceFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
                const Vec2<deviceFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};
                const size_t offset_1D_neighbour = neighbour.N_ * (neighbour.N_ + 1) /2;

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const deviceFloat coordinate = (polynomial_nodes[offset_1D_neighbour + neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                            element.p_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }

            const Face2D_t& face = faces[element.faces_[0][0]]; // CHECK this is kinda wrong, but we only use the normal and tangent so let's assume all the faces on a side have the same
            const Vec2<deviceFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<deviceFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};
            for (int k = 0; k <= element.N_; ++k) {
                const Vec2<deviceFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                Vec2<deviceFloat> local_velocity {neighbour_velocity.dot(face.normal_), neighbour_velocity.dot(face.tangent_)};
                local_velocity.x() = (2 * element.p_extrapolated_[0][k] + SEM::Constants::c * local_velocity.x()) / SEM::Constants::c;
            
                //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
            }
        }
    }
}

__global__
auto SEM::Meshes::compute_symmetry_boundaries(size_t n_symmetry_boundaries, Element2D_t* elements, const size_t* symmetry_boundaries, const Face2D_t* faces, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t boundary_index = index; boundary_index < n_symmetry_boundaries; boundary_index += stride) {
        const size_t element_index = symmetry_boundaries[boundary_index];
        Element2D_t& element = elements[element_index];

        if (element.faces_[0].size() == 0) { // Only one neighbour
            const Face2D_t& face = faces[element.faces_[0][0]];
            const int face_side = face.elements_[0] == element_index;
            const Element2D_t& neighbour = elements[face.elements_[face_side]];
            const size_t neighbour_side = face.elements_side_[face_side];
            const Vec2<deviceFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<deviceFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};

            if (element.N_ == neighbour.N_) { // Conforming
                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<deviceFloat> neighbour_velocity {neighbour.u_extrapolated_[neighbour_side][neighbour.N_ - k], neighbour.v_extrapolated_[neighbour_side][neighbour.N_ - k]};
                    const Vec2<deviceFloat> local_velocity {-(neighbour_velocity.dot(face.normal_)), neighbour_velocity.dot(face.tangent_)};
                
                    element.p_extrapolated_[0][k] = neighbour.p_extrapolated_[neighbour_side][neighbour.N_ - k];
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
            else {
                const size_t offset_1D = element.N_ * (element.N_ + 1) /2;
                const size_t offset_1D_neighbour = neighbour.N_ * (neighbour.N_ + 1) /2;

                for (int i = 0; i <= element.N_; ++i) {
                    element.p_extrapolated_[0][i] = 0.0;
                    element.u_extrapolated_[0][i] = 0.0;
                    element.v_extrapolated_[0][i] = 0.0;
                }

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const deviceFloat coordinate = (polynomial_nodes[offset_1D_neighbour + neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                            element.p_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }

                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<deviceFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                    const Vec2<deviceFloat> local_velocity {-(neighbour_velocity.dot(face.normal_)), neighbour_velocity.dot(face.tangent_)};
                
                    //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
        }
        else {
            const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

            for (int i = 0; i <= element.N_; ++i) {
                element.p_extrapolated_[0][i] = 0.0;
                element.u_extrapolated_[0][i] = 0.0;
                element.v_extrapolated_[0][i] = 0.0;
            }

            for (size_t face_index = 0; face_index < element.faces_[0].size(); ++face_index) {
                const Face2D_t& face = faces[element.faces_[0][face_index]];
                const int face_side = face.elements_[0] == element_index;
                const Element2D_t& neighbour = elements[face.elements_[face_side]];
                const size_t neighbour_side = face.elements_side_[face_side];
                const Vec2<deviceFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
                const Vec2<deviceFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};
                const size_t offset_1D_neighbour = neighbour.N_ * (neighbour.N_ + 1) /2;

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const deviceFloat coordinate = (polynomial_nodes[offset_1D_neighbour + neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                            element.p_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }

            const Face2D_t& face = faces[element.faces_[0][0]]; // CHECK this is kinda wrong, but we only use the normal and tangent so let's assume all the faces on a side have the same
            const Vec2<deviceFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<deviceFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};
            for (int k = 0; k <= element.N_; ++k) {
                const Vec2<deviceFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                const Vec2<deviceFloat> local_velocity {-(neighbour_velocity.dot(face.normal_)), neighbour_velocity.dot(face.tangent_)};
            
                //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
            }
        }
    }
}

__global__
auto SEM::Meshes::compute_inflow_boundaries(size_t n_inflow_boundaries, Element2D_t* elements, const size_t* inflow_boundaries, const Face2D_t* faces, deviceFloat t, const Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t boundary_index = index; boundary_index < n_inflow_boundaries; boundary_index += stride) {
        Element2D_t& element = elements[inflow_boundaries[boundary_index]];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;;
        const std::array<Vec2<deviceFloat>, 2> points{nodes[element.nodes_[0]], nodes[element.nodes_[1]]};

        for (int k = 0; k <= element.N_; ++k) {
            const deviceFloat interp = (polynomial_nodes[offset_1D + k] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = points[1] * interp + points[0] * (1 - interp);

            const std::array<deviceFloat, 3> state = SEM::Meshes::Mesh2D_t::g(global_coordinates, t);

            element.p_extrapolated_[0][k] = state[0];
            element.u_extrapolated_[0][k] = state[1];
            element.v_extrapolated_[0][k] = state[2];
        }
    }
}

__global__
auto SEM::Meshes::compute_outflow_boundaries(size_t n_outflow_boundaries, Element2D_t* elements, const size_t* outflow_boundaries, const Face2D_t* faces, const deviceFloat* polynomial_nodes, const deviceFloat* weights, const deviceFloat* barycentric_weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t boundary_index = index; boundary_index < n_outflow_boundaries; boundary_index += stride) {
        const size_t element_index = outflow_boundaries[boundary_index];
        Element2D_t& element = elements[element_index];

        if (element.faces_[0].size() == 0) { // Only one neighbour
            const Face2D_t& face = faces[element.faces_[0][0]];
            const int face_side = face.elements_[0] == element_index;
            const Element2D_t& neighbour = elements[face.elements_[face_side]];
            const size_t neighbour_side = face.elements_side_[face_side];

            if (element.N_ == neighbour.N_) { // Conforming
                for (int k = 0; k <= element.N_; ++k) {
                    element.p_extrapolated_[0][k] = neighbour.p_extrapolated_[neighbour_side][neighbour.N_ - k];
                    element.u_extrapolated_[0][k] = neighbour.u_extrapolated_[neighbour_side][neighbour.N_ - k];
                    element.v_extrapolated_[0][k] = neighbour.v_extrapolated_[neighbour_side][neighbour.N_ - k];
                }
            }
            else {
                const size_t offset_1D = element.N_ * (element.N_ + 1) /2;
                const size_t offset_1D_neighbour = neighbour.N_ * (neighbour.N_ + 1) /2;

                for (int i = 0; i <= element.N_; ++i) {
                    element.p_extrapolated_[0][i] = 0.0;
                    element.u_extrapolated_[0][i] = 0.0;
                    element.v_extrapolated_[0][i] = 0.0;
                }

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const deviceFloat coordinate = (polynomial_nodes[offset_1D_neighbour + neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                            element.p_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }
        }
        else {
            const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

            for (int i = 0; i <= element.N_; ++i) {
                element.p_extrapolated_[0][i] = 0.0;
                element.u_extrapolated_[0][i] = 0.0;
                element.v_extrapolated_[0][i] = 0.0;
            }

            for (size_t face_index = 0; face_index < element.faces_[0].size(); ++face_index) {
                const Face2D_t& face = faces[element.faces_[0][face_index]];
                const int face_side = face.elements_[0] == element_index;
                const Element2D_t& neighbour = elements[face.elements_[face_side]];
                const size_t neighbour_side = face.elements_side_[face_side];
                const Vec2<deviceFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
                const Vec2<deviceFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};
                const size_t offset_1D_neighbour = neighbour.N_ * (neighbour.N_ + 1) /2;

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const deviceFloat coordinate = (polynomial_nodes[offset_1D_neighbour + neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[offset_1D + i])) {
                            element.p_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[offset_1D + i]/(coordinate - polynomial_nodes[offset_1D + i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const deviceFloat T = barycentric_weights[offset_1D + i]/((coordinate - polynomial_nodes[offset_1D + i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[offset_1D_neighbour + j]/weights[offset_1D + i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }
        }
    }
}

__global__
auto SEM::Meshes::local_interfaces(size_t n_local_interfaces, Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_local_interfaces; interface_index += stride) {
        const Element2D_t& source_element = elements[local_interfaces_origin[interface_index]];
        Element2D_t& destination_element = elements[local_interfaces_destination[interface_index]];
        const size_t element_side = local_interfaces_origin_side[interface_index];

        for (int k = 0; k <= source_element.N_; ++k) {
            destination_element.p_extrapolated_[0][k] = source_element.p_extrapolated_[element_side][k];
            destination_element.u_extrapolated_[0][k] = source_element.u_extrapolated_[element_side][k];
            destination_element.v_extrapolated_[0][k] = source_element.v_extrapolated_[element_side][k];
        }
    }
}

__global__
auto SEM::Meshes::get_MPI_interfaces(size_t n_MPI_interface_elements, const Element2D_t* elements, const size_t* MPI_interfaces_origin, const size_t* MPI_interfaces_origin_side, int maximum_N, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_MPI_interface_elements; interface_index += stride) {
        const Element2D_t& source_element = elements[MPI_interfaces_origin[interface_index]];
        const size_t element_side = MPI_interfaces_origin_side[interface_index];
        const size_t boundary_offset = interface_index * (maximum_N + 1);

        for (int k = 0; k <= source_element.N_; ++k) {
            p[boundary_offset + k] = source_element.p_extrapolated_[element_side][k];
            u[boundary_offset + k] = source_element.u_extrapolated_[element_side][k];
            v[boundary_offset + k] = source_element.v_extrapolated_[element_side][k];
        }
    }
}

__global__
auto SEM::Meshes::get_MPI_interfaces_N(size_t n_MPI_interface_elements, const Element2D_t* elements, const size_t* MPI_interfaces_origin, int* N) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_MPI_interface_elements; interface_index += stride) {
        N[interface_index] = elements[MPI_interfaces_origin[interface_index]].N_;
    }
}

__global__
auto SEM::Meshes::put_MPI_interfaces(size_t n_MPI_interface_elements, Element2D_t* elements, const size_t* MPI_interfaces_destination, int maximum_N, const deviceFloat* p, const deviceFloat* u, const deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_MPI_interface_elements; interface_index += stride) {
        Element2D_t& destination_element = elements[MPI_interfaces_destination[interface_index]];
        const size_t boundary_offset = interface_index * (maximum_N + 1);

        for (int k = 0; k <= destination_element.N_; ++k) {
            destination_element.p_extrapolated_[0][k] = p[boundary_offset + k];
            destination_element.u_extrapolated_[0][k] = u[boundary_offset + k];
            destination_element.v_extrapolated_[0][k] = v[boundary_offset + k];
        }
    }
}

__global__
auto SEM::Meshes::put_MPI_interfaces_N(size_t n_MPI_interface_elements, Element2D_t* elements, const size_t* MPI_interfaces_destination, const int* N) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_MPI_interface_elements; interface_index += stride) {
        Element2D_t& destination_element = elements[MPI_interfaces_destination[interface_index]];

        if (destination_element.N_ != N[interface_index]) {
            destination_element.resize_boundary_storage(N[interface_index]);
        }
    }
}

__global__
auto SEM::Meshes::put_MPI_interfaces_N_and_rebuild(size_t n_MPI_interface_elements, Element2D_t* elements, Element2D_t* new_elements, const size_t* MPI_interfaces_destination, const int* N) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_MPI_interface_elements; interface_index += stride) {
        Element2D_t& destination_element = elements[MPI_interfaces_destination[interface_index]];

        if (destination_element.N_ != N[interface_index]) {
            destination_element.resize_boundary_storage(N[interface_index]);
        }

        new_elements[MPI_interfaces_destination[interface_index]] = std::move(elements[MPI_interfaces_destination[interface_index]]);
    }
}

__global__
auto SEM::Meshes::p_adapt(size_t n_elements, Element2D_t* elements, int N_max, const Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (size_t i = index; i < n_elements; i += stride) {
        if (elements[i].would_p_refine(N_max)) {
            Element2D_t new_element(elements[i].N_ + 2, elements[i].split_level_, elements[i].faces_, elements[i].nodes_);

            new_element.interpolate_from(elements[i], polynomial_nodes, barycentric_weights);

            const std::array<Vec2<deviceFloat>, 4> points {nodes[new_element.nodes_[0]],
                                                           nodes[new_element.nodes_[1]],
                                                           nodes[new_element.nodes_[2]],
                                                           nodes[new_element.nodes_[3]]};
            new_element.compute_geometry(points, polynomial_nodes); 

            elements[i] = std::move(new_element);
        }
    }
}

__global__
auto SEM::Meshes::hp_adapt(size_t n_elements, size_t n_faces, size_t n_nodes, size_t n_splitting_elements, Element2D_t* elements, Element2D_t* new_elements, Face2D_t* faces, Face2D_t* new_faces, const size_t* block_offsets, const size_t* faces_block_offsets, int max_split_level, int N_max, Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights, int faces_blockSize) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    
    for (size_t i = index; i < n_elements; i += stride) {
        Element2D_t& element = elements[i];

        size_t n_splitting_elements_before = block_offsets[block_id]
        for (size_t j = i - thread_id; j < i; ++j) {
            n_splitting_elements_before += elements[j].would_h_refine(max_split_level);
        }
        const size_t element_index = i + 3 * n_splitting_elements_before;

        // h refinement
        if (element.would_h_refine(max_split_level)) {
            const size_t new_node_index = n_nodes + n_splitting_elements_before;

            std::array<size_t, 4> new_node_indices {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)}; // CHECK this won't work with elements with more than 4 sides
            std::array<Vec2<deviceFloat>, 4> new_nodes {}; // CHECK this won't work with elements with more than 4 sides
            nodes[new_node_index] = Vec2<deviceFloat>{0};
            for (size_t side_index = 0; side_index < element.nodes_.size(); ++side_index) {
                nodes[new_node_index] += nodes[element.nodes_[side_index]];

                if (element.additional_nodes_[side_index]) {
                    const size_t face_index = element.faces_[side_index][0];
                    new_nodes[side_index] = (nodes[faces[face_index].nodes_[0]] + nodes[faces[face_index].nodes_[1]])/2;

                    const int face_block_id = face_index/faces_blockSize;
                    const int face_thread_id = face_index%faces_blockSize;

                    new_node_indices[side_index] = n_nodes + n_splitting_elements + faces_block_offsets[face_block_id];
                    for (size_t j = face_index - face_thread_id; j < face_index; ++j) {
                        new_node_indices[side_index] += faces[j].refine_;
                    }
                }
                else {
                    const std::array<SEM::Entities::Vec2<deviceFloat>, 2> side_nodes = {nodes[element.nodes_[side_index]], (side_index + 1 < element.faces_.size()) ? nodes[element.nodes_[side_index + 1]] : nodes[element.nodes_[0]]};
                    const SEM::Entities::Vec2<deviceFloat> new_node = (side_nodes[0] + side_nodes[1])/2;

                    for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                        const SEM::Entities::Face2D_t& face = faces[element.faces_[side_index][face_index]];
                        if (nodes[face.nodes_[0]].almost_equal(new_node)) {
                            new_node_indices[side_index] = face.nodes_[0];
                            new_nodes[side_index] = nodes[face.nodes_[0]];
                            break;
                        } 
                        else if (nodes[face.nodes_[1]].almost_equal(new_node)) {
                            new_node_indices[side_index] = face.nodes_[1];
                            new_nodes[side_index] = nodes[face.nodes_[1]];
                            break;
                        }
                    }
                }
            }
            nodes[new_node_index] /= element.nodes_.size();

            const std::array<Vec2<deviceFloat>, 4> element_nodes = {nodes[element.nodes_[0]], nodes[element.nodes_[1]], nodes[element.nodes_[2]], nodes[element.nodes_[3]]}; // CHECK this won't work with elements with more than 4 sides

            // CHECK follow Hilbert curve
            const size_t new_face_index = n_faces + 4 * n_splitting_elements_before;

            for (size_t side_index = 0; side_index < element.nodes_.size(); ++side_index) {
                const size_t previous_side_index = (side_index > 0) ? side_index - 1 : element.nodes_.size() - 1;
                const size_t next_side_index = (side_index + 1 < element.nodes_.size()) ? side_index + 1 : 0;
                const size_t opposite_side_index = (side_index + 2 < element.nodes_.size()) ? side_index + 2 : side_index + 2 - element.nodes_.size();

                const std::array<size_t, 2> side_element_indices {element_index + side_index, element_index + next_side_index};
                const std::array<size_t, 2> side_element_sides {next_side_index, previous_side_index};
                
                new_faces[new_face_index + side_index] = Face2D_t{element.N_, {new_node_indices[side_index], new_node_index}, side_element_indices, side_element_sides};
            
                std:array<cuda_vector<size_t>, 4> new_element_faces {device_vector<deviceFloat>{},  device_vector<deviceFloat>{}, device_vector<deviceFloat>{}, device_vector<deviceFloat>{}};
            
                new_element_faces[next_side_index] = device_vector<deviceFloat>{1};
                new_element_faces[opposite_side_index] = device_vector<deviceFloat>{1};

                new_element_faces[next_side_index][0] = side_index;
                new_element_faces[opposite_side_index][0] = previous_side_index;
                
                if (element.additional_nodes_[side_index]) {
                    new_element_faces[side_index] = device_vector<deviceFloat>{1};
    
                    const size_t face_index = element.faces_[side_index][0];
                    const int face_block_id = face_index/faces_blockSize;
                    const int face_thread_id = face_index%faces_blockSize;
    
                    size_t splitting_face_index = n_faces + 4 * n_splitting_elements + faces_block_offsets[face_block_id];
                    for (size_t j = face_index - face_thread_id; j < face_index; ++j) {
                        splitting_face_index += faces[j].refine_;
                    }
    
                    if (i == faces[face_index].elements_[0]) { // forward
                        new_element_faces[side_index][0] = face_index;
                    }
                    else { // backward
                        new_element_faces[side_index][0] = splitting_face_index
                    }
                }
                else {
                    size_t n_side_faces = 0;
                    const Vec2<deviceFloat> new_node = (nodes[element.nodes_[side_index]] + nodes[element.nodes_[next_side_index]])/2;
    
                    const Vec2<deviceFloat>, 2> AB = new_node - nodes[element.nodes_[side_index]];
    
                    const deviceFloat AB_dot_inv  = 1/AB.dot(AB);
    
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            const Vec2<deviceFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<deviceFloat> AD = face_new_node - nodes[element.nodes_[side_index]];
                            const Vec2<deviceFloat> AE = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const deviceFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && E_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && E_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                        else {
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<deviceFloat> AD = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                    }
    
                    new_element_faces[side_index] = device_vector<deviceFloat>{n_side_faces};
    
                    size_t new_element_side_face_index 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            const int face_block_id = face_index/faces_blockSize;
                            const int face_thread_id = face_index%faces_blockSize;
    
                            size_t splitting_face_index = n_faces + 4 * n_splitting_elements + faces_block_offsets[face_block_id];
                            for (size_t j = face_index - face_thread_id; j < face_index; ++j) {
                                splitting_face_index += faces[j].refine_;
                            }
                            const Vec2<deviceFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<deviceFloat> AD = face_new_node - nodes[element.nodes_[side_index]];
                            const Vec2<deviceFloat> AE = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const deviceFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                    new_element_faces[side_index][new_element_side_face_index] = face_index;
                                ++new_element_side_face_index;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && E_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && E_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                    new_element_faces[side_index][new_element_side_face_index] = splitting_face_index;
                                ++new_element_side_face_index;
                            }
                        }
                        else {
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<deviceFloat> AD = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                    new_element_faces[side_index][new_element_side_face_index] = face_index;
                                ++new_element_side_face_index;
                            }
                        }
                    }
                }

                if (element.additional_nodes_[previous_side_index]) {
                    new_element_faces[previous_side_index] = device_vector<deviceFloat>{1};
    
                    const size_t face_index = element.faces_[previous_side_index][0];
                    const int face_block_id = face_index/faces_blockSize;
                    const int face_thread_id = face_index%faces_blockSize;
    
                    size_t splitting_face_index = n_faces + 4 * n_splitting_elements + faces_block_offsets[face_block_id];
                    for (size_t j = face_index - face_thread_id; j < face_index; ++j) {
                        splitting_face_index += faces[j].refine_;
                    }
    
                    if (i == faces[face_index].elements_[0]) { // forward
                        new_element_faces[previous_side_index][0] = splitting_face_index;
                    }
                    else { // backward
                        new_element_faces[previous_side_index][0] = face_index
                    }
                }
                else {
                    size_t n_side_faces = 0;
                    const Vec2<deviceFloat> new_node = (nodes[element.nodes_[previous_side_index]] + nodes[element.nodes_[side_index]])/2;
    
                    const Vec2<deviceFloat>, 2> AB = nodes[element.nodes_[side_index]] - new_node;
    
                    const deviceFloat AB_dot_inv  = 1/AB.dot(AB);
    
                    for (size_t side_face_index = 0; side_face_index < element.faces_[previous_side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[previous_side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            const Vec2<deviceFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<deviceFloat> AD = face_new_node - new_node;
                            const Vec2<deviceFloat> AE = nodes[face.nodes_[1]] - new_node;
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const deviceFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && E_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && E_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                        else {
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<deviceFloat> AD = nodes[face.nodes_[1]] - new_node;
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                    }
    
                    new_element_faces[previous_side_index] = device_vector<deviceFloat>{n_side_faces};
    
                    size_t new_element_side_face_index 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[previous_side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[previous_side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            const int face_block_id = face_index/faces_blockSize;
                            const int face_thread_id = face_index%faces_blockSize;
    
                            size_t splitting_face_index = n_faces + 4 * n_splitting_elements + faces_block_offsets[face_block_id];
                            for (size_t j = face_index - face_thread_id; j < face_index; ++j) {
                                splitting_face_index += faces[j].refine_;
                            }
                            const Vec2<deviceFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<deviceFloat> AD = face_new_node - new_node;
                            const Vec2<deviceFloat> AE = nodes[face.nodes_[1]] - new_node;
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const deviceFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                new_element_faces[previous_side_index][new_element_side_face_index] = face_index;
                                ++new_element_side_face_index;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && E_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && E_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                new_element_faces[previous_side_index][new_element_side_face_index] = splitting_face_index;
                                ++new_element_side_face_index;
                            }
                        }
                        else {
                            const Vec2<deviceFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<deviceFloat> AD = nodes[face.nodes_[1]] - new_node;
    
                            const deviceFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const deviceFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && C_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                                && D_proj + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                                && D_proj <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {
        
                                    new_element_faces[previous_side_index][new_element_side_face_index] = face_index;
                                ++new_element_side_face_index;
                            }
                        }
                    }
                }

                std::array<size_t, 4> new_element_node_indices {};
                new_element_node_indices[side_index] = element.nodes_[side_index];
                new_element_node_indices[next_side_index] = new_node_indices[side_index];
                new_element_node_indices[opposite_side_index] = new_node_index;
                new_element_node_indices[previous_side_index] = new_node_indices[previous_side_index];

                new_elements[element_index + side_index] = Element2D_t{element.N_, element.split_level_ + 1, new_element_faces, new_element_node_indices};

                const std::array<Vec2<deviceFloat>, 4> new_element_nodes {};
                new_element_nodes[side_index] = nodes[element.nodes_[side_index]];
                new_element_nodes[next_side_index] = new_nodes[side_index];
                new_element_nodes[opposite_side_index] = nodes[new_node_index];
                new_element_nodes[previous_side_index] = new_nodes[previous_side_index];

                new_elements[element_index + side_index].interpolate_from(new_element_nodes, element_nodes, element, polynomial_nodes, barycentric_weights);
                new_elements[element_index + side_index].compute_geometry(new_element_nodes, polynomial_nodes);
            }

            for (size_t side_index = 0; side_index < element.nodes_.size(); ++side_index) {
                const size_t next_side_index = (side_index + 1 < element.nodes_.size()) ? side_index + 1 : 0;
                const Vec2<deviceFloat> new_node = (nodes[element.nodes_[side_index]] + nodes[element.nodes_[next_side_index]])/2;
                const std::array<std::array<Vec2<deviceFloat>, 2>, 2> elements_nodes {
                    {new_node, nodes[new_node_index]}, // Not the same new node... this is confusing
                    {nodes[new_node_index], new_node}
                };

                new_faces[new_face_index + side_index].compute_geometry(new_elements, {new_node, nodes[new_node_index]}, elements_nodes);
            }
        }
        // p refinement
        else if (element.would_p_refine(N_max)) {
            new_elements[element_index] = Element2D_t{element.N_ + 2, element.split_level_, element.faces_, element.nodes_};

            // Adjusting faces for splitting elements
            for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                size_t side_n_splitting_faces = 0;
                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    side_n_splitting_faces += faces[element.faces_[side_index][face_index]].refine_;
                }

                if (side_n_splitting_faces > 0) {
                    cuda_vector<deviceFloat> side_new_faces(element.faces_[side_index].size() + side_n_splitting_faces);

                    size_t side_new_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        if (faces[face_index].refine_) {
                            side_new_faces[side_new_face_index] = face_index;

                            const int face_block_id = face_index/faces_blockSize;
                            const int face_thread_id = face_index%faces_blockSize;

                            size_t new_face_index = n_faces + 4 * n_splitting_elements + faces_block_offsets[face_block_id];
                            for (size_t j = face_index - face_thread_id; j < face_index; ++j) {
                                new_face_index += faces[j].refine_;
                            }

                            side_new_faces[side_new_face_index + 1] = new_face_index;

                            side_new_face_index += 2;
                        }
                        else {
                            side_new_faces[side_new_face_index] = face_index;
                            ++side_new_face_index;
                        }
                        side_n_splitting_faces += faces[element.faces_[side_index][face_index]].refine_;
                    }

                    new_elements[element_index].faces_[side_index] = std::move(side_new_faces);
                }
            }

            new_elements[element_index].interpolate_from(element, polynomial_nodes, barycentric_weights);

            const std::array<Vec2<deviceFloat>, 4> points {nodes[new_elements[element_index].nodes_[0]],
                                                           nodes[new_elements[element_index].nodes_[1]],
                                                           nodes[new_elements[element_index].nodes_[2]],
                                                           nodes[new_elements[element_index].nodes_[3]]};
            new_elements[element_index].compute_geometry(points, polynomial_nodes); 
        }
        // move
        else {
            // Adjusting faces for splitting elements
            for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                size_t side_n_splitting_faces = 0;
                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    side_n_splitting_faces += faces[element.faces_[side_index][face_index]].refine_;
                }

                if (side_n_splitting_faces > 0) {
                    cuda_vector<deviceFloat> side_new_faces(element.faces_[side_index].size() + side_n_splitting_faces);

                    size_t side_new_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        if (faces[face_index].refine_) {
                            side_new_faces[side_new_face_index] = face_index;

                            const int face_block_id = face_index/faces_blockSize;
                            const int face_thread_id = face_index%faces_blockSize;

                            size_t new_face_index = n_faces + 4 * n_splitting_elements + faces_block_offsets[face_block_id];
                            for (size_t j = face_index - face_thread_id; j < face_index; ++j) {
                                new_face_index += faces[j].refine_;
                            }

                            side_new_faces[side_new_face_index + 1] = new_face_index;

                            side_new_face_index += 2;
                        }
                        else {
                            side_new_faces[side_new_face_index] = face_index;
                            ++side_new_face_index;
                        }
                        side_n_splitting_faces += faces[element.faces_[side_index][face_index]].refine_;
                    }

                    element.faces_[side_index] = std::move(side_new_faces);
                }
            }

            new_elements[element_index] = std::move(element);
        }
    }
}

__global__
auto SEM::Meshes::split_faces(size_t n_faces, size_t n_nodes, size_t n_splitting_elements, Face2D_t* faces, Face2D_t* new_faces, const Element2D_t* elements, const Vec2<deviceFloat>* nodes, const size_t* faces_block_offsets, int max_split_level, int elements_blockSize) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;

    for (size_t face_index = index; face_index < n_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];
        const size_t element_L_index = face.elements_[0];
        const size_t element_R_index = face.elements_[1];
        const size_t element_L_side_index = face.elements_side_[0];
        const size_t element_R_side_index = face.elements_side_[1];
        const Element2D_t& element_L = elements[element_L_index];
        const Element2D_t& element_R = elements[element_R_index];

        if (element_L.additional_nodes_[element_L_side_index] || element_R.additional_nodes_[element_R_side_index]) {
            size_t new_node_index = n_nodes + n_splitting_elements + faces_block_offsets[block_id];
            size_t new_face_index = n_faces + 4 * n_splitting_elements + faces_block_offsets[block_id];
            for (size_t j = face_index - thread_id; j < face_index; ++j) {
                new_node_index += faces[j].refine_;
                new_face_index += faces[j].refine_;
            }

            const Vec2<deviceFloat> new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;

            const int element_L_block_id = element_L_index/elements_blockSize;
            const int element_L_thread_id = element_L_index%elements_blockSize;
            const int element_R_block_id = element_R_index/elements_blockSize;
            const int element_R_thread_id = element_R_index%elements_blockSize;

            size_t element_L_new_index = element_L_index + 3 * block_offsets[element_L_block_id];
            for (size_t j = element_L_index - element_L_thread_id; j < element_L_index; ++j) {
                element_L_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }

            size_t element_R_new_index = element_R_index + 3 * block_offsets[element_R_block_id];
            for (size_t j = element_R_index - element_R_thread_id; j < element_R_index; ++j) {
                element_R_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }

            std::array<std::array<size_t, 2>, 2> new_element_indices {
                {element_L_new_index, element_R_new_index},
                {element_L_new_index, element_R_new_index}
            };



        }
        else {
            const int element_L_block_id = element_L_index/elements_blockSize;
            const int element_L_thread_id = element_L_index%elements_blockSize;
            const int element_R_block_id = element_R_index/elements_blockSize;
            const int element_R_thread_id = element_R_index%elements_blockSize;

            size_t element_L_new_index = element_L_index + 3 * block_offsets[element_L_block_id];
            for (size_t j = element_L_index - element_L_thread_id; j < element_L_index; ++j) {
                element_L_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }

            size_t element_R_new_index = element_R_index + 3 * block_offsets[element_R_block_id];
            for (size_t j = element_R_index - element_R_thread_id; j < element_R_index; ++j) {
                element_R_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }
            
            if (element_L.would_h_refine(max_split_level)) {
                const size_t next_side_index = (element_L_side_index + 1 < element_L.faces_.size()) ? element_L_side_index + 1 : 0;
                const std::array<size_t, 2> new_indices {element_L_new_index + element_L_side_index, element_L_new_index + next_side_index};
                const std::array<Vec2<deviceFloat>, 2> element_nodes {nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[next_side_index]]};
                const Vec2<deviceFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;

                const std::array<Vec2<deviceFloat>, 2>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                const std::array<deviceFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                const std::array<Vec2<deviceFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                const std::array<Vec2<deviceFloat>, 2> AD {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                const std::array<deviceFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<deviceFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};

                // The face is within the first element
                if (C_proj[0] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && C_proj[0] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                    && D_proj[0] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && D_proj[0] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {

                    element_L_new_index = new_indices[0];
                }
                // The face is within the second element
                if (C_proj[1] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && C_proj[1] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                    && D_proj[1] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && D_proj[1] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {

                    element_L_new_index = new_indices[1];
                }

            }
            if (element_R.would_h_refine(max_split_level)) {
                const size_t next_side_index = (element_R_side_index + 1 < element_R.faces_.size()) ? element_R_side_index + 1 : 0;
                const std::array<size_t, 2> new_indices {element_R_new_index + element_R_side_index, element_R_new_index + next_side_index};
                const std::array<Vec2<deviceFloat>, 2> element_nodes {nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[next_side_index]]};
                const Vec2<deviceFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;

                const std::array<Vec2<deviceFloat>, 2>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                const std::array<deviceFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                const std::array<Vec2<deviceFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                const std::array<Vec2<deviceFloat>, 2> AD {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                const std::array<deviceFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<deviceFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};

                // The face is within the first element
                if (C_proj[0] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && C_proj[0] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                    && D_proj[0] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && D_proj[0] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {

                    element_R_new_index = new_indices[0];
                }
                // The face is within the second element
                if (C_proj[1] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && C_proj[1] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()
                    && D_proj[1] + std::numeric_limits<deviceFloat>::epsilon() >= static_cast<deviceFloat>(0) 
                    && D_proj[1] <= static_cast<deviceFloat>(1) + std::numeric_limits<deviceFloat>::epsilon()) {

                    element_R_new_index = new_indices[1];
                }
            }

            face.elements_ = {element_L_new_index, element_R_new_index};

            const int N_max = std::max(element_L.N_ + 2 * element_L.would_p_refine(N_max), element_R.N_ + 2 * element_R.would_p_refine(N_max));

            if (face.N_ != N_max) {
                face.resize_storage(N_max);
            }

            new_faces[face_index] = std::move(face);
        }

        if (face.N_ == -1) { // Face has split
            Face2D_t& new_face = new_faces[face_index];
            const Element2D_t& element_L = elements[face.elements_[0]];
            const Element2D_t& element_R = elements[face.elements_[1]];

            size_t min_element_index = static_cast<size_t>(-1);
            size_t min_element_side = static_cast<size_t>(-1);
            if (element_L.would_h_refine(max_split_level)) {
                if (element_R.would_h_refine(max_split_level)) {
                    if (face.elements_[0] < face.elements_[1]) {
                        min_element_index = face.elements_[0];
                        min_element_side = face.elements_side_[0];
                    }
                    else {
                        min_element_index = face.elements_[1];
                        min_element_side = face.elements_side_[1];
                    }
                }
                else {
                    min_element_index = face.elements_[0];
                    min_element_side = face.elements_side_[0];
                }
            }
            else if (element_R.would_h_refine(max_split_level)) {
                min_element_index = face.elements_[1];
                min_element_side = face.elements_side_[1];
            }
            else { // This should not happen
                fprintf(stderr, "Error: Splitting face %llu has no elements (%llu, %llu) that would split. Exiting.\n", face_index, face.elements_[0], face.elements_[1]);    
                exit(54);
            }

            const int neighbour_block_id = min_element_index/elements_blockSize;
            const int neighbour_thread_id = min_element_index%elements_blockSize;

            size_t new_face_index = n_faces + nodes_block_offsets[neighbour_block_id] + block_offsets[neighbour_block_id]; // Wow this just happens to work, we need to add 3 faces per splitting element to the number of additional nodes, and the element offset is 3 * n_splitting
            for (size_t neighbour_side_index = 0; neighbour_side_index < min_element_side; ++neighbour_side_index) {
                new_face_index += elements[min_element_index].additional_nodes_[neighbour_side_index];
            }
            for (size_t j = min_element_index - neighbour_thread_id; j < min_element_index; ++j) {
                if (elements[j].would_h_refine(max_split_level)) {
                    new_face_index += 4;
                    for (size_t side_index = 0; side_index < elements[j].faces_.size(); ++side_index) {
                        new_face_index += elements[j].additional_nodes_[side_index];
                    }
                }
            }

            Face2D_t& new_face_2 = new_faces[new_face_index];

            // Elements that yielded, always the second one on a face, and non-splitting elements still have the old face
            Element2D_t& new_face_element = new_elements[new_face.elements_[1]];
            Element2D_t& new_face_element_2 = new_elements[new_face_2.elements_[1]];

            new_face.compute_geometry(new_elements, nodes);
            new_face_2.compute_geometry(new_elements, nodes);
        }
        else {
            const size_t element_L_index = face.elements_[0];
            const size_t element_R_index = face.elements_[1];
            const size_t element_L_side = face.elements_side_[0];
            const size_t element_R_side = face.elements_side_[1];
            const Element2D_t& element_L = elements[element_L_index];
            const Element2D_t& element_R = elements[element_R_index];

            const int N_max = std::max(element_L.N_ + 2 * element_L.would_p_refine(N_max), element_R.N_ + 2 * element_R.would_p_refine(N_max));

            if (face.N_ != N_max) {
                face.resize_storage(N_max);
            }

            const int element_L_block_id = element_L_index/elements_blockSize;
            const int element_L_thread_id = element_L_index%elements_blockSize;

            size_t element_L_new_index = element_L_index + block_offsets[element_L_block_id];
            for (size_t j = element_L_index - element_L_thread_id; j < element_L_index; ++j) {
                element_L_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }

            const int element_R_block_id = element_R_index/elements_blockSize;
            const int element_R_thread_id = element_R_index%elements_blockSize;

            size_t element_R_new_index = element_R_index + block_offsets[element_R_block_id];
            for (size_t j = element_R_index - element_R_thread_id; j < element_R_index; ++j) {
                element_R_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }

            // CHECK the new indices thing won't use the Hilbert curve like this
            if (element_L.would_h_refine(max_split_level)) {
                const std::array<size_t, 2> new_indices {element_L_new_index + element_L_side, (element_L_side + 1 < element_L.faces_.size()) ? element_L_new_index + element_L_side + 1: element_L_new_index};
                bool found_face = false;
                for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[0]].faces_[element_L_side].size(), ++element_face_index) {
                    if (new_elements[new_indices[0]].faces_[element_L_side][element_face_index] == face_index) {
                        face.elements_[0] = new_indices[0];
                        found_face = true;
                        break;
                    }
                }
                if (!found_face) {
                    for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[1]].faces_[element_L_side].size(), ++element_face_index) {
                        if (new_elements[new_indices[1]].faces_[element_L_side][element_face_index] == face_index) {
                            face.elements_[0] = new_indices[1];
                            break;
                        }
                    }
                }
            }
            else {
                face.elements_[0] = element_L_new_index;
            }

            if (element_R.would_h_refine(max_split_level)) {
                const std::array<size_t, 2> new_indices {element_R_new_index + element_R_side, (element_R_side + 1 < element_R.faces_.size()) ? element_R_new_index + element_R_side + 1: element_R_new_index};
                bool found_face = false;
                for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[0]].faces_[element_R_side].size(), ++element_face_index) {
                    if (new_elements[new_indices[0]].faces_[element_R_side][element_face_index] == face_index) {
                        face.elements_[1] = new_indices[0];
                        found_face = true;
                        break;
                    }
                }
                if (!found_face) {
                    for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[1]].faces_[element_R_side].size(), ++element_face_index) {
                        if (new_elements[new_indices[1]].faces_[element_R_side][element_face_index] == face_index) {
                            face.elements_[1] = new_indices[1];
                            break;
                        }
                    }
                }
            }
            else {
                face.elements_[1] = element_R_new_index;
            }

            new_faces[face_index] = std::move(face);
        }
    }
}

__global__
auto SEM::Meshes::find_nodes(size_t n_elements, Element2D_t* elements, const Face2D_t* faces, const Vec2<deviceFloat>* nodes, int max_split_level) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (size_t i = index; i < n_elements; i += stride) {
        SEM::Entities::Element2D_t& element = elements[i];
        element.additional_nodes_ = {false, false, false, false};
        if (element.would_h_refine(max_split_level)) {
            element.additional_nodes_ = {true, true, true, true};
            for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                const std::array<SEM::Entities::Vec2<deviceFloat>, 2> side_nodes = {nodes[element.nodes_[side_index]], (side_index + 1 < element.faces_.size()) ? nodes[element.nodes_[side_index + 1]] : nodes[element.nodes_[0]]};
                const SEM::Entities::Vec2<deviceFloat> new_node = (side_nodes[0] + side_nodes[1])/2;

                // Here we check if the new node already exists
                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    const SEM::Entities::Face2D_t& face = faces[element.faces_[side_index][face_index]];
                    if (nodes[face.nodes_[0]].almost_equal(new_node) || nodes[face.nodes_[1]].almost_equal(new_node)) {
                        element.additional_nodes_[side_index] = false;
                        break;
                    }
                }
            }
        }
    }
}


__global__
auto SEM::Meshes::move_faces(size_t n_faces, Face2D_t* faces, Face2D_t* new_faces, Element2D_t* elements, Element2D_t* new_elements, const Vec2<deviceFloat>* nodes, const size_t* block_offsets, const size_t* nodes_block_offsets, int max_split_level, int N_max, int elements_blockSize) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < n_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];

        if (face.N_ == -1) { // Face has split
            Face2D_t& new_face = new_faces[face_index];
            const Element2D_t& element_L = elements[face.elements_[0]];
            const Element2D_t& element_R = elements[face.elements_[1]];

            size_t min_element_index = static_cast<size_t>(-1);
            size_t min_element_side = static_cast<size_t>(-1);
            if (element_L.would_h_refine(max_split_level)) {
                if (element_R.would_h_refine(max_split_level)) {
                    if (face.elements_[0] < face.elements_[1]) {
                        min_element_index = face.elements_[0];
                        min_element_side = face.elements_side_[0];
                    }
                    else {
                        min_element_index = face.elements_[1];
                        min_element_side = face.elements_side_[1];
                    }
                }
                else {
                    min_element_index = face.elements_[0];
                    min_element_side = face.elements_side_[0];
                }
            }
            else if (element_R.would_h_refine(max_split_level)) {
                min_element_index = face.elements_[1];
                min_element_side = face.elements_side_[1];
            }
            else { // This should not happen
                fprintf(stderr, "Error: Splitting face %llu has no elements (%llu, %llu) that would split. Exiting.\n", face_index, face.elements_[0], face.elements_[1]);    
                exit(54);
            }

            const int neighbour_block_id = min_element_index/elements_blockSize;
            const int neighbour_thread_id = min_element_index%elements_blockSize;

            size_t new_face_index = n_faces + nodes_block_offsets[neighbour_block_id] + block_offsets[neighbour_block_id]; // Wow this just happens to work, we need to add 3 faces per splitting element to the number of additional nodes, and the element offset is 3 * n_splitting
            for (size_t neighbour_side_index = 0; neighbour_side_index < min_element_side; ++neighbour_side_index) {
                new_face_index += elements[min_element_index].additional_nodes_[neighbour_side_index];
            }
            for (size_t j = min_element_index - neighbour_thread_id; j < min_element_index; ++j) {
                if (elements[j].would_h_refine(max_split_level)) {
                    new_face_index += 4;
                    for (size_t side_index = 0; side_index < elements[j].faces_.size(); ++side_index) {
                        new_face_index += elements[j].additional_nodes_[side_index];
                    }
                }
            }

            Face2D_t& new_face_2 = new_faces[new_face_index];

            // Elements that yielded, always the second one on a face, and non-splitting elements still have the old face
            Element2D_t& new_face_element = new_elements[new_face.elements_[1]];
            Element2D_t& new_face_element_2 = new_elements[new_face_2.elements_[1]];

            new_face.compute_geometry(new_elements, nodes);
            new_face_2.compute_geometry(new_elements, nodes);
        }
        else {
            const size_t element_L_index = face.elements_[0];
            const size_t element_R_index = face.elements_[1];
            const size_t element_L_side = face.elements_side_[0];
            const size_t element_R_side = face.elements_side_[1];
            const Element2D_t& element_L = elements[element_L_index];
            const Element2D_t& element_R = elements[element_R_index];

            const int N_max = std::max(element_L.N_ + 2 * element_L.would_p_refine(N_max), element_R.N_ + 2 * element_R.would_p_refine(N_max));

            if (face.N_ != N_max) {
                face.resize_storage(N_max);
            }

            const int element_L_block_id = element_L_index/elements_blockSize;
            const int element_L_thread_id = element_L_index%elements_blockSize;

            size_t element_L_new_index = element_L_index + block_offsets[element_L_block_id];
            for (size_t j = element_L_index - element_L_thread_id; j < element_L_index; ++j) {
                element_L_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }

            const int element_R_block_id = element_R_index/elements_blockSize;
            const int element_R_thread_id = element_R_index%elements_blockSize;

            size_t element_R_new_index = element_R_index + block_offsets[element_R_block_id];
            for (size_t j = element_R_index - element_R_thread_id; j < element_R_index; ++j) {
                element_R_new_index += 3 * elements[j].would_h_refine(max_split_level);
            }

            // CHECK the new indices thing won't use the Hilbert curve like this
            if (element_L.would_h_refine(max_split_level)) {
                const std::array<size_t, 2> new_indices {element_L_new_index + element_L_side, (element_L_side + 1 < element_L.faces_.size()) ? element_L_new_index + element_L_side + 1: element_L_new_index};
                bool found_face = false;
                for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[0]].faces_[element_L_side].size(), ++element_face_index) {
                    if (new_elements[new_indices[0]].faces_[element_L_side][element_face_index] == face_index) {
                        face.elements_[0] = new_indices[0];
                        found_face = true;
                        break;
                    }
                }
                if (!found_face) {
                    for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[1]].faces_[element_L_side].size(), ++element_face_index) {
                        if (new_elements[new_indices[1]].faces_[element_L_side][element_face_index] == face_index) {
                            face.elements_[0] = new_indices[1];
                            break;
                        }
                    }
                }
            }
            else {
                face.elements_[0] = element_L_new_index;
            }

            if (element_R.would_h_refine(max_split_level)) {
                const std::array<size_t, 2> new_indices {element_R_new_index + element_R_side, (element_R_side + 1 < element_R.faces_.size()) ? element_R_new_index + element_R_side + 1: element_R_new_index};
                bool found_face = false;
                for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[0]].faces_[element_R_side].size(), ++element_face_index) {
                    if (new_elements[new_indices[0]].faces_[element_R_side][element_face_index] == face_index) {
                        face.elements_[1] = new_indices[0];
                        found_face = true;
                        break;
                    }
                }
                if (!found_face) {
                    for (size_t element_face_index = 0; element_face_index < new_elements[new_indices[1]].faces_[element_R_side].size(), ++element_face_index) {
                        if (new_elements[new_indices[1]].faces_[element_R_side][element_face_index] == face_index) {
                            face.elements_[1] = new_indices[1];
                            break;
                        }
                    }
                }
            }
            else {
                face.elements_[1] = element_R_new_index;
            }

            new_faces[face_index] = std::move(face);
        }
    }
}


__global__
auto SEM::Meshes::adjust_boundaries(size_t n_boundaries, Element2D_t* elements, const size_t* boundaries, const Face2D_t* faces) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t boundary_index = index; boundary_index < n_boundaries; boundary_index += stride) {
        Element2D_t& destination_element = elements[boundaries[boundary_index]];
        int N_max = destination_element.N_;

        for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
            const Face2D_t face = faces[destination_element.faces_[0][face_index]];
            const int element_index = face.elements_[0] == boundaries[boundary_index];
            const Element2D_t& source_element = elements[face.elements_[element_index]];

            N_max = std::max(N_max, source_element.N_);
        }

        if (destination_element.N_ != N_max) {
            destination_element.resize_boundary_storage(N_max);
        }
    }
}

__global__
auto SEM::Meshes::rebuild_boundaries(size_t n_boundaries, Element2D_t* elements, Element2D_t* new_elements, const size_t* boundaries, const Face2D_t* faces) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t boundary_index = index; boundary_index < n_boundaries; boundary_index += stride) {
        Element2D_t& destination_element = elements[boundaries[boundary_index]];
        int N_max = destination_element.N_;

        for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
            const Face2D_t face = faces[destination_element.faces_[0][face_index]];
            const int element_index = face.elements_[0] == boundaries[boundary_index];
            const Element2D_t& source_element = elements[face.elements_[element_index]];

            N_max = std::max(N_max, source_element.N_);
        }

        if (destination_element.N_ != N_max) {
            destination_element.resize_boundary_storage(N_max);
        }

        new_elements[boundaries[boundary_index]] = std::move(elements[boundaries[boundary_index]]);
    }
}

__global__
auto SEM::Meshes::adjust_interfaces(size_t n_local_interfaces, Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_local_interfaces; interface_index += stride) {
        const Element2D_t& source_element = elements[local_interfaces_origin[interface_index]];
        Element2D_t& destination_element = elements[local_interfaces_destination[interface_index]];

        if (destination_element.N_ != source_element.N_) {
            destination_element.resize_boundary_storage(source_element.N_);
        }
    }
}

__global__
auto SEM::Meshes::rebuild_interfaces(size_t n_local_interfaces, Element2D_t* elements, Element2D_t* new_elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_destination) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < n_local_interfaces; interface_index += stride) {
        const Element2D_t& source_element = new_elements[local_interfaces_origin[interface_index]];
        Element2D_t& destination_element = elements[local_interfaces_destination[interface_index]];

        if (destination_element.N_ != source_element.N_) {
            destination_element.resize_boundary_storage(source_element.N_);
        }

        new_elements[local_interfaces_destination[interface_index]] = std::move(elements[local_interfaces_destination[interface_index]]);
    }
}

__global__
auto SEM::Meshes::adjust_faces(size_t n_faces, Face2D_t* faces, const Element2D_t* elements) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < n_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];
        const Element2D_t& element_L = elements[face.elements_[0]];
        const Element2D_t& element_R = elements[face.elements_[1]];

        const int N_max = std::max(element_L.N_, element_R.N_);

        if (face.N_ != N_max) {
            face.resize_storage(N_max);
        }
    }
}
