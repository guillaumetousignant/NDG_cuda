#include "meshes/Mesh2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "helpers/constants.h"
#include "functions/Utilities.h"
#include "functions/quad_map.cuh"
#include "functions/quad_metrics.cuh"
#include "cgnslib.h"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

using SEM::Entities::device_vector;
using SEM::Entities::Vec2;
using SEM::Entities::Element2D_t;
using SEM::Entities::Face2D_t;

constexpr int CGIO_MAX_NAME_LENGTH = 33; // Includes the null terminator

SEM::Meshes::Mesh2D_t::Mesh2D_t(std::filesystem::path filename, int initial_N, int maximum_N, const SEM::Entities::device_vector<deviceFloat>& polynomial_nodes, cudaStream_t &stream) :       
        initial_N_(initial_N),  
        maximum_N_(maximum_N),      
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

    compute_element_geometry<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), nodes_.data(), polynomial_nodes.data());
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
                
                // Calculating min element length, from left and right sides, top and bottom sides, and finally both diagonals                                                    
                element.delta_xy_min_ = std::min(std::min(
                    std::min((host_nodes[element.nodes_[1]] - host_nodes[element.nodes_[0]]).magnitude(), (host_nodes[element.nodes_[2]] - host_nodes[element.nodes_[3]]).magnitude()), 
                    std::min((host_nodes[element.nodes_[1]] - host_nodes[element.nodes_[2]]).magnitude(), (host_nodes[element.nodes_[0]] - host_nodes[element.nodes_[3]]).magnitude())), 
                    std::min((host_nodes[element.nodes_[1]] - host_nodes[element.nodes_[3]]).magnitude(), (host_nodes[element.nodes_[2]] - host_nodes[element.nodes_[0]]).magnitude()));
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

                // Calculating min element length from its (only) side                                                    
                element.delta_xy_min_ = (host_nodes[element.nodes_[1]] - host_nodes[element.nodes_[0]]).magnitude();
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

    // Building boundaries
    std::vector<size_t> wall_boundaries;
    std::vector<size_t> symmetry_boundaries;

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

            case BCType_t::BCTypeNull:
                break;

            default:
                std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << i << " has an unknown boundary type. For now only BCWall, BCSymmetryPlane and BCTypeNull are implemented. Exiting." << std::endl;
                exit(35);
        }
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

    if (n_mpi_interfaces > 0) {
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
    }

    // Transferring onto the GPU
    nodes_ = host_nodes;
    elements_ = host_elements;
    faces_ = host_faces;
    wall_boundaries_ = wall_boundaries;
    symmetry_boundaries_ = symmetry_boundaries;
    interfaces_origin_ = interfaces_origin;
    interfaces_origin_side_ = interfaces_origin_side;
    interfaces_destination_ = interfaces_destination;
    mpi_interfaces_origin_ = mpi_interfaces_origin;
    mpi_interfaces_origin_side_ = mpi_interfaces_origin_side;
    mpi_interfaces_destination_ = mpi_interfaces_destination_in_this_proc;

    // Setting sizes
    N_elements_ = n_elements_domain;
    elements_numBlocks_ = (N_elements_ + elements_blockSize_ - 1) / elements_blockSize_;
    faces_numBlocks_ = (faces_.size() + faces_blockSize_ - 1) / faces_blockSize_;
    wall_boundaries_numBlocks_ = (wall_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    symmetry_boundaries_numBlocks_ = (symmetry_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    all_boundaries_numBlocks_ = (interfaces_origin_.size() + wall_boundaries_.size() + symmetry_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    interfaces_numBlocks_ = (interfaces_origin_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    mpi_interfaces_numBlocks_ = (mpi_interfaces_origin_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;

    host_delta_t_array_ = std::vector<deviceFloat>(elements_numBlocks_);
    device_delta_t_array_ = device_vector<deviceFloat>(elements_numBlocks_);

    device_interfaces_p_ = device_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    device_interfaces_u_ = device_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    device_interfaces_v_ = device_vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    host_interfaces_p_ = std::vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    host_interfaces_u_ = std::vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    host_interfaces_v_ = std::vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    host_receiving_interfaces_p_ = std::vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    host_receiving_interfaces_u_ = std::vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));
    host_receiving_interfaces_v_ = std::vector<deviceFloat>(mpi_interfaces_origin.size() * (maximum_N_ + 1));

    requests_ = std::vector<MPI_Request>(n_mpi_interfaces * 6);
    statuses_ = std::vector<MPI_Status>(n_mpi_interfaces * 6);

    allocate_element_storage<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data());
    allocate_boundary_storage<<<all_boundaries_numBlocks_, boundaries_blockSize_, 0, stream_>>>(N_elements_, elements_.size(), elements_.data());
    allocate_face_storage<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data());

    const SEM::Entities::device_vector<std::array<size_t, 4>> device_element_to_face(element_to_face);
    fill_element_faces<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(elements_.size(), elements_.data(), device_element_to_face.data());
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
            const size_t node_index_next = (j < element.nodes_.size() - 1) ? element.nodes_[j + 1] : element.nodes_[0];

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
            const std::array<size_t, 2> nodes{elements[i].nodes_[j], (j < elements[i].nodes_.size() - 1) ? elements[i].nodes_[j + 1] : elements[i].nodes_[0]};

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

auto SEM::Meshes::Mesh2D_t::initial_conditions(const deviceFloat* polynomial_nodes) -> void {
    initial_conditions_2D<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), nodes_.data(), polynomial_nodes);
}

auto SEM::Meshes::Mesh2D_t::print() const -> void {
    std::vector<Face2D_t> host_faces(faces_.size());
    std::vector<Element2D_t> host_elements(elements_.size());
    std::vector<Vec2<deviceFloat>> host_nodes(nodes_.size());
    std::vector<size_t> host_wall_boundaries(wall_boundaries_.size());
    std::vector<size_t> host_symmetry_boundaries(symmetry_boundaries_.size());
    std::vector<size_t> host_interfaces_origin(interfaces_origin_.size());
    std::vector<size_t> host_interfaces_origin_side(interfaces_origin_side_.size());
    std::vector<size_t> host_interfaces_destination(interfaces_destination_.size());
    
    faces_.copy_to(host_faces);
    elements_.copy_to(host_elements);
    nodes_.copy_to(host_nodes);
    wall_boundaries_.copy_to(host_wall_boundaries);
    symmetry_boundaries_.copy_to(host_symmetry_boundaries);
    interfaces_origin_.copy_to(host_interfaces_origin);
    interfaces_origin_side_.copy_to(host_interfaces_origin_side);
    interfaces_destination_.copy_to(host_interfaces_destination);
    
    std::cout << "N elements: " << N_elements_ << std::endl;
    std::cout << "N elements and ghosts: " << elements_.size() << std::endl;
    std::cout << "N faces: " << faces_.size() << std::endl;
    std::cout << "N nodes: " << nodes_.size() << std::endl;
    std::cout << "N wall boundaries: " << wall_boundaries_.size() << std::endl;
    std::cout << "N symmetry boundaries: " << symmetry_boundaries_.size() << std::endl;
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

    std::cout << std::endl;
}

auto SEM::Meshes::Mesh2D_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) const -> void {
    device_vector<deviceFloat> x(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> y(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> p(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> u(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> v(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> dp_dt(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> du_dt(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> dv_dt(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<int> N(N_elements_);

    SEM::Meshes::get_solution<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, N_interpolation_points, elements_.data(), nodes_.data(), interpolation_matrices, x.data(), y.data(), p.data(), u.data(), v.data(), N.data(), dp_dt.data(), du_dt.data(), dv_dt.data());
    
    std::vector<deviceFloat> x_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> y_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> p_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> u_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> v_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> dp_dt_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> du_dt_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> dv_dt_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<int> N_host(N_elements_);

    x.copy_to(x_host);
    y.copy_to(y_host);
    p.copy_to(p_host);
    u.copy_to(u_host);
    v.copy_to(v_host);
    dp_dt.copy_to(dp_dt_host);
    du_dt.copy_to(du_dt_host);
    dv_dt.copy_to(dv_dt_host);
    N.copy_to(N_host);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    data_writer.write_data(N_interpolation_points, N_elements_, time, global_rank, x_host, y_host, p_host, u_host, v_host, N_host, dp_dt_host, du_dt_host, dv_dt_host);
}

__host__ __device__
auto SEM::Meshes::Mesh2D_t::g(Vec2<deviceFloat> xy, deviceFloat t) -> std::array<deviceFloat, 3> {    
    const deviceFloat p = std::exp(-(SEM::Constants::k.x() * (xy.x() - SEM::Constants::xy0.x()) + SEM::Constants::k.y() * (xy.y() - SEM::Constants::xy0.y()) - SEM::Constants::c * t) * (SEM::Constants::k.x() * (xy.x() - SEM::Constants::xy0.x()) + SEM::Constants::k.y() * (xy.y() - SEM::Constants::xy0.y()) - SEM::Constants::c * t) / (SEM::Constants::d * SEM::Constants::d));

    return {p,
            p * SEM::Constants::k.x() / SEM::Constants::c,
            p * SEM::Constants::k.y() / SEM::Constants::c};
}

auto SEM::Meshes::Mesh2D_t::adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) -> void {
    std::cout << "Warning, SEM::Meshes::Mesh2D_t::adapt is not implemented." << std::endl;
}

auto SEM::Meshes::Mesh2D_t::boundary_conditions() -> void {
    if (interfaces_origin_.size() > 0) {
        SEM::Meshes::local_interfaces<<<interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(interfaces_origin_.size(), elements_.data(), interfaces_origin_.data(), interfaces_origin_side_.data(), interfaces_destination_.data());
    }

    if (mpi_interfaces_origin_.size() > 0) {
        int global_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
        int global_size;
        MPI_Comm_size(MPI_COMM_WORLD, &global_size);

        SEM::Meshes::get_MPI_interfaces<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(global_rank, mpi_interfaces_origin_.size(), elements_.data(), mpi_interfaces_origin_.data(), mpi_interfaces_origin_side_.data(), maximum_N_, device_interfaces_p_.data(), device_interfaces_u_.data(), device_interfaces_v_.data());

        device_interfaces_p_.copy_to(host_interfaces_p_);
        device_interfaces_u_.copy_to(host_interfaces_u_);
        device_interfaces_v_.copy_to(host_interfaces_v_);

        
        constexpr MPI_Datatype data_type = (sizeof(deviceFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
        for (size_t i = 0; i < mpi_interfaces_size_.size(); ++i) {
            MPI_Isend(host_interfaces_p_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * global_rank + mpi_interfaces_process_[i]), MPI_COMM_WORLD, &requests_[3 * (mpi_interfaces_size_.size() + i)]);
            MPI_Irecv(host_receiving_interfaces_p_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i],  3 * (global_size * mpi_interfaces_process_[i] + global_rank), MPI_COMM_WORLD, &requests_[3 * i]);

            MPI_Isend(host_interfaces_u_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i], 3 * (global_size * global_rank + mpi_interfaces_process_[i]) + 1, MPI_COMM_WORLD, &requests_[3 * (mpi_interfaces_size_.size() + i) + 1]);
            MPI_Irecv(host_receiving_interfaces_u_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i],  3 * (global_size * mpi_interfaces_process_[i] + global_rank) + 1, MPI_COMM_WORLD, &requests_[3 * i + 1]);

            MPI_Isend(host_interfaces_v_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i],  3 * (global_size * global_rank + mpi_interfaces_process_[i]) + 2, MPI_COMM_WORLD, &requests_[3 * (mpi_interfaces_size_.size() + i) + 2]);
            MPI_Irecv(host_receiving_interfaces_v_.data() + mpi_interfaces_offset_[i], mpi_interfaces_size_[i] * (maximum_N_ + 1), data_type, mpi_interfaces_process_[i],  3 * (global_size * mpi_interfaces_process_[i] + global_rank) + 2, MPI_COMM_WORLD, &requests_[3 * i + 2]);
        }

        MPI_Waitall(3 * mpi_interfaces_size_.size(), requests_.data(), statuses_.data());

        device_interfaces_p_.copy_from(host_receiving_interfaces_p_);
        device_interfaces_u_.copy_from(host_receiving_interfaces_u_);
        device_interfaces_v_.copy_from(host_receiving_interfaces_v_);

        SEM::Meshes::put_MPI_interfaces<<<mpi_interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(mpi_interfaces_destination_.size(), elements_.data(), mpi_interfaces_destination_.data(), maximum_N_, device_interfaces_p_.data(), device_interfaces_u_.data(), device_interfaces_v_.data());

        MPI_Waitall(3 * mpi_interfaces_size_.size(), requests_.data() + 3 * mpi_interfaces_size_.size(), statuses_.data() + 3 * mpi_interfaces_size_.size());
    }
}

auto SEM::Meshes::Mesh2D_t::interpolate_to_boundaries(const device_vector<deviceFloat>& lagrange_interpolant_left, const device_vector<deviceFloat>& lagrange_interpolant_right) -> void {
    SEM::Meshes::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), lagrange_interpolant_left.data(), lagrange_interpolant_right.data());
}

auto SEM::Meshes::Mesh2D_t::project_to_faces() -> void {
    SEM::Meshes::project_to_faces<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data(), elements_.data());
}

auto SEM::Meshes::Mesh2D_t::project_to_elements() -> void {
    SEM::Meshes::project_to_elements<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, faces_.data(), elements_.data());
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
        SEM::Entities::Element2D_t& element = elements[element_index];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;
        const std::array<Vec2<deviceFloat>, 4> points {nodes[element.nodes_[0]],
                                                       nodes[element.nodes_[1]],
                                                       nodes[element.nodes_[2]],
                                                       nodes[element.nodes_[3]]};

        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
                const std::array<Vec2<deviceFloat>, 2> metrics = SEM::quad_metrics(coordinates, points);

                element.dxi_dx_[i * (element.N_ + 1) + j]  = metrics[0].x();
                element.deta_dx_[i * (element.N_ + 1) + j] = metrics[0].y();
                element.dxi_dy_[i * (element.N_ + 1) + j]  = metrics[1].x();
                element.deta_dy_[i * (element.N_ + 1) + j] = metrics[1].y();
                element.jacobian_[i * (element.N_ + 1) + j] = metrics[0].x() * metrics[1].y() - metrics[0].y() * metrics[1].x();
            }

            const Vec2<deviceFloat> coordinates_bottom {polynomial_nodes[offset_1D + i], -1};
            const Vec2<deviceFloat> coordinates_right  {1, polynomial_nodes[offset_1D + i]};
            const Vec2<deviceFloat> coordinates_top    {polynomial_nodes[offset_1D + i], 1};
            const Vec2<deviceFloat> coordinates_left   {-1, polynomial_nodes[offset_1D + i]};

            const std::array<Vec2<deviceFloat>, 2> metrics_bottom = SEM::quad_metrics(coordinates_bottom, points);
            const std::array<Vec2<deviceFloat>, 2> metrics_right  = SEM::quad_metrics(coordinates_right, points);
            const std::array<Vec2<deviceFloat>, 2> metrics_top    = SEM::quad_metrics(coordinates_top, points);
            const std::array<Vec2<deviceFloat>, 2> metrics_left   = SEM::quad_metrics(coordinates_left, points);

            element.scaling_factor_[0][i] = std::sqrt(metrics_bottom[0].x() * metrics_bottom[0].x() + metrics_bottom[1].x() * metrics_bottom[1].x());
            element.scaling_factor_[1][i] = std::sqrt(metrics_right[0].y() * metrics_right[0].y() + metrics_right[1].y() * metrics_right[1].y());
            element.scaling_factor_[2][i] = std::sqrt(metrics_top[0].x() * metrics_top[0].x() + metrics_top[1].x() * metrics_top[1].x());
            element.scaling_factor_[3][i] = std::sqrt(metrics_left[0].y() * metrics_left[0].y() + metrics_left[1].y() * metrics_left[1].y());
        }    
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
        SEM::Entities::Face2D_t& face = faces[face_index];
        face.tangent_ = nodes[face.nodes_[1]] - nodes[face.nodes_[0]]; 
        face.length_ = face.tangent_.magnitude();
        face.tangent_ /= face.length_; // CHECK should be normalized or not?
        face.normal_ = Vec2<deviceFloat>(face.tangent_.y(), -face.tangent_.x());         

        const Vec2<deviceFloat> center = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]]) * 0.5;
        Vec2<deviceFloat> element_center {0.0};
        for (const auto node_index : elements[face.elements_[0]].nodes_) {
            element_center += nodes[node_index];
        }
        element_center /= elements[face.elements_[0]].nodes_.size();

        const Vec2<deviceFloat> delta = center - element_center; // CHECK doesn't work with ghost cells
        const double sign = std::copysign(1.0, face.normal_.dot(delta));
        face.normal_ *= sign;
        face.tangent_ *= sign;
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
auto SEM::Meshes::get_solution(size_t N_elements, size_t N_interpolation_points, const Element2D_t* elements, const Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, int* N, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
        const Element2D_t& element = elements[element_index];
        const size_t offset_interp_2D = element_index * N_interpolation_points * N_interpolation_points;
        const size_t offset_interp = element.N_ * (element.N_ + 1) * N_interpolation_points/2;

        N[element_index] = element.N_;
        const std::array<Vec2<deviceFloat>, 4> points {nodes[element.nodes_[0]],
                                                       nodes[element.nodes_[1]],
                                                       nodes[element.nodes_[2]],
                                                       nodes[element.nodes_[3]]};

        element.interpolate_solution(N_interpolation_points, points, interpolation_matrices + offset_interp, x + offset_interp_2D, y + offset_interp_2D, p + offset_interp_2D, u + offset_interp_2D, v + offset_interp_2D, dp_dt + offset_interp_2D, du_dt + offset_interp_2D, dv_dt + offset_interp_2D);
    }
}

template __global__ auto SEM::Meshes::estimate_error<SEM::Polynomials::ChebyshevPolynomial_t>(size_t N_elements, Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;
template __global__ auto SEM::Meshes::estimate_error<SEM::Polynomials::LegendrePolynomial_t>(size_t N_elements, Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

template<typename Polynomial>
__global__
auto SEM::Meshes::estimate_error<Polynomial>(size_t N_elements, Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
        elements[element_index].estimate_error<Polynomial>(polynomial_nodes, weights);
    }
}

__global__
auto SEM::Meshes::interpolate_to_boundaries(size_t N_elements, Element2D_t* elements, const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
        elements[element_index].interpolate_to_boundaries(lagrange_interpolant_minus, lagrange_interpolant_plus);
    }
}

__global__
auto SEM::Meshes::project_to_faces(size_t N_faces, Face2D_t* faces, const Element2D_t* elements) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t face_index = index; face_index < N_faces; face_index += stride) {
        Face2D_t& face = faces[face_index];

        // Getting element solution
        const Element2D_t& element_L = elements[face.elements_[0]];
        // Conforming
        if ((face.N_ == element_L.N_) 
                && (face.nodes_[0] == element_L.nodes_[face.elements_side_[0]]) 
                && (face.nodes_[1] == element_L.nodes_[(face.elements_side_[0] + 1) * (!(face.elements_side_[0] == (element_L.nodes_.size() - 1)))])) {
            for (int i = 0; i <= face.N_; ++i) {
                face.p_[0][i] = element_L.p_extrapolated_[face.elements_side_[0]][i];
                face.u_[0][i] = element_L.u_extrapolated_[face.elements_side_[0]][i];
                face.v_[0][i] = element_L.v_extrapolated_[face.elements_side_[0]][i];
            }
        }
        else { // We need to interpolate
            printf("Warning, non-conforming surfaces are not implemented yet to project to boundaries.\n");
        }

        const Element2D_t& element_R = elements[face.elements_[1]];
        // Conforming, but reversed
        if ((face.N_ == element_R.N_) 
                && (face.nodes_[1] == element_R.nodes_[face.elements_side_[1]]) 
                && (face.nodes_[0] == element_R.nodes_[(face.elements_side_[1] + 1) * (!(face.elements_side_[1] == (element_R.nodes_.size() - 1)))])) {
            for (int i = 0; i <= face.N_; ++i) {
                face.p_[1][face.N_ - i] = element_R.p_extrapolated_[face.elements_side_[1]][i];
                face.u_[1][face.N_ - i] = element_R.u_extrapolated_[face.elements_side_[1]][i];
                face.v_[1][face.N_ - i] = element_R.v_extrapolated_[face.elements_side_[1]][i];
            }
        }
        else { // We need to interpolate
            printf("Warning, non-conforming surfaces are not implemented yet to project to boundaries.\n");
        }
    }
}

__global__
auto SEM::Meshes::project_to_elements(size_t N_elements, const Face2D_t* faces, Element2D_t* elements) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
        Element2D_t& element = elements[element_index];

        for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
            // Conforming, forward
            if ((element.faces_[side_index].size() == 1)
                    && (faces[element.faces_[side_index][0]].N_ == element.N_)  
                    && (faces[element.faces_[side_index][0]].nodes_[0] == element.nodes_[side_index]) 
                    && (faces[element.faces_[side_index][0]].nodes_[1] == element.nodes_[(side_index + 1) * !(side_index == (element.faces_.size() - 1))])) {

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
                    && (faces[element.faces_[side_index][0]].nodes_[0] == element.nodes_[(side_index + 1) * !(side_index == (element.faces_.size() - 1))])) {

                const Face2D_t& face = faces[element.faces_[side_index][0]];
                for (int j = 0; j <= face.N_; ++j) {
                    element.p_flux_extrapolated_[side_index][face.N_ - j] = -face.p_flux_[j] * element.scaling_factor_[side_index][j];
                    element.u_flux_extrapolated_[side_index][face.N_ - j] = -face.u_flux_[j] * element.scaling_factor_[side_index][j];
                    element.v_flux_extrapolated_[side_index][face.N_ - j] = -face.v_flux_[j] * element.scaling_factor_[side_index][j];
                }
            }
            else { // We need to interpolate
                printf("Warning, non-conforming surfaces are not implemented yet to project to elements.\n");
            }
        }
    }
}

__global__
auto SEM::Meshes::local_interfaces(size_t N_local_interfaces, Element2D_t* elements, const size_t* local_interfaces_origin, const size_t* local_interfaces_origin_side, const size_t* local_interfaces_destination) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < N_local_interfaces; interface_index += stride) {
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
auto SEM::Meshes::get_MPI_interfaces(size_t N_MPI_interface_elements, const Element2D_t* elements, const size_t* MPI_interfaces_origin, const size_t* MPI_interfaces_origin_side, int maximum_N, deviceFloat* p_, deviceFloat* u_, deviceFloat* v_) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < N_MPI_interface_elements; interface_index += stride) {
        const Element2D_t& source_element = elements[MPI_interfaces_origin[interface_index]];
        const size_t element_side = MPI_interfaces_origin_side[interface_index];
        const size_t boundary_offset = interface_index * (maximum_N + 1);

        for (int k = 0; k <= source_element.N_; ++k) {
            p_[boundary_offset + k] = source_element.p_extrapolated_[element_side][k];
            u_[boundary_offset + k] = source_element.u_extrapolated_[element_side][k];
            v_[boundary_offset + k] = source_element.v_extrapolated_[element_side][k];
        }
    }
}

__global__
auto SEM::Meshes::put_MPI_interfaces(size_t N_MPI_interface_elements, Element2D_t* elements, const size_t* MPI_interfaces_destination, int maximum_N, const deviceFloat* p_, const deviceFloat* u_, const deviceFloat* v_) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t interface_index = index; interface_index < N_MPI_interface_elements; interface_index += stride) {
        Element2D_t& destination_element = elements[MPI_interfaces_destination[interface_index]];
        const size_t boundary_offset = interface_index * (maximum_N + 1);

        for (int k = 0; k <= destination_element.N_; ++k) {
            destination_element.p_extrapolated_[0][k] = p_[boundary_offset + k];
            destination_element.u_extrapolated_[0][k] = u_[boundary_offset + k];
            destination_element.v_extrapolated_[0][k] = v_[boundary_offset + k];
        }
    }
}
