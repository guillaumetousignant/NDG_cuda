#include "meshes/Mesh2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "helpers/ProgressBar_t.h"
#include "functions/Utilities.h"
#include "functions/quad_map.cuh"
#include "entities/cuda_vector.cuh"
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
using SEM::Entities::Element2D_t;
using SEM::Entities::Face2D_t;

constexpr int CGIO_MAX_NAME_LENGTH = 33; // Includes the null terminator

SEM::Meshes::Mesh2D_t::Mesh2D_t(std::filesystem::path filename, int initial_N, cudaStream_t &stream) :       
        initial_N_(initial_N),        
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

auto SEM::Meshes::Mesh2D_t::read_su2(std::filesystem::path filename) -> void {
    std::cerr << "Error: SU2 meshes not implemented yet. Exiting." << std::endl;
    exit(15);
}

auto SEM::Meshes::Mesh2D_t::read_cgns(std::filesystem::path filename) -> void {
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

        if (connectivity_sizes[index_connectivity - 1] != connectivity_donor_sizes[index_connectivity - 1]) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", connectivity " << index_connectivity << " has a different number of elements in the origin and destination zones. Exiting." << std::endl;
            exit(30);
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
            exit(31);
        }

        cg_boco_gridlocation_read(index_file, index_base, index_zone, index_boundary, &boundary_grid_locations[index_boundary - 1]);

        if (boundary_grid_locations[index_boundary - 1] != GridLocation_t::EdgeCenter) {
            std::cerr << "Error: CGNS mesh, base " << index_base << ", zone " << index_zone << ", boundary " << index_boundary << " has a grid location that isn't EdgeCenter. For now only EdgeCenter grid locations are supported. Exiting." << std::endl;
            exit(32);
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

    // Putting nodes in the format used by the mesh
    std::vector<Vec2<deviceFloat>> host_nodes(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        host_nodes[i].x() = xy[0][i];
        host_nodes[i].y() = xy[1][i];
    }

    // Figuring out which sections are the domain and which are ghost cells
    std::vector<bool> section_is_domain(n_sections);
    int n_elements_domain = 0;
    int n_elements_ghost = 0;
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
            for (int j = 0; j < section_ranges[i][1] - section_ranges[i][0] + 1; ++j) {
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
            for (int j = 0; j < section_ranges[i][1] - section_ranges[i][0] + 1; ++j) {
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
    auto [host_faces, node_to_face, element_to_face] = build_faces(n_nodes, initial_N_, host_elements);

    // Building boundaries
    std::vector<size_t> wall_boundaries;
    std::vector<size_t> symmetry_boundaries;

    for (int i = 0; i < n_boundaries; ++i) {
        switch (boundary_types[i]) {
            case BCType_t::BCWall:
                wall_boundaries.reserve(wall_boundaries.size() + boundary_sizes[i]);
                for (int j = 0; j < boundary_sizes[i]; ++j) {
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
                for (int j = 0; j < boundary_sizes[i]; ++j) {
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

    // Building interfaces
    size_t n_interface_elements = 0;
    std::vector<size_t> interface_start_index(n_connectivity);
    for (int i = 0; i < n_connectivity; ++i) {
        interface_start_index[i] = n_interface_elements;
        n_interface_elements += connectivity_sizes[i];
    }
    std::vector<std::array<size_t, 2>> interfaces(n_interface_elements);

    for (int i = 0; i < n_connectivity; ++i) {
        for (int j = 0; j < connectivity_sizes[i]; ++j) {
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

            interfaces[interface_start_index[i] + j] = {section_start_indices[origin_section_index] + interface_elements[i][j] - section_ranges[origin_section_index][0], 
                                                        section_start_indices[donor_section_index] + interface_donor_elements[i][j] - section_ranges[donor_section_index][0]};
        }
    }

    // Transferring onto the GPU
    nodes_ = host_nodes;
    elements_ = host_elements;
    faces_ = host_faces;
    interfaces_ = interfaces;
    wall_boundaries_ = wall_boundaries;
    symmetry_boundaries_ = symmetry_boundaries;

    // Setting sizes
    N_elements_ = n_elements_domain;
    elements_numBlocks_ = (N_elements_ + elements_blockSize_ - 1) / elements_blockSize_;
    faces_numBlocks_ = (faces_.size() + faces_blockSize_ - 1) / faces_blockSize_;
    interfaces_numBlocks_ = (interfaces_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    wall_boundaries_numBlocks_ = (wall_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;
    symmetry_boundaries_numBlocks_ = (symmetry_boundaries_.size() + boundaries_blockSize_ - 1) / boundaries_blockSize_;

    host_delta_t_array_ = std::vector<deviceFloat>(elements_numBlocks_);
    device_delta_t_array_ = device_vector<deviceFloat>(elements_numBlocks_);

    allocate_element_storage<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(elements_.size(), elements_.data());
    allocate_face_storage<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data());
    compute_face_geometry<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces_.size(), faces_.data(), elements_.data(), nodes_.data());

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

                    auto it = find(element_neighbor.nodes_.begin(), element_neighbor.nodes_.end(), node_index);
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

auto SEM::Meshes::Mesh2D_t::build_faces(size_t n_nodes, int initial_N, const std::vector<Element2D_t>& elements) -> std::tuple<std::vector<Face2D_t>, std::vector<std::vector<size_t>>, std::vector<std::array<size_t, 4>>> {
    size_t total_edges = 0;
    for (const auto& element: elements) {
        total_edges += element.nodes_.size();
    }

    std::vector<Face2D_t> faces;
    faces.reserve(total_edges/2); // This is not exact

    std::vector<std::vector<size_t>> node_to_face(n_nodes);
    std::vector<std::array<size_t, 4>> element_to_face(elements.size());

    for (size_t i = 0; i < elements.size(); ++i) {
        for (size_t j = 0; j < elements[i].nodes_.size(); ++j) {
            std::array<size_t, 2> nodes{elements[i].nodes_[j], (j < elements[i].nodes_.size() - 1) ? elements[i].nodes_[j + 1] : elements[i].nodes_[0]};

            if (nodes[0] ==  nodes[1]) { // Shitty workaround for 4-sided boundaries
                for (auto just_any_other_node_index : elements[i].nodes_) {
                    if (just_any_other_node_index != nodes[0]) {
                        nodes[1] = just_any_other_node_index;
                        break;
                    }
                }
            }

            bool found = false;
            for (auto face_index: node_to_face[nodes[0]]) {
                if (((faces[face_index].nodes_[0] == nodes[0]) && (faces[face_index].nodes_[1] == nodes[1])) || ((faces[face_index].nodes_[0] == nodes[1]) && (faces[face_index].nodes_[1] == nodes[0]))) {
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

    // Faces have to be moved, or else this copies the vector, and the device (???) vector copy for face vectors is used, which bad allocs for some reason.
    // 1) Why doesn't this move the vector, as it would be if it was plain returned?
    // 2) Why is the device copy used, it shouldn't be able to be called from that's like the whole point.
    // 3) Why does it bad alloc, the copied face should have its size default-constructed to 0.
    return {std::move(faces), std::move(node_to_face), std::move(element_to_face)};
}

auto SEM::Meshes::Mesh2D_t::initial_conditions(const deviceFloat* polynomial_nodes) -> void {
    initial_conditions_2D<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), nodes_.data(), polynomial_nodes);
}

auto SEM::Meshes::Mesh2D_t::print() -> void {
    std::vector<Face2D_t> host_faces(faces_.size());
    std::vector<Element2D_t> host_elements(elements_.size());
    std::vector<Vec2<deviceFloat>> host_nodes(nodes_.size());
    std::vector<std::array<size_t, 2>> host_interfaces(interfaces_.size());
    std::vector<size_t> host_wall_boundaries(wall_boundaries_.size());
    std::vector<size_t> host_symmetry_boundaries(symmetry_boundaries_.size());
    
    faces_.copy_to(host_faces);
    elements_.copy_to(host_elements);
    nodes_.copy_to(host_nodes);
    interfaces_.copy_to(host_interfaces);
    wall_boundaries_.copy_to(host_wall_boundaries);
    symmetry_boundaries_.copy_to(host_symmetry_boundaries);

    std::cout << "N elements: " << N_elements_ << std::endl;
    std::cout << "N elements and ghosts: " << elements_.size() << std::endl;
    std::cout << "N faces: " << faces_.size() << std::endl;
    std::cout << "N nodes: " << nodes_.size() << std::endl;
    std::cout << "N interfaces: " << interfaces_.size() << std::endl;
    std::cout << "N wall boundaries: " << wall_boundaries_.size() << std::endl;
    std::cout << "N symmetry boundaries: " << symmetry_boundaries_.size() << std::endl;
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
    std::cout << '\t' <<  "Interface origin and destination:" << std::endl;
    for (size_t i = 0; i < host_interfaces.size(); ++i) {
        std::cout << '\t' << '\t' << "interface " << i << " : " << host_interfaces[i][0] << " " << host_interfaces[i][1] << std::endl;
    }
}

auto SEM::Meshes::Mesh2D_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices, const SEM::Helpers::DataWriter_t& data_writer) -> void {
    device_vector<deviceFloat> x(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> y(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> p(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> u(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<deviceFloat> v(N_elements_ * N_interpolation_points * N_interpolation_points);
    device_vector<int> N(N_elements_);

    SEM::Meshes::get_solution<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, N_interpolation_points, elements_.data(), nodes_.data(), interpolation_matrices, x.data(), y.data(), p.data(), u.data(), v.data(), N.data());
    
    std::vector<deviceFloat> x_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> y_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> p_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> u_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<deviceFloat> v_host(N_elements_ * N_interpolation_points * N_interpolation_points);
    std::vector<int> N_host(N_elements_);

    x.copy_to(x_host);
    y.copy_to(y_host);
    p.copy_to(p_host);
    u.copy_to(u_host);
    v.copy_to(v_host);
    N.copy_to(N_host);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    data_writer.write_data(N_interpolation_points, N_elements_, time, global_rank, x_host, y_host, p_host, u_host, v_host, N_host);
}

template auto SEM::Meshes::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const SEM::Entities::NDG_t<SEM::Polynomials::ChebyshevPolynomial_t> &NDG, deviceFloat viscosity, const SEM::Helpers::DataWriter_t& data_writer) -> void; // Get with the times c++, it's crazy I have to do this
template auto SEM::Meshes::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> &NDG, deviceFloat viscosity, const SEM::Helpers::DataWriter_t& data_writer) -> void;

template<typename Polynomial>
auto SEM::Meshes::Mesh2D_t::solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const SEM::Entities::NDG_t<Polynomial> &NDG, deviceFloat viscosity, const SEM::Helpers::DataWriter_t& data_writer) -> void {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    deviceFloat time = 0.0;
    const deviceFloat t_end = output_times.back();
    SEM::Helpers::ProgressBar_t bar;
    size_t timestep = 0;

    deviceFloat delta_t = get_delta_t(CFL);
    if (global_rank == 0) {
        bar.set_status_text("Writing solution");
        bar.update(0.0);
    }
    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_.data(), data_writer);
    if (global_rank == 0) {
        bar.set_status_text("Iteration 0");
        bar.update(0.0);
    }
    
    while (time < t_end) {
        ++timestep;
        delta_t = get_delta_t(CFL);
        if (time + delta_t > t_end) {
            delta_t = t_end - time;
        }

        // Kinda algorithm 62
        deviceFloat t = time;
        SEM::Meshes::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        boundary_conditions();
        //SEM::Meshes::calculate_wave_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces.size(), faces_.data(), elements_.data());
        //SEM::Meshes::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, faces_, NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        //SEM::Meshes::rk3_first_step<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, delta_t, 1.0/3.0);

        t = time + 0.33333333333f * delta_t;
        SEM::Meshes::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        boundary_conditions();
        //SEM::Meshes::calculate_wave_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces.size(), faces_.data(), elements_.data());
        //SEM::Meshes::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, faces_, NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        //SEM::Meshes::rk3_step<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, delta_t, -5.0/9.0, 15.0/16.0);

        t = time + 0.75f * delta_t;
        SEM::Meshes::interpolate_to_boundaries<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        boundary_conditions();
        //SEM::Meshes::calculate_wave_fluxes<<<faces_numBlocks_, faces_blockSize_, 0, stream_>>>(faces.size(), faces_.data(), elements_.data());
        //SEM::Meshes::compute_dg_derivative<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, faces_, NDG.weights_.data(), NDG.derivative_matrices_hat_.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());
        //SEM::Meshes::rk3_step<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_, delta_t, -153.0/128.0, 8.0/15.0);
        
        time += delta_t;
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                //SEM::Meshes::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), NDG.nodes_.data(), NDG.weights_.data());
                if (global_rank == 0) {
                    bar.set_status_text("Writing solution");
                    bar.update(time/t_end);
                }
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_.data(), data_writer);
                break;
            }
        }
        if (global_rank == 0) {
            std::stringstream ss;
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
            bar.update(time/t_end);
        }

        if (timestep % adaptivity_interval_ == 0) {
            //SEM::Meshes::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), NDG.nodes_.data(), NDG.weights_.data());
            adapt(NDG.N_max_, NDG.nodes_.data(), NDG.barycentric_weights_.data());
        }
    }

    bool did_write = false;
    for (auto const& e : std::as_const(output_times)) {
        if ((time >= e) && (time < e + delta_t)) {
            did_write = true;
            break;
        }
    }

    if (!did_write) {
        //SEM::Meshes::estimate_error<Polynomial><<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, elements_.data(), NDG.nodes_.data(), NDG.weights_.data());
        if (global_rank == 0) {
            bar.set_status_text("Writing solution");
            bar.update(1.0);
        }
        write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_.data(), data_writer);
    }
    if (global_rank == 0) {
        bar.set_status_text("Done");
        bar.update(1.0);
    }
    if (global_rank == 0) {
        std::cout << std::endl;
    }
}

__host__ __device__
auto SEM::Meshes::Mesh2D_t::g(Vec2<deviceFloat> xy) -> std::array<deviceFloat, 3> {
    constexpr Vec2<deviceFloat> xy0 {0, 0};
    const Vec2<deviceFloat> k {std::sqrt(static_cast<deviceFloat>(2.0)) / 2, -std::sqrt(static_cast<deviceFloat>(2.0)) / 2};
    const deviceFloat d = 0.2 / (2 * std::sqrt(std::log(2.0)));
    constexpr deviceFloat c = 1;
    
    std::array<deviceFloat, 3> state {
        static_cast<deviceFloat>(std::exp(-((k.x() * (xy.x() - xy0.x()) + k.y() * (xy.y() - xy0.y())) * (k.x() * (xy.x() - xy0.x()) + k.y() * (xy.y() - xy0.y()))) / (d * d))),
        0.0,
        0.0
    };

    state[1] = state[0] * k.x() / c;
    state[2] = state[0] * k.y() / c;

    return state;
}

auto SEM::Meshes::Mesh2D_t::get_delta_t(const deviceFloat CFL) -> deviceFloat {   
    SEM::Meshes::reduce_wave_delta_t<elements_blockSize_/2><<<elements_numBlocks_, elements_blockSize_/2, 0, stream_>>>(CFL, N_elements_, elements_.data(), device_delta_t_array_.data());
    device_delta_t_array_.copy_to(host_delta_t_array_);

    deviceFloat delta_t_min_local = std::numeric_limits<deviceFloat>::infinity();
    for (int i = 0; i < elements_numBlocks_; ++i) {
        delta_t_min_local = min(delta_t_min_local, host_delta_t_array_[i]);
    }

    deviceFloat delta_t_min;
    constexpr MPI_Datatype data_type = (sizeof(deviceFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(&delta_t_min_local, &delta_t_min, 1, data_type, MPI_MIN, MPI_COMM_WORLD);
    return delta_t_min;
}

auto SEM::Meshes::Mesh2D_t::adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) -> void {
    std::cout << "Warning, SEM::Meshes::Mesh2D_t::adapt is not implemented." << std::endl;
}

auto SEM::Meshes::Mesh2D_t::boundary_conditions() -> void {
    SEM::Meshes::local_interfaces<<<interfaces_numBlocks_, boundaries_blockSize_, 0, stream_>>>(interfaces_.size(), elements_.data(), interfaces_.data());

}

__global__
auto SEM::Meshes::allocate_element_storage(size_t n_elements, Element2D_t* elements) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        elements[i].allocate_storage();
    }
}

__global__
auto SEM::Meshes::allocate_face_storage(size_t n_faces, Face2D_t* faces) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_faces; i += stride) {
        faces[i].allocate_storage();
    }
}

__global__
auto SEM::Meshes::fill_element_faces(size_t n_elements, Element2D_t* elements, const std::array<size_t, 4>* element_to_face) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        for (size_t j = 0; j < elements[i].faces_.size(); ++j) {
            elements[i].faces_[j][0] = element_to_face[i][j];
        }
    }
}

__global__
auto SEM::Meshes::compute_face_geometry(size_t n_faces, Face2D_t* faces, const Element2D_t* elements, const Vec2<deviceFloat>* nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_faces; i += stride) {
        SEM::Entities::Face2D_t& face = faces[i];
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

    for (size_t elem_index = index; elem_index < n_elements; elem_index += stride) {
        Element2D_t& element = elements[elem_index];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;
        
        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
                const std::array<Vec2<deviceFloat>, 4> points {nodes[element.nodes_[0]],
                                                               nodes[element.nodes_[1]],
                                                               nodes[element.nodes_[2]],
                                                               nodes[element.nodes_[3]]};
                const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points);

                const std::array<deviceFloat, 3> state = SEM::Meshes::Mesh2D_t::g(global_coordinates);
                element.p_[i * (element.N_ + 1) + j] = state[0];
                element.u_[i * (element.N_ + 1) + j] = state[1];
                element.v_[i * (element.N_ + 1) + j] = state[2];
            }
        }
    }
}

__global__
auto SEM::Meshes::get_solution(size_t N_elements, size_t N_interpolation_points, Element2D_t* elements, const Vec2<deviceFloat>* nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, int* N) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < N_elements; element_index += stride) {
        Element2D_t& element = elements[element_index];
        const size_t offset_interp_2D = element_index * N_interpolation_points * N_interpolation_points;
        const size_t offset_interp = element.N_ * (element.N_ + 1) * N_interpolation_points/2;

        N[element_index] = element.N_;
        const std::array<Vec2<deviceFloat>, 4> points {nodes[element.nodes_[0]],
                                                       nodes[element.nodes_[1]],
                                                       nodes[element.nodes_[2]],
                                                       nodes[element.nodes_[3]]};

        element.interpolate_solution(N_interpolation_points, points, interpolation_matrices + offset_interp, x + offset_interp_2D, y + offset_interp_2D, p + offset_interp_2D, u + offset_interp_2D, v + offset_interp_2D);
    }
}

template __global__ void SEM::Meshes::estimate_error<SEM::Polynomials::ChebyshevPolynomial_t>(size_t N_elements, Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights);
template __global__ void SEM::Meshes::estimate_error<SEM::Polynomials::LegendrePolynomial_t>(size_t N_elements, Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights);

template<typename Polynomial>
__global__
void SEM::Meshes::estimate_error<Polynomial>(size_t N_elements, Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].estimate_error<Polynomial>(polynomial_nodes, weights);
    }
}

__global__
void SEM::Meshes::interpolate_to_boundaries(size_t N_elements, Element2D_t* elements, const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_elements; i += stride) {
        elements[i].interpolate_to_boundaries(lagrange_interpolant_minus, lagrange_interpolant_plus);
    }
}

__global__
void SEM::Meshes::local_interfaces(size_t N_local_interfaces, Element2D_t* elements, const std::array<size_t, 2>* local_interfaces) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_local_interfaces; i += stride) {
        const Element2D_t& source_element = elements[local_interfaces[i][0]];
        Element2D_t& destination_element = elements[local_interfaces[i][1]];

        for (size_t j = 0; j < source_element.faces_.size(); ++j) {
            for (int k = 0; k <= source_element.N_; ++k) {
                destination_element.p_extrapolated_[j][k] = source_element.p_extrapolated_[j][k];
                destination_element.u_extrapolated_[j][k] = source_element.u_extrapolated_[j][k];
                destination_element.v_extrapolated_[j][k] = source_element.v_extrapolated_[j][k];
            }
        }
    }
}

__global__
void SEM::Meshes::calculate_wave_fluxes(size_t N_faces, Face2D_t* faces, const Element2D_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_faces; i += stride) {
        Face2D_t& face = faces[i];

        // Getting element solution
        for (size_t side_index = 0; side_index < face.elements_.size(); ++side_index) {
            const Element2D_t& element = elements[face.elements_[side_index]];

            // Conforming
            if ((face.N_ == element.N_) 
                    && (face.nodes_[0] == element.nodes_[face.elements_side_[side_index]]) 
                    && (face.nodes_[1] == element.nodes_[(face.elements_side_[side_index] + 1) * (!(face.elements_side_[side_index] == (element.nodes_.size() - 1)))])) {
                for (int i = 0; i <= face.N_; ++i) {
                    face.p_[side_index][i] = element.p_extrapolated_[face.elements_side_[side_index]][i];
                    face.u_[side_index][i] = element.u_extrapolated_[face.elements_side_[side_index]][i];
                    face.v_[side_index][i] = element.v_extrapolated_[face.elements_side_[side_index]][i];
                }
            }
            // Conforming, but reversed
            else if ((face.N_ == element.N_) 
                    && (face.nodes_[1] == element.nodes_[face.elements_side_[side_index]]) 
                    && (face.nodes_[0] == element.nodes_[(face.elements_side_[side_index] + 1) * (!(face.elements_side_[side_index] == (element.nodes_.size() - 1)))])) {
                for (int i = 0; i <= face.N_; ++i) {
                    face.p_[side_index][face.N_ - i] = element.p_extrapolated_[face.elements_side_[side_index]][i];
                    face.u_[side_index][face.N_ - i] = element.u_extrapolated_[face.elements_side_[side_index]][i];
                    face.v_[side_index][face.N_ - i] = element.v_extrapolated_[face.elements_side_[side_index]][i];
                }
            }
            else { // We need to interpolate
                printf("Warning, non-conforming surfaces are not implemented yet.\n");
            }
        }

        // Computing fluxes

        /*deviceFloat u;
        const deviceFloat u_left = elements[faces[i].elements_[0]].phi_R_;
        const deviceFloat u_right = elements[faces[i].elements_[1]].phi_L_;

        if (u_left < 0.0f && u_right > 0.0f) { // In expansion fan
            u = 0.5f * (u_left + u_right);
        }
        else if (u_left >= u_right) { // Shock
            if (u_left > 0.0f) {
                u = u_left;
            }
            else {
                u = u_right;
            }
        }
        else { // Expansion fan
            if (u_left > 0.0f) {
                u = u_left;
            }
            else  {
                u = u_right;
            }
        }
    
        faces[i].flux_ = u_right;
        faces[i].nl_flux_ = 0.5f * u * u;*/
    }
}
