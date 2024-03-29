#include "meshes/Mesh2D_t.h"
#include "helpers/constants.h"
#include "functions/Utilities.h"
#include "functions/Hilbert_splitting.h"
#include "functions/quad_map.h"
#include "functions/analytical_solution.h"
#ifdef NDG_USE_CGNS
    #include "cgnslib.h"
#endif
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <cmath>

namespace fs = std::filesystem;

using SEM::Host::Entities::Vec2;
using SEM::Host::Entities::Element2D_t;
using SEM::Host::Entities::Face2D_t;
using namespace SEM::Host::Hilbert;

constexpr int CGIO_MAX_NAME_LENGTH = 33; // Includes the null terminator
constexpr hostFloat pi = 3.14159265358979323846;

SEM::Host::Meshes::Mesh2D_t::Mesh2D_t(
        std::filesystem::path filename, 
        int initial_N, 
        int maximum_N, 
        size_t n_interpolation_points, 
        int max_split_level, 
        size_t adaptivity_interval, 
        size_t load_balancing_interval, 
        hostFloat tolerance_min, 
        hostFloat tolerance_max, 
        hostFloat load_balancing_threshold, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes,
        std::vector<hostFloat> worker_weights) :       
            initial_N_{initial_N},  
            maximum_N_{maximum_N},
            n_interpolation_points_{n_interpolation_points},
            max_split_level_{max_split_level},
            adaptivity_interval_{adaptivity_interval},
            load_balancing_interval_{load_balancing_interval},
            tolerance_min_{tolerance_min},
            tolerance_max_{tolerance_max},
            load_balancing_threshold_{load_balancing_threshold},
            worker_weights_{worker_weights} {

    total_weight_ = hostFloat{0};
    for (const auto weight : worker_weights_) {
        total_weight_ += weight;
    }

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

    compute_element_geometry(n_elements_, elements_, nodes_, polynomial_nodes);
    compute_boundary_geometry(n_elements_, elements_, nodes_, polynomial_nodes);
    compute_element_status(n_elements_, elements_, nodes_);
    compute_face_geometry(faces_, elements_, nodes_);
}

auto SEM::Host::Meshes::Mesh2D_t::read_su2(std::filesystem::path filename) -> void {
    std::string line;
    std::string token;
    size_t value;

    std::ifstream meshfile(filename);
    if (!meshfile.is_open()) {
        std::cerr << "Error: file '" << filename << "' could not be opened. Exiting." << std::endl;
        exit(69);
    }

    do {
        std::getline(meshfile, line);  
    }
    while (line.empty());

    std::istringstream liness(line);
    liness >> token;
    liness >> value;
    if (token != "NDIME=") {
        std::cerr << "Error: first token should be 'NDIME=', found '" << token << "'. Exiting." << std::endl;
        exit(70);
    }

    if (value != 2) {
        std::cerr << "Error: program only works for 2 dimensions, found '" << value << "'. Exiting." << std::endl;
        exit(71);
    }

    std::vector<std::array<size_t, 4>> elements;
    std::vector<std::array<size_t, 2>> wall;
    std::vector<std::array<size_t, 2>> symmetry;
    std::vector<std::array<size_t, 2>> inflow;
    std::vector<std::array<size_t, 2>> outflow;

    while (!meshfile.eof()) {
        do {
            std::getline(meshfile, line);  
        }
        while (line.empty() && !meshfile.eof());

        std::istringstream liness(line);
        liness >> token;
        std::transform(token.begin(), token.end(), token.begin(),
            [](unsigned char c){ return std::toupper(c); });

        if (token == "NPOIN=") {
            liness >> value;
            nodes_ = std::vector<Vec2<hostFloat>>(value);

            for (size_t i = 0; i < nodes_.size(); ++i) {
                std::getline(meshfile, line);
                std::istringstream liness2(line);
                liness2 >> nodes_[i].x() >> nodes_[i].y();
            }
        }
        else if (token == "NELEM=") {
            liness >> value;
            elements = std::vector<std::array<size_t, 4>>(value);

            for (size_t i = 0; i < elements.size(); ++i) {
                std::getline(meshfile, line);
                std::istringstream liness2(line);
                liness2 >> token;
                if (token == "9") {
                    liness2 >> elements[i][0] >> elements[i][1] >> elements[i][2] >> elements[i][3];
                }
                else {
                    std::cerr << "Error: expected token '9', found '" << token << "'. Exiting." << std::endl;
                    exit(73);
                }
            }
        }
        else if (token == "NMARK=") {
            size_t n_markers;
            liness >> n_markers;

            size_t n_wall = 0;
            size_t n_symmetry = 0;
            size_t n_inflow = 0;
            size_t n_outflow = 0;

            for (size_t i = 0; i < n_markers; ++i) {
                std::string type;
                do {
                    std::getline(meshfile, line);
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
                        std::getline(meshfile, line);
                        if (!line.empty()) {
                            std::istringstream liness(line);
                            liness >> token;
                            liness >> value;
                        }   
                    }
                    while (token != "MARKER_ELEMS=");

                    n_outflow += value;
                    outflow.reserve(n_outflow);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(meshfile, line);
                        std::istringstream liness6(line);

                        liness6 >> token;
                        if (token != "3") {
                            std::cerr << "Error: expected token '3', found '" << token << "'. Exiting." << std::endl;
                            exit(75);
                        }

                        size_t val0, val1;
                        liness6 >> val0 >> val1;
                        outflow.push_back({val0, val1});
                    }
                }
                else if (type == "wall" || type == "airfoil") {
                    do {
                        std::getline(meshfile, line);
                        if (!line.empty()) {
                            std::istringstream liness(line);
                            liness >> token;
                            liness >> value;
                        }   
                    }
                    while (token != "MARKER_ELEMS=");

                    n_wall += value;
                    wall.reserve(n_wall);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(meshfile, line);
                        std::istringstream liness6(line);

                        liness6 >> token;
                        if (token != "3") {
                            std::cerr << "Error: expected token '3', found '" << token << "'. Exiting." << std::endl;
                            exit(75);
                        }

                        size_t val0, val1;
                        liness6 >> val0 >> val1;
                        wall.push_back({val0, val1});
                    }
                }
                else if (type == "inlet") {
                    do {
                        std::getline(meshfile, line);
                        if (!line.empty()) {
                            std::istringstream liness(line);
                            liness >> token;
                            liness >> value;
                        }   
                    }
                    while (token != "MARKER_ELEMS=");

                    n_inflow += value;
                    inflow.reserve(n_inflow);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(meshfile, line);
                        std::istringstream liness6(line);

                        liness6 >> token;
                        if (token != "3") {
                            std::cerr << "Error: expected token '3', found '" << token << "'. Exiting." << std::endl;
                            exit(75);
                        }

                        size_t val0, val1;
                        liness6 >> val0 >> val1;
                        inflow.push_back({val0, val1});
                    }
                }
                else if (type == "symmetry") {
                    do {
                        std::getline(meshfile, line);
                        if (!line.empty()) {
                            std::istringstream liness(line);
                            liness >> token;
                            liness >> value;
                        }   
                    }
                    while (token != "MARKER_ELEMS=");

                    n_symmetry += value;
                    symmetry.reserve(n_symmetry);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(meshfile, line);
                        std::istringstream liness6(line);

                        liness6 >> token;
                        if (token != "3") {
                            std::cerr << "Error: expected token '3', found '" << token << "'. Exiting." << std::endl;
                            exit(75);
                        }

                        size_t val0, val1;
                        liness6 >> val0 >> val1;
                        symmetry.push_back({val0, val1});
                    }
                }
                else {
                    std::cerr << "Error: expected marker tag 'farfield', 'wall', 'symmetry' or 'inlet', found '" << type << "'. Exiting." << std::endl;
                    exit(74);
                }
            }
        }
        else {
            if (!meshfile.eof()) {
                std::cerr << "Error: expected marker 'NPOIN=', 'NELEM=' or 'NMARK=', found '" << token << "'. Exiting." << std::endl;
                exit(72);
            }
        }
    }

    meshfile.close();

    // Building elements
    const size_t n_elements_domain = elements.size();
    const size_t n_elements_ghost = wall.size() + symmetry.size() + inflow.size() + outflow.size();
    const size_t n_elements = n_elements_domain + n_elements_ghost;

    elements_ = std::vector<Element2D_t>(n_elements);
    for (size_t i = 0; i < elements.size(); ++i) {
        Element2D_t& element = elements_[i];
        element.N_ = initial_N_;
        element.nodes_ = elements[i];       
    }

    for (size_t i = 0; i < wall.size(); ++i) {
        Element2D_t& element = elements_[n_elements_domain + i];
        element.N_ = initial_N_;
        element.nodes_ = {wall[i][0],
                          wall[i][1],
                          wall[i][1],
                          wall[i][0]};
        element.status_ = Hilbert::Status::A; // This is not a random status, when splitting the first two elements are 0 and 1, which is needed for boundaries
        element.rotation_ = 0;
    }

    for (size_t i = 0; i < symmetry.size(); ++i) {
        Element2D_t& element = elements_[n_elements_domain + wall.size() + i];
        element.N_ = initial_N_;
        element.nodes_ = {symmetry[i][0],
                          symmetry[i][1],
                          symmetry[i][1],
                          symmetry[i][0]};
        element.status_ = Hilbert::Status::A; // This is not a random status, when splitting the first two elements are 0 and 1, which is needed for boundaries
        element.rotation_ = 0;
    }

    for (size_t i = 0; i < inflow.size(); ++i) {
        Element2D_t& element = elements_[n_elements_domain + wall.size() + symmetry.size() + i];
        element.N_ = initial_N_;
        element.nodes_ = {inflow[i][0],
                          inflow[i][1],
                          inflow[i][1],
                          inflow[i][0]};
        element.status_ = Hilbert::Status::A; // This is not a random status, when splitting the first two elements are 0 and 1, which is needed for boundaries
        element.rotation_ = 0;
    }

    for (size_t i = 0; i < outflow.size(); ++i) {
        Element2D_t& element = elements_[n_elements_domain + wall.size() + symmetry.size() + inflow.size() + i];
        element.N_ = initial_N_;
        element.nodes_ = {outflow[i][0],
                          outflow[i][1],
                          outflow[i][1],
                          outflow[i][0]};
        element.status_ = Hilbert::Status::A; // This is not a random status, when splitting the first two elements are 0 and 1, which is needed for boundaries
        element.rotation_ = 0;
    }

    // Computing nodes to elements
    const std::vector<std::vector<size_t>> node_to_element = build_node_to_element(nodes_.size(), elements_);

    // Computing element to elements
    const std::vector<std::vector<size_t>> element_to_element = build_element_to_element(elements_, node_to_element);

    // Computing faces and filling element faces
    auto [host_faces, node_to_face, element_to_face] = build_faces(n_elements_domain, nodes_.size(), initial_N_, elements_);
    faces_ = std::move(host_faces);

    // Building boundaries
    wall_boundaries_ = std::vector<size_t>(wall.size());
    symmetry_boundaries_ = std::vector<size_t>(symmetry.size());
    inflow_boundaries_ = std::vector<size_t>(inflow.size());
    outflow_boundaries_ = std::vector<size_t>(outflow.size());

    for (size_t i = 0; i < wall_boundaries_.size(); ++i) {
        wall_boundaries_[i] = n_elements_domain + i;
    }

    for (size_t i = 0; i < symmetry_boundaries_.size(); ++i) {
        symmetry_boundaries_[i] = n_elements_domain + wall_boundaries_.size() + i;
    }

    for (size_t i = 0; i < inflow_boundaries_.size(); ++i) {
        inflow_boundaries_[i] = n_elements_domain + wall_boundaries_.size() + symmetry_boundaries_.size() + i;
    }

    for (size_t i = 0; i < outflow_boundaries_.size(); ++i) {
        outflow_boundaries_[i] = n_elements_domain + wall_boundaries_.size() + symmetry_boundaries_.size() + inflow_boundaries_.size() + i;
    }

    // No self-interfaces
    
    // No mpi interfaces

    // Setting sizes
    n_elements_ = n_elements_domain;
    global_element_offset_ = 0;
    n_elements_global_ = n_elements_;

    allocate_element_storage(n_elements_, elements_);
    allocate_boundary_storage(n_elements_, elements_);
    allocate_face_storage(faces_);

    fill_element_faces(n_elements_, elements_, element_to_face);
    fill_boundary_element_faces(n_elements_, elements_, element_to_face);

    // Allocating output arrays
    x_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    y_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    p_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    u_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    v_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
}

auto SEM::Host::Meshes::Mesh2D_t::read_cgns(std::filesystem::path filename) -> void {
    #ifdef NDG_USE_CGNS
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
    nodes_ = std::vector<Vec2<hostFloat>>(n_nodes);
    for (cgsize_t i = 0; i < n_nodes; ++i) {
        nodes_[i].x() = xy[0][i];
        nodes_[i].y() = xy[1][i];
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
    elements_ = std::vector<Element2D_t>(n_elements);
    std::vector<size_t> section_start_indices(n_sections);
    size_t element_domain_index = 0;
    size_t element_ghost_index = n_elements_domain;
    for (int i = 0; i < n_sections; ++i) {
        if (section_is_domain[i]) {
            section_start_indices[i] = element_domain_index;
            for (cgsize_t j = 0; j < section_ranges[i][1] - section_ranges[i][0] + 1; ++j) {
                Element2D_t& element = elements_[section_start_indices[i] + j];
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
                Element2D_t& element = elements_[section_start_indices[i] + j];
                element.N_ = initial_N_;
                element.nodes_ = {static_cast<size_t>(connectivity[i][2 * j] - 1),
                                  static_cast<size_t>(connectivity[i][2 * j + 1] - 1),
                                  static_cast<size_t>(connectivity[i][2 * j + 1] - 1),
                                  static_cast<size_t>(connectivity[i][2 * j] - 1)};
                element.status_ = Hilbert::Status::A; // This is not a random status, when splitting the first two elements are 0 and 1, which is needed for boundaries
                element.rotation_ = 0;
            }
            element_ghost_index += section_ranges[i][1] - section_ranges[i][0] + 1;
        }
    }

    // Computing nodes to elements
    const std::vector<std::vector<size_t>> node_to_element = build_node_to_element(n_nodes, elements_);

    // Computing element to elements
    const std::vector<std::vector<size_t>> element_to_element = build_element_to_element(elements_, node_to_element);

    // Computing faces and filling element faces
    auto [host_faces, node_to_face, element_to_face] = build_faces(n_elements_domain, n_nodes, initial_N_, elements_);
    faces_ = std::move(host_faces);

    // Building boundaries
    for (int i = 0; i < n_boundaries; ++i) {
        switch (boundary_types[i]) {
            case BCType_t::BCWall:
                wall_boundaries_.reserve(wall_boundaries_.size() + boundary_sizes[i]);
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

                    wall_boundaries_.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
                }
                break;

            case BCType_t::BCSymmetryPlane:
                symmetry_boundaries_.reserve(symmetry_boundaries_.size() + boundary_sizes[i]);
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

                    symmetry_boundaries_.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
                }
                
                break;

            case BCType_t::BCInflow:
                inflow_boundaries_.reserve(inflow_boundaries_.size() + boundary_sizes[i]);
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

                    inflow_boundaries_.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
                }

                break;
            
            case BCType_t::BCOutflow:
                outflow_boundaries_.reserve(outflow_boundaries_.size() + boundary_sizes[i]);
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

                    outflow_boundaries_.push_back(section_start_indices[section_index] + boundary_elements[i][j] - section_ranges[section_index][0]);
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

    if (n_interface_elements > 0) {
        interfaces_origin_ = std::vector<size_t>(n_interface_elements);
        interfaces_origin_side_ = std::vector<size_t>(n_interface_elements);
        interfaces_destination_ = std::vector<size_t>(n_interface_elements);

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
                    const size_t face_side_index = faces_[face_index].elements_[0] == donor_boundary_element_index;
                    const size_t donor_domain_element_index = faces_[face_index].elements_[face_side_index];
                    
                    interfaces_origin_[interface_start_index[i] + j] = donor_domain_element_index;
                    interfaces_origin_side_[interface_start_index[i] + j] = faces_[face_index].elements_side_[face_side_index];
                    interfaces_destination_[interface_start_index[i] + j] = section_start_indices[origin_section_index] + interface_elements[i][j] - section_ranges[origin_section_index][0];
                }
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

    if (n_mpi_interface_elements > 0) {
        size_t n_mpi_interfaces = 0;
        for (int j = 0; j < n_zones; ++j) {
            n_mpi_interfaces += process_used_in_interface[j];
        }

        mpi_interfaces_outgoing_size_ = std::vector<size_t>(n_mpi_interfaces);
        mpi_interfaces_incoming_size_ = std::vector<size_t>(n_mpi_interfaces);
        mpi_interfaces_outgoing_offset_ = std::vector<size_t>(n_mpi_interfaces);
        mpi_interfaces_incoming_offset_ = std::vector<size_t>(n_mpi_interfaces);
        mpi_interfaces_process_ = std::vector<int>(n_mpi_interfaces);
        mpi_interfaces_origin_ = std::vector<size_t>(n_mpi_interface_elements);
        mpi_interfaces_origin_side_ = std::vector<size_t>(n_mpi_interface_elements);
        std::vector<size_t> mpi_interfaces_destination(n_mpi_interface_elements);
        mpi_interfaces_destination_ = std::vector<size_t>(n_mpi_interface_elements);

        size_t mpi_interface_offset = 0;
        size_t mpi_interface_index = 0;
        for (int j = 0; j < n_zones; ++j) {
            if (process_used_in_interface[j]) {
                mpi_interfaces_outgoing_offset_[mpi_interface_index] = mpi_interface_offset;
                mpi_interfaces_incoming_offset_[mpi_interface_index] = mpi_interface_offset;
                mpi_interfaces_process_[mpi_interface_index] = j;
                for (int i = 0; i < n_connectivity; ++i) {
                    if (mpi_interface_process[i] == j) {
                        mpi_interfaces_outgoing_size_[mpi_interface_index] += connectivity_sizes[i];
                        mpi_interfaces_incoming_size_[mpi_interface_index] += connectivity_sizes[i];
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
                            const size_t face_side_index = faces_[face_index].elements_[0] == boundary_element_index;

                            mpi_interfaces_origin_[mpi_interface_offset + k]      = faces_[face_index].elements_[face_side_index];;
                            mpi_interfaces_origin_side_[mpi_interface_offset + k] = faces_[face_index].elements_side_[face_side_index];
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
        const MPI_Datatype size_t_data_type = (sizeof(size_t) == sizeof(unsigned long long)) ? MPI_UNSIGNED_LONG_LONG : (sizeof(size_t) == sizeof(unsigned long)) ? MPI_UNSIGNED_LONG : MPI_UNSIGNED; // CHECK this is a bad way of doing this

        for (size_t i = 0; i < n_mpi_interfaces; ++i) {
            MPI_Isend(mpi_interfaces_destination.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], size_t_data_type, mpi_interfaces_process_[i], global_size * global_rank + mpi_interfaces_process_[i], MPI_COMM_WORLD, &adaptivity_requests[n_mpi_interfaces + i]);
            MPI_Irecv(mpi_interfaces_destination_.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], size_t_data_type, mpi_interfaces_process_[i],  global_size * mpi_interfaces_process_[i] + global_rank, MPI_COMM_WORLD, &adaptivity_requests[i]);
        }

        MPI_Waitall(n_mpi_interfaces, adaptivity_requests.data(), adaptivity_statuses.data());

        for (size_t i = 0; i < n_mpi_interfaces; ++i) {
            for (size_t j = 0; j < mpi_interfaces_incoming_size_[i]; ++j) {
                int donor_section_index = -1;
                for (int k = 0; k < n_sections; ++k) {
                    if ((mpi_interfaces_destination_[mpi_interfaces_incoming_offset_[i] + j] >= section_ranges[k][0]) && (mpi_interfaces_destination_[mpi_interfaces_incoming_offset_[i] + j] <= section_ranges[k][1])) {
                        donor_section_index = k;
                        break;
                    }
                }

                if (donor_section_index == -1) {
                    std::cerr << "Error: Process " << mpi_interfaces_process_[i] << " sent element " << mpi_interfaces_destination_[mpi_interfaces_incoming_offset_[i] + j] << " to process " << global_rank << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(51);
                }

                mpi_interfaces_destination_[mpi_interfaces_incoming_offset_[i] + j] = section_start_indices[donor_section_index] + mpi_interfaces_destination_[mpi_interfaces_incoming_offset_[i] + j] - section_ranges[donor_section_index][0];
            }
        }

        MPI_Waitall(n_mpi_interfaces, adaptivity_requests.data() + n_mpi_interfaces, adaptivity_statuses.data() + n_mpi_interfaces);

        // Transferring onto the GPU
        interfaces_N_ = std::vector<int>(mpi_interfaces_origin_.size());
        interfaces_p_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
        interfaces_u_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
        interfaces_v_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
        interfaces_refine_ = std::make_unique<bool[]>(mpi_interfaces_origin_.size());
        receiving_interfaces_p_ = std::vector<hostFloat>(mpi_interfaces_destination.size() * (maximum_N_ + 1));
        receiving_interfaces_u_ = std::vector<hostFloat>(mpi_interfaces_destination.size() * (maximum_N_ + 1));
        receiving_interfaces_v_ = std::vector<hostFloat>(mpi_interfaces_destination.size() * (maximum_N_ + 1));
        receiving_interfaces_N_ = std::vector<int>(mpi_interfaces_destination.size());
        receiving_interfaces_refine_ = std::make_unique<bool[]>(mpi_interfaces_destination.size());
        receiving_interfaces_refine_without_splitting_ = std::make_unique<bool[]>(mpi_interfaces_destination.size());
        receiving_interfaces_creating_node_ = std::make_unique<bool[]>(mpi_interfaces_destination.size());
        interfaces_refine_without_splitting_ = std::make_unique<bool[]>(mpi_interfaces_origin_.size());

        requests_ = std::vector<MPI_Request>(n_mpi_interfaces * 2 * mpi_boundaries_n_transfers);
        statuses_ = std::vector<MPI_Status>(requests_.size());
        requests_adaptivity_ = std::vector<MPI_Request>(n_mpi_interfaces * 2 * mpi_adaptivity_split_n_transfers);
        statuses_adaptivity_ = std::vector<MPI_Status>(requests_adaptivity_.size());
    }

    // Setting sizes
    n_elements_ = n_elements_domain;

    // Sharing number of elements to calculate offset
    std::vector<size_t> n_elements_per_process(global_size);
    const MPI_Datatype size_t_data_type = (sizeof(size_t) == sizeof(unsigned long long)) ? MPI_UNSIGNED_LONG_LONG : (sizeof(size_t) == sizeof(unsigned long)) ? MPI_UNSIGNED_LONG : MPI_UNSIGNED; // CHECK this is a bad way of doing this
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

    allocate_element_storage(n_elements_, elements_);
    allocate_boundary_storage(n_elements_, elements_);
    allocate_face_storage(faces_);

    fill_element_faces(n_elements_, elements_, element_to_face);
    fill_boundary_element_faces(n_elements_, elements_, element_to_face);

    // Allocating output arrays
    x_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    y_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    p_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    u_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    v_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    #else
    std::cerr << "Error: Program was launched with a cgns mesh, but it was compiled without cgns support. Exiting." << std::endl;
    exit(76);
    #endif
}

auto SEM::Host::Meshes::Mesh2D_t::build_node_to_element(size_t n_nodes, const std::vector<Element2D_t>& elements) -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> node_to_element(n_nodes);

    for (size_t j = 0; j < elements.size(); ++j) {
        for (auto node_index: elements[j].nodes_) {
            if (std::find(node_to_element[node_index].begin(), node_to_element[node_index].end(), j) == node_to_element[node_index].end()) { // CHECK This will be slower, but is needed because boundaries have 4 sides and not 2. Remove when variable geometry elements are added.
                node_to_element[node_index].push_back(j);
            }
        }
    }

    return node_to_element;
}

auto SEM::Host::Meshes::Mesh2D_t::build_element_to_element(
        const std::vector<Element2D_t>& elements, 
        const std::vector<std::vector<size_t>>& node_to_element
        ) -> std::vector<std::vector<size_t>> {
    
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

auto SEM::Host::Meshes::Mesh2D_t::build_faces(
        size_t n_elements_domain, 
        size_t n_nodes, 
        int initial_N, 
        std::vector<Element2D_t>& elements
        ) -> std::tuple<std::vector<Face2D_t>, std::vector<std::vector<size_t>>, std::vector<std::array<size_t, 4>>> {
    
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
            if ((faces[face_index].nodes_[0] == nodes[0]) && (faces[face_index].nodes_[1] == nodes[1])) {
                faces[face_index].elements_[1] = i;
                faces[face_index].elements_side_[1] = 0;
                element_to_face[i][0] = face_index;
                for (size_t j = 1; j < element_to_face[i].size(); ++j) {
                    element_to_face[i][j] = static_cast<size_t>(-1);
                }
                std::swap(elements[i].nodes_[0], elements[i].nodes_[1]); // This is needed for "inside out" boundary elements, sometimes present in su2 meshes. 
                                                                         // The element array is not const anymore. Should not create a race condition, as each boundary element should only have one face.
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

auto SEM::Host::Meshes::Mesh2D_t::initial_conditions(const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void {
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        Element2D_t& element = elements_[element_index];
        const std::array<Vec2<hostFloat>, 4> points {nodes_[element.nodes_[0]],
                                                     nodes_[element.nodes_[1]],
                                                     nodes_[element.nodes_[2]],
                                                     nodes_[element.nodes_[3]]};
        
        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<hostFloat> coordinates(polynomial_nodes[element.N_][i], polynomial_nodes[element.N_][j]);
                const Vec2<hostFloat> global_coordinates = SEM::Host::quad_map(coordinates, points);

                const std::array<hostFloat, 3> state = SEM::Host::g(global_coordinates, hostFloat(0));
                element.p_[i * (element.N_ + 1) + j] = state[0];
                element.u_[i * (element.N_ + 1) + j] = state[1];
                element.v_[i * (element.N_ + 1) + j] = state[2];

                element.G_p_[i * (element.N_ + 1) + j] = 0;
                element.G_u_[i * (element.N_ + 1) + j] = 0;
                element.G_v_[i * (element.N_ + 1) + j] = 0;
            }
        }
    }
}

auto SEM::Host::Meshes::Mesh2D_t::print() const -> void {
    std::cout << "N elements: " << n_elements_ << std::endl;
    std::cout << "N elements and ghosts: " << elements_.size() << std::endl;
    std::cout << "N faces: " << faces_.size() << std::endl;
    std::cout << "N nodes: " << nodes_.size() << std::endl;
    std::cout << "N wall boundaries: " << wall_boundaries_.size() << std::endl;
    std::cout << "N symmetry boundaries: " << symmetry_boundaries_.size() << std::endl;
    std::cout << "N inflow boundaries: " << inflow_boundaries_.size() << std::endl;
    std::cout << "N outflow boundaries: " << outflow_boundaries_.size() << std::endl;
    std::cout << "N interfaces: " << interfaces_origin_.size() << std::endl;
    std::cout << "N mpi incoming interfaces: " << mpi_interfaces_destination_.size() << std::endl;
    std::cout << "N mpi outgoing interfaces: " << mpi_interfaces_origin_.size() << std::endl;
    std::cout << "Initial N: " << initial_N_ << std::endl;

    std::cout << std::endl <<  "Connectivity" << std::endl;
    std::cout << '\t' <<  "Nodes:" << std::endl;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        std::cout << '\t' << '\t' << "node " << std::setw(6) << i << " : " << std::setw(6) << nodes_[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Element nodes:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << std::setw(6) << i << " : ";
        for (auto node_index : elements_[i].nodes_) {
            std::cout << std::setw(6) << node_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face nodes:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : ";
        for (auto node_index : faces_[i].nodes_) {
            std::cout << std::setw(6) << node_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face elements:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : ";
        for (auto element_index : faces_[i].elements_) {
            std::cout << std::setw(6) << element_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face elements side:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : ";
        for (auto side_index : faces_[i].elements_side_) {
            std::cout << side_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl <<  "Geometry" << std::endl;
    std::cout << '\t' <<  "Element Hilbert status:" << std::endl;
    constexpr std::array<char, 4> status_letter {'H', 'A', 'R', 'B'};
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << std::setw(6) << i << " : " << status_letter[elements_[i].status_] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Element rotation:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << std::setw(6) << i << " : " << elements_[i].rotation_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Element min length:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << std::setw(6) << i << " : " << elements_[i].delta_xy_min_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Element N:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << '\t' << "element " << std::setw(6) << i << " : " << elements_[i].N_ << std::endl;
    }
    
    std::cout << std::endl << '\t' <<  "Face N:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].N_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face length:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].length_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face normal:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].normal_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face tangent:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].tangent_ << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face offset:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : " << std::setw(6) << faces_[i].offset_[0] << " " << std::setw(6) << faces_[i].offset_[1] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face scale:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : " << std::setw(6) << faces_[i].scale_[0] << " " << std::setw(6) << faces_[i].scale_[1] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Face refine:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].refine_ << std::endl;
    }

    std::cout << std::endl <<  "Interfaces" << std::endl;
    std::cout << '\t' <<  "Interface destination, origin and origin side:" << std::endl;
    for (size_t i = 0; i < interfaces_origin_.size(); ++i) {
        std::cout << '\t' << '\t' << "interface " << std::setw(6) << i << " : " << std::setw(6) << interfaces_destination_[i] << " " << std::setw(6) << interfaces_origin_[i] << " " << std::setw(6) << interfaces_origin_side_[i] << std::endl;
    }

    std::cout <<  "MPI interfaces:" << std::endl;
    for (size_t j = 0; j < mpi_interfaces_process_.size(); ++j) {
        std::cout << '\t' << "MPI interface to process " << mpi_interfaces_process_[j] << " of outgoing size " << mpi_interfaces_outgoing_size_[j] << ", incoming size " << mpi_interfaces_incoming_size_[j] << ", outgoing offset " << mpi_interfaces_outgoing_offset_[j] << " and incoming offset " << mpi_interfaces_incoming_offset_[j] << ":" << std::endl;
        std::cout << '\t' << '\t' << "Outgoing element and element side:" << std::endl;
        for (size_t i = 0; i < mpi_interfaces_outgoing_size_[j]; ++i) {
            std::cout << '\t' << '\t' << '\t' << "mpi interface " << std::setw(6) << i << " : " << std::setw(6) << mpi_interfaces_origin_[mpi_interfaces_outgoing_offset_[j] + i] << " " << std::setw(6) << mpi_interfaces_origin_side_[mpi_interfaces_outgoing_offset_[j] + i] << std::endl;
        }
        std::cout << '\t' << '\t' << "Incoming element:" << std::endl;
        for (size_t i = 0; i < mpi_interfaces_incoming_size_[j]; ++i) {
            std::cout << '\t' << '\t' << '\t' << "mpi interface " << std::setw(6) << i << " : " << std::setw(6) << mpi_interfaces_destination_[mpi_interfaces_incoming_offset_[j] + i] << std::endl;
        }
    }

    std::cout << std::endl << '\t' <<  "Wall boundaries:" << std::endl;
    for (size_t i = 0; i < wall_boundaries_.size(); ++i) {
        std::cout << '\t' << '\t' << "wall " << std::setw(6) << i << " : " << std::setw(6) << wall_boundaries_[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Symmetry boundaries:" << std::endl;
    for (size_t i = 0; i < symmetry_boundaries_.size(); ++i) {
        std::cout << '\t' << '\t' << "symmetry " << std::setw(6) << i << " : " << std::setw(6) << symmetry_boundaries_[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Inflow boundaries:" << std::endl;
    for (size_t i = 0; i < inflow_boundaries_.size(); ++i) {
        std::cout << '\t' << '\t' << "inflow " << std::setw(6) << i << " : " << std::setw(6) << inflow_boundaries_[i] << std::endl;
    }

    std::cout << std::endl << '\t' <<  "Outflow boundaries:" << std::endl;
    for (size_t i = 0; i < outflow_boundaries_.size(); ++i) {
        std::cout << '\t' << '\t' << "outflow " << std::setw(6) << i << " : " << std::setw(6) << outflow_boundaries_[i] << std::endl;
    }

    std::cout << std::endl <<  "Element faces:" << std::endl;
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        const Element2D_t& element = elements_[element_index];
        std::cout << '\t' << "Element " << element_index << " has:"; // CHECK only works for quadrilaterals
        std::cout << std::endl << '\t' << '\t' << "Bottom faces: "; 
        for (auto face_index : element.faces_[0]) {
            std::cout << face_index << " ";
        }
        std::cout << std::endl << '\t' << '\t' << "Right faces: ";
        for (auto face_index : element.faces_[1]) {
            std::cout << face_index << " ";
        }
        std::cout << std::endl << '\t' << '\t' << "Top faces: ";
        for (auto face_index : element.faces_[2]) {
            std::cout << face_index << " ";
        }
        std::cout << std::endl << '\t' << '\t' << "Left faces: ";
        for (auto face_index : element.faces_[3]) {
            std::cout << face_index << " ";
        }
        std::cout << std::endl;
    }

    for (size_t element_index = n_elements_; element_index < elements_.size(); ++element_index) {
        const Element2D_t& element = elements_[element_index];
        std::cout << '\t' << "Element " << element_index << " has:" << std::endl << '\t' << "Faces: "; // CHECK only works for quadrilaterals
        for (auto face_index : element.faces_[0]) {
            std::cout << face_index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

auto SEM::Host::Meshes::Mesh2D_t::print_to_file(std::filesystem::path filename) const -> void {
    if (filename.has_parent_path()) {
        fs::create_directory(filename.parent_path());
    }
    std::ofstream meshfile(filename);
    if (!meshfile.is_open()) {
        std::cerr << "Error: file '" << filename << "' could not be opened. Exiting." << std::endl;
        exit(78);
    }

    meshfile << "N elements: " << n_elements_ << std::endl;
    meshfile << "N elements and ghosts: " << elements_.size() << std::endl;
    meshfile << "N faces: " << faces_.size() << std::endl;
    meshfile << "N nodes: " << nodes_.size() << std::endl;
    meshfile << "N wall boundaries: " << wall_boundaries_.size() << std::endl;
    meshfile << "N symmetry boundaries: " << symmetry_boundaries_.size() << std::endl;
    meshfile << "N inflow boundaries: " << inflow_boundaries_.size() << std::endl;
    meshfile << "N outflow boundaries: " << outflow_boundaries_.size() << std::endl;
    meshfile << "N interfaces: " << interfaces_origin_.size() << std::endl;
    meshfile << "N mpi incoming interfaces: " << mpi_interfaces_destination_.size() << std::endl;
    meshfile << "N mpi outgoing interfaces: " << mpi_interfaces_origin_.size() << std::endl;
    meshfile << "Initial N: " << initial_N_ << std::endl;

    meshfile << std::endl <<  "Connectivity" << std::endl;
    meshfile << '\t' <<  "Nodes:" << std::endl;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        meshfile << '\t' << '\t' << "node " << std::setw(6) << i << " : " << std::setw(6) << nodes_[i] << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Element nodes:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        meshfile << '\t' << '\t' << "element " << std::setw(6) << i << " : ";
        for (auto node_index : elements_[i].nodes_) {
            meshfile << std::setw(6) << node_index << " ";
        }
        meshfile << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face nodes:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : ";
        for (auto node_index : faces_[i].nodes_) {
            meshfile << std::setw(6) << node_index << " ";
        }
        meshfile << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face elements:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : ";
        for (auto element_index : faces_[i].elements_) {
            meshfile << std::setw(6) << element_index << " ";
        }
        meshfile << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face elements side:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : ";
        for (auto side_index : faces_[i].elements_side_) {
            meshfile << side_index << " ";
        }
        meshfile << std::endl;
    }

    meshfile << std::endl <<  "Geometry" << std::endl;
    meshfile << '\t' <<  "Element Hilbert status:" << std::endl;
    constexpr std::array<char, 4> status_letter {'H', 'A', 'R', 'B'};
    for (size_t i = 0; i < elements_.size(); ++i) {
        meshfile << '\t' << '\t' << "element " << std::setw(6) << i << " : " << status_letter[elements_[i].status_] << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Element rotation:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        meshfile << '\t' << '\t' << "element " << std::setw(6) << i << " : " << elements_[i].rotation_ << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Element min length:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        meshfile << '\t' << '\t' << "element " << std::setw(6) << i << " : " << elements_[i].delta_xy_min_ << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Element N:" << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        meshfile << '\t' << '\t' << "element " << std::setw(6) << i << " : " << elements_[i].N_ << std::endl;
    }
    
    meshfile << std::endl << '\t' <<  "Face N:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].N_ << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face length:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].length_ << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face normal:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].normal_ << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face tangent:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].tangent_ << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face offset:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : " << std::setw(6) << faces_[i].offset_[0] << " " << std::setw(6) << faces_[i].offset_[1] << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face scale:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : " << std::setw(6) << faces_[i].scale_[0] << " " << std::setw(6) << faces_[i].scale_[1] << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Face refine:" << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        meshfile << '\t' << '\t' << "face " << std::setw(6) << i << " : " << faces_[i].refine_ << std::endl;
    }

    meshfile << std::endl <<  "Interfaces" << std::endl;
    meshfile << '\t' <<  "Interface destination, origin and origin side:" << std::endl;
    for (size_t i = 0; i < interfaces_origin_.size(); ++i) {
        meshfile << '\t' << '\t' << "interface " << std::setw(6) << i << " : " << std::setw(6) << interfaces_destination_[i] << " " << std::setw(6) << interfaces_origin_[i] << " " << std::setw(6) << interfaces_origin_side_[i] << std::endl;
    }

    meshfile <<  "MPI interfaces:" << std::endl;
    for (size_t j = 0; j < mpi_interfaces_process_.size(); ++j) {
        meshfile << '\t' << "MPI interface to process " << mpi_interfaces_process_[j] << " of outgoing size " << mpi_interfaces_outgoing_size_[j] << ", incoming size " << mpi_interfaces_incoming_size_[j] << ", outgoing offset " << mpi_interfaces_outgoing_offset_[j] << " and incoming offset " << mpi_interfaces_incoming_offset_[j] << ":" << std::endl;
        meshfile << '\t' << '\t' << "Outgoing element and element side:" << std::endl;
        for (size_t i = 0; i < mpi_interfaces_outgoing_size_[j]; ++i) {
            meshfile << '\t' << '\t' << '\t' << "mpi interface " << std::setw(6) << i << " : " << std::setw(6) << mpi_interfaces_origin_[mpi_interfaces_outgoing_offset_[j] + i] << " " << std::setw(6) << mpi_interfaces_origin_side_[mpi_interfaces_outgoing_offset_[j] + i] << std::endl;
        }
        meshfile << '\t' << '\t' << "Incoming element:" << std::endl;
        for (size_t i = 0; i < mpi_interfaces_incoming_size_[j]; ++i) {
            meshfile << '\t' << '\t' << '\t' << "mpi interface " << std::setw(6) << i << " : " << std::setw(6) << mpi_interfaces_destination_[mpi_interfaces_incoming_offset_[j] + i] << std::endl;
        }
    }

    meshfile << std::endl << '\t' <<  "Wall boundaries:" << std::endl;
    for (size_t i = 0; i < wall_boundaries_.size(); ++i) {
        meshfile << '\t' << '\t' << "wall " << std::setw(6) << i << " : " << std::setw(6) << wall_boundaries_[i] << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Symmetry boundaries:" << std::endl;
    for (size_t i = 0; i < symmetry_boundaries_.size(); ++i) {
        meshfile << '\t' << '\t' << "symmetry " << std::setw(6) << i << " : " << std::setw(6) << symmetry_boundaries_[i] << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Inflow boundaries:" << std::endl;
    for (size_t i = 0; i < inflow_boundaries_.size(); ++i) {
        meshfile << '\t' << '\t' << "inflow " << std::setw(6) << i << " : " << std::setw(6) << inflow_boundaries_[i] << std::endl;
    }

    meshfile << std::endl << '\t' <<  "Outflow boundaries:" << std::endl;
    for (size_t i = 0; i < outflow_boundaries_.size(); ++i) {
        meshfile << '\t' << '\t' << "outflow " << std::setw(6) << i << " : " << std::setw(6) << outflow_boundaries_[i] << std::endl;
    }

    meshfile << std::endl <<  "Element faces:" << std::endl;
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        const Element2D_t& element = elements_[element_index];
        meshfile << '\t' << "Element " << element_index << " has:"; // CHECK only works for quadrilaterals
        meshfile << std::endl << '\t' << '\t' << "Bottom faces: "; 
        for (auto face_index : element.faces_[0]) {
            meshfile << face_index << " ";
        }
        meshfile << std::endl << '\t' << '\t' << "Right faces: ";
        for (auto face_index : element.faces_[1]) {
            meshfile << face_index << " ";
        }
        meshfile << std::endl << '\t' << '\t' << "Top faces: ";
        for (auto face_index : element.faces_[2]) {
            meshfile << face_index << " ";
        }
        meshfile << std::endl << '\t' << '\t' << "Left faces: ";
        for (auto face_index : element.faces_[3]) {
            meshfile << face_index << " ";
        }
        meshfile << std::endl;
    }
    
    for (size_t element_index = n_elements_; element_index < elements_.size(); ++element_index) {
        const Element2D_t& element = elements_[element_index];
        meshfile << '\t' << "Element " << element_index << " has:" << std::endl << '\t' << "Faces: "; // CHECK only works for quadrilaterals
        for (auto face_index : element.faces_[0]) {
            meshfile << face_index << " ";
        }
        meshfile << std::endl;
    }

    meshfile.close();
}

auto SEM::Host::Meshes::Mesh2D_t::write_data(
        hostFloat time, 
        const std::vector<std::vector<hostFloat>>& interpolation_matrices, 
        const SEM::Helpers::DataWriter_t& data_writer) -> void {
    
    get_solution(interpolation_matrices);

    data_writer.write_data(n_interpolation_points_, n_elements_, time, x_output_, y_output_, p_output_, u_output_, v_output_);
}

auto SEM::Host::Meshes::Mesh2D_t::write_complete_data(
        hostFloat time, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& interpolation_matrices, 
        const SEM::Helpers::DataWriter_t& data_writer) -> void {
    
    std::vector<hostFloat> dp_dt(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<hostFloat> du_dt(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<hostFloat> dv_dt(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<int> N(n_elements_);
    std::vector<hostFloat> p_error(n_elements_);
    std::vector<hostFloat> u_error(n_elements_);
    std::vector<hostFloat> v_error(n_elements_);
    std::vector<hostFloat> p_sigma(n_elements_);
    std::vector<hostFloat> u_sigma(n_elements_);
    std::vector<hostFloat> v_sigma(n_elements_);
    std::vector<int> refine(n_elements_);
    std::vector<int> coarsen(n_elements_);
    std::vector<int> split_level(n_elements_);
    std::vector<hostFloat> p_analytical_error(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<hostFloat> u_analytical_error(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<hostFloat> v_analytical_error(n_elements_ * n_interpolation_points_ * n_interpolation_points_);
    std::vector<int> status(n_elements_);
    std::vector<int> rotation(n_elements_);

    get_complete_solution(time, polynomial_nodes, interpolation_matrices, N, dp_dt, du_dt, dv_dt, p_error, u_error, v_error, p_sigma, u_sigma, v_sigma, refine, coarsen, split_level, p_analytical_error, u_analytical_error, v_analytical_error, status, rotation);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

    data_writer.write_complete_data(n_interpolation_points_, n_elements_, time, global_rank, x_output_, y_output_, p_output_, u_output_, v_output_, N, dp_dt, du_dt, dv_dt, p_error, u_error, v_error, p_sigma, u_sigma, v_sigma, refine, coarsen, split_level, p_analytical_error, u_analytical_error, v_analytical_error, status, rotation);
}

auto SEM::Host::Meshes::Mesh2D_t::adapt(
        int N_max, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    size_t n_splitting_elements = 0;
    for (int i = 0; i < n_elements_; ++i) {
        n_splitting_elements += elements_[i].would_h_refine(max_split_level_);
    }

    if (n_splitting_elements == 0) {
        size_t n_splitting_mpi_interface_incoming_elements = 0;
        size_t n_mpi_interface_new_nodes = 0;
        if (!mpi_interfaces_origin_.empty()) {
            int global_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
            int global_size;
            MPI_Comm_size(MPI_COMM_WORLD, &global_size);
            
            SEM::Host::Meshes::get_MPI_interfaces_N(mpi_interfaces_origin_.size(), N_max, elements_.data(), mpi_interfaces_origin_.data(), interfaces_N_.data());

            for (size_t mpi_interface_element_index = 0; mpi_interface_element_index < mpi_interfaces_origin_.size(); ++mpi_interface_element_index) {
                interfaces_refine_[mpi_interface_element_index] = false;
            }

            for (size_t i = 0; i < mpi_interfaces_process_.size(); ++i) {
                MPI_Isend(interfaces_N_.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]), MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * (mpi_interfaces_process_.size() + i)]);
                MPI_Irecv(receiving_interfaces_N_.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank), MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * i]);
            
                MPI_Isend(interfaces_refine_.get() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], MPI_C_BOOL, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 1, MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * (mpi_interfaces_process_.size() + i) + 1]);
                MPI_Irecv(receiving_interfaces_refine_.get() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], MPI_C_BOOL, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 1, MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * i + 1]);
            }

            MPI_Waitall(mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size(), requests_adaptivity_.data(), statuses_adaptivity_.data());

            SEM::Host::Meshes::copy_mpi_interfaces_error(mpi_interfaces_destination_.size(), elements_.data(), faces_.data(), nodes_.data(), mpi_interfaces_destination_.data(), receiving_interfaces_N_.data(), receiving_interfaces_refine_.get(), receiving_interfaces_refine_without_splitting_.get(), receiving_interfaces_creating_node_.get());
    
            for (int i = 0; i < mpi_interfaces_destination_.size(); ++i) {
                n_splitting_mpi_interface_incoming_elements += receiving_interfaces_refine_[i] && !receiving_interfaces_refine_without_splitting_[i];
                n_mpi_interface_new_nodes += receiving_interfaces_creating_node_[i];
            }

            MPI_Waitall(mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size(), requests_adaptivity_.data() + mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size(), statuses_adaptivity_.data() + mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size());
        }

        if (n_splitting_mpi_interface_incoming_elements == 0) {
            SEM::Host::Meshes::p_adapt(n_elements_, elements_.data(), N_max, nodes_.data(), polynomial_nodes, barycentric_weights);

            // We need to adjust the boundaries in all cases, or check if of our neighbours have to change
            if (!wall_boundaries_.empty()) {
                SEM::Host::Meshes::adjust_boundaries(wall_boundaries_.size(), elements_.data(), wall_boundaries_.data(), faces_.data());
            }
            if (!symmetry_boundaries_.empty()) {
                SEM::Host::Meshes::adjust_boundaries(symmetry_boundaries_.size(), elements_.data(), symmetry_boundaries_.data(), faces_.data());
            }
            if (!inflow_boundaries_.empty()) {
                SEM::Host::Meshes::adjust_boundaries(inflow_boundaries_.size(), elements_.data(), inflow_boundaries_.data(), faces_.data());
            }
            if (!outflow_boundaries_.empty()) {
                SEM::Host::Meshes::adjust_boundaries(outflow_boundaries_.size(), elements_.data(), outflow_boundaries_.data(), faces_.data());
            }
            if (!interfaces_origin_.empty()) {
                SEM::Host::Meshes::adjust_interfaces(interfaces_origin_.size(), elements_.data(), interfaces_origin_.data(), interfaces_destination_.data());
            }
            if (!mpi_interfaces_destination_.empty()) {
                const size_t n_nodes_old = nodes_.size();
                if (n_mpi_interface_new_nodes > 0) {
                    nodes_.resize(nodes_.size() + n_mpi_interface_new_nodes);
                }
                
                SEM::Host::Meshes::adjust_MPI_incoming_interfaces(mpi_interfaces_destination_.size(), n_nodes_old, elements_.data(), faces_.data(), mpi_interfaces_destination_.data(), receiving_interfaces_N_.data(), nodes_.data(), receiving_interfaces_refine_.get(), receiving_interfaces_refine_without_splitting_.get(), receiving_interfaces_creating_node_.get(), polynomial_nodes);
            }

            SEM::Host::Meshes::adjust_faces(faces_.size(), faces_.data(), elements_.data());
        }
        else {
            SEM::Host::Meshes::no_new_nodes(n_elements_, elements_.data());

            size_t n_splitting_faces = 0;
            for (auto& face: faces_) {
                face.refine_ = false;
                if (elements_[face.elements_[0]].additional_nodes_[face.elements_side_[0]]) {
                    const Element2D_t& element = elements_[face.elements_[0]];
                    const size_t element_side = face.elements_side_[0];
                    const size_t element_next_side = (element_side + 1 < element.nodes_.size()) ? element_side + 1 : size_t{0};
                    
                    const Vec2<hostFloat> new_node = (nodes_[element.nodes_[element_side]] + nodes_[element.nodes_[element_next_side]])/2;

                    std::array<bool, 2> side_face {false, false};
                    const std::array<Vec2<hostFloat>, 2> AB {
                        new_node - nodes_[element.nodes_[element_side]],
                        nodes_[element.nodes_[element_next_side]] - new_node
                    };

                    const std::array<hostFloat, 2> AB_dot_inv {
                        1/AB[0].dot(AB[0]),
                        1/AB[1].dot(AB[1])
                    };
                    
                    const std::array<Vec2<hostFloat>, 2> AC {
                        nodes_[face.nodes_[0]] - nodes_[element.nodes_[element_side]],
                        nodes_[face.nodes_[0]] - new_node
                    };
                    const std::array<Vec2<hostFloat>, 2> AD {
                        nodes_[face.nodes_[1]] - nodes_[element.nodes_[element_side]],
                        nodes_[face.nodes_[1]] - new_node
                    };

                    const std::array<hostFloat, 2> C_proj {
                        AC[0].dot(AB[0]) * AB_dot_inv[0],
                        AC[1].dot(AB[1]) * AB_dot_inv[1]
                    };
                    const std::array<hostFloat, 2> D_proj {
                        AD[0].dot(AB[0]) * AB_dot_inv[0],
                        AD[1].dot(AB[1]) * AB_dot_inv[1]
                    };

                    // CHECK this projection is different than the one used for splitting, maybe wrong
                    // The face is within the first element
                    if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                        && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                        || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                        && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                        side_face[0] = true;
                    }
                    // The face is within the second element
                    if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                        && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                        || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                        && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                        side_face[1] = true;
                    }
                    
                    if (side_face[0] && side_face[1]) {
                        face.refine_ = true;
                        ++n_splitting_faces;
                    }
                }
                else if (elements_[face.elements_[1]].additional_nodes_[face.elements_side_[1]] && !face.refine_) {
                    const Element2D_t& element = elements_[face.elements_[1]];
                    const size_t element_side = face.elements_side_[1];
                    const size_t element_next_side = (element_side + 1 < element.nodes_.size()) ? element_side + 1 : size_t{0};

                    const Vec2<hostFloat> new_node = (nodes_[element.nodes_[element_side]] + nodes_[element.nodes_[element_next_side]])/2;

                    std::array<bool, 2> side_face {false, false};
                    const std::array<Vec2<hostFloat>, 2> AB {
                        new_node - nodes_[element.nodes_[element_side]],
                        nodes_[element.nodes_[element_next_side]] - new_node
                    };

                    const std::array<hostFloat, 2> AB_dot_inv {
                        1/AB[0].dot(AB[0]),
                        1/AB[1].dot(AB[1])
                    };

                    const std::array<Vec2<hostFloat>, 2> AC {
                        nodes_[face.nodes_[0]] - nodes_[element.nodes_[element_side]],
                        nodes_[face.nodes_[0]] - new_node
                    };
                    const std::array<Vec2<hostFloat>, 2> AD {
                        nodes_[face.nodes_[1]] - nodes_[element.nodes_[element_side]],
                        nodes_[face.nodes_[1]] - new_node
                    };

                    const std::array<hostFloat, 2> C_proj {
                        AC[0].dot(AB[0]) * AB_dot_inv[0],
                        AC[1].dot(AB[1]) * AB_dot_inv[1]
                    };
                    const std::array<hostFloat, 2> D_proj {
                        AD[0].dot(AB[0]) * AB_dot_inv[0],
                        AD[1].dot(AB[1]) * AB_dot_inv[1]
                    };

                    // CHECK this projection is different than the one used for splitting, maybe wrong
                    // The face is within the first element
                    if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                        && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                        || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                        && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                        side_face[0] = true;
                    }
                    // The face is within the second element
                    if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                        && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                        || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                        && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                        side_face[1] = true;
                    }
                
                    if (side_face[0] && side_face[1]) {
                        face.refine_ = true;
                        ++n_splitting_faces;
                    }
                }
            }

            std::vector<Element2D_t> new_elements(elements_.size() + n_splitting_mpi_interface_incoming_elements);

            std::vector<size_t> new_mpi_interfaces_destination(mpi_interfaces_destination_.size() + n_splitting_mpi_interface_incoming_elements);

            std::vector<size_t> elements_new_indices(elements_.size());

            if (n_splitting_faces == 0) {
                SEM::Host::Meshes::p_adapt_move(n_elements_, elements_.data(), new_elements.data(), N_max, nodes_.data(), polynomial_nodes, barycentric_weights, elements_new_indices.data());

                if (!wall_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(wall_boundaries_.size(), n_elements_, elements_.data(), new_elements.data(), wall_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!symmetry_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(symmetry_boundaries_.size(), n_elements_ + wall_boundaries_.size(), elements_.data(), new_elements.data(), symmetry_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!inflow_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(inflow_boundaries_.size(), n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size(), elements_.data(), new_elements.data(), inflow_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!outflow_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(outflow_boundaries_.size(), n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size() + inflow_boundaries_.size(), elements_.data(), new_elements.data(), outflow_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!interfaces_origin_.empty()) {
                    SEM::Host::Meshes::move_interfaces(interfaces_origin_.size(), n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size() + inflow_boundaries_.size() + outflow_boundaries_.size(), elements_.data(), new_elements.data(), interfaces_origin_.data(), interfaces_destination_.data(), elements_new_indices.data());
                }

                const size_t n_nodes_old = nodes_.size();
                if (n_mpi_interface_new_nodes > 0) {
                    nodes_.resize(nodes_.size() + n_mpi_interface_new_nodes);
                }

                // MPI
                SEM::Host::Meshes::split_mpi_incoming_interfaces(mpi_interfaces_destination_.size(), faces_.size(), n_nodes_old, n_splitting_elements, n_splitting_faces, n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size() + inflow_boundaries_.size() + outflow_boundaries_.size() + interfaces_origin_.size(), elements_.data(), new_elements.data(), mpi_interfaces_destination_.data(), new_mpi_interfaces_destination.data(), faces_.data(), nodes_.data(), polynomial_nodes, receiving_interfaces_N_.data(), receiving_interfaces_refine_.get(), receiving_interfaces_refine_without_splitting_.get(), receiving_interfaces_creating_node_.get(), elements_new_indices.data());

                // Faces
                SEM::Host::Meshes::adjust_faces_neighbours(faces_.size(), faces_.data(), elements_.data(), nodes_.data(), max_split_level_, N_max, elements_new_indices.data());
            }
            else {
                std::vector<Face2D_t> new_faces(faces_.size() + n_splitting_faces);

                const size_t old_n_nodes = nodes_.size();
                nodes_.resize(nodes_.size() + n_splitting_faces + n_mpi_interface_new_nodes);

                SEM::Host::Meshes::p_adapt_split_faces(n_elements_, faces_.size(), old_n_nodes, n_splitting_elements, elements_.data(), new_elements.data(), faces_.data(), N_max, nodes_.data(), polynomial_nodes, barycentric_weights, elements_new_indices.data());

                if (!wall_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(wall_boundaries_.size(), n_elements_, elements_.data(), new_elements.data(), wall_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!symmetry_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(symmetry_boundaries_.size(), n_elements_ + wall_boundaries_.size(), elements_.data(), new_elements.data(), symmetry_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!inflow_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(inflow_boundaries_.size(), n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size(), elements_.data(), new_elements.data(), inflow_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!outflow_boundaries_.empty()) {
                    SEM::Host::Meshes::move_boundaries(outflow_boundaries_.size(), n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size() + inflow_boundaries_.size(), elements_.data(), new_elements.data(), outflow_boundaries_.data(), faces_.data(), elements_new_indices.data());
                }
                if (!interfaces_origin_.empty()) {
                    SEM::Host::Meshes::move_interfaces(interfaces_origin_.size(), n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size() + inflow_boundaries_.size() + outflow_boundaries_.size(), elements_.data(), new_elements.data(), interfaces_origin_.data(), interfaces_destination_.data(), elements_new_indices.data());
                }

                // MPI
                SEM::Host::Meshes::split_mpi_incoming_interfaces(mpi_interfaces_destination_.size(), faces_.size(), old_n_nodes, n_splitting_elements, n_splitting_faces, n_elements_ + wall_boundaries_.size() + symmetry_boundaries_.size() + inflow_boundaries_.size() + outflow_boundaries_.size() + interfaces_origin_.size(), elements_.data(), new_elements.data(), mpi_interfaces_destination_.data(), new_mpi_interfaces_destination.data(), faces_.data(), nodes_.data(), polynomial_nodes, receiving_interfaces_N_.data(), receiving_interfaces_refine_.get(), receiving_interfaces_refine_without_splitting_.get(), receiving_interfaces_creating_node_.get(), elements_new_indices.data());

                SEM::Host::Meshes::split_faces(faces_.size(), old_n_nodes, n_splitting_elements, faces_.data(), new_faces.data(), elements_.data(), nodes_.data(), max_split_level_, N_max, elements_new_indices.data());

                faces_ = std::move(new_faces);
            }

            elements_ = std::move(new_elements);

            mpi_interfaces_destination_ = std::move(new_mpi_interfaces_destination);

            // Updating quantities
            if (!mpi_interfaces_destination_.empty()) {
                size_t interface_offset = 0;
                for (size_t i = 0; i < mpi_interfaces_incoming_size_.size(); ++i) {
                    size_t splitting_incoming_elements = 0;
                    for (size_t j = 0; j < mpi_interfaces_incoming_size_[i]; ++j) {
                        splitting_incoming_elements += receiving_interfaces_refine_[mpi_interfaces_incoming_offset_[i] + j] && !receiving_interfaces_refine_without_splitting_[mpi_interfaces_incoming_offset_[i] + j];
                    }
                    mpi_interfaces_incoming_offset_[i] = interface_offset;
                    mpi_interfaces_incoming_size_[i] += splitting_incoming_elements;
                    interface_offset += mpi_interfaces_incoming_size_[i];
                }
            }

            // Boundary solution exchange
            receiving_interfaces_p_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
            receiving_interfaces_u_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
            receiving_interfaces_v_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
            receiving_interfaces_N_ = std::vector<int>(mpi_interfaces_destination_.size());
            receiving_interfaces_refine_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
            receiving_interfaces_refine_without_splitting_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
            receiving_interfaces_creating_node_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
        }

        return;
    }

    // Sending and receiving splitting elements on MPI interfaces
    size_t n_splitting_mpi_interface_incoming_elements = 0;
    size_t n_mpi_interface_new_nodes = 0;
    size_t n_splitting_mpi_interface_outgoing_elements = 0;
    if (!mpi_interfaces_origin_.empty()) {
        int global_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
        int global_size;
        MPI_Comm_size(MPI_COMM_WORLD, &global_size);
        
        SEM::Host::Meshes::get_MPI_interfaces_adaptivity(mpi_interfaces_origin_.size(), n_elements_, mpi_interfaces_process_.size(), elements_.data(), faces_.data(), nodes_.data(), mpi_interfaces_origin_.data(), mpi_interfaces_origin_side_.data(), mpi_interfaces_destination_.data(), mpi_interfaces_process_.data(), mpi_interfaces_outgoing_size_.data(), mpi_interfaces_incoming_size_.data(), mpi_interfaces_incoming_offset_.data(), interfaces_N_.data(), interfaces_refine_.get(), interfaces_refine_without_splitting_.get(), max_split_level_, N_max);

        for (size_t i = 0; i < mpi_interfaces_process_.size(); ++i) {
            MPI_Isend(interfaces_N_.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]), MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * (mpi_interfaces_process_.size() + i)]);
            MPI_Irecv(receiving_interfaces_N_.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank), MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * i]);
        
            MPI_Isend(interfaces_refine_.get() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], MPI_C_BOOL, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 1, MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * (mpi_interfaces_process_.size() + i) + 1]);
            MPI_Irecv(receiving_interfaces_refine_.get() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], MPI_C_BOOL, mpi_interfaces_process_[i], mpi_adaptivity_split_offset * global_size * global_size + mpi_adaptivity_split_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 1, MPI_COMM_WORLD, &requests_adaptivity_[mpi_adaptivity_split_n_transfers * i + 1]);
        }

        MPI_Waitall(mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size(), requests_adaptivity_.data(), statuses_adaptivity_.data());

        SEM::Host::Meshes::copy_mpi_interfaces_error(mpi_interfaces_destination_.size(), elements_.data(), faces_.data(), nodes_.data(), mpi_interfaces_destination_.data(), receiving_interfaces_N_.data(), receiving_interfaces_refine_.get(), receiving_interfaces_refine_without_splitting_.get(), receiving_interfaces_creating_node_.get());

        for (int i = 0; i < mpi_interfaces_destination_.size(); ++i) {
            n_splitting_mpi_interface_incoming_elements += receiving_interfaces_refine_[i] && !receiving_interfaces_refine_without_splitting_[i];
            n_mpi_interface_new_nodes += receiving_interfaces_creating_node_[i];
        }

        for (int i = 0; i < mpi_interfaces_origin_.size(); ++i) {
            n_splitting_mpi_interface_outgoing_elements += interfaces_refine_[i] && !interfaces_refine_without_splitting_[i];
        }

        MPI_Waitall(mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size(), requests_adaptivity_.data() + mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size(), statuses_adaptivity_.data() + mpi_adaptivity_split_n_transfers * mpi_interfaces_process_.size());
    }

    SEM::Host::Meshes::find_nodes(n_elements_, elements_.data(), faces_.data(), nodes_.data(), max_split_level_);
    if (!interfaces_origin_.empty()) {
        SEM::Host::Meshes::copy_interfaces_error(interfaces_origin_.size(), elements_.data(), interfaces_origin_.data(), interfaces_origin_side_.data(), interfaces_destination_.data());
    }
    if (!wall_boundaries_.empty()) {
        SEM::Host::Meshes::copy_boundaries_error(wall_boundaries_.size(), elements_.data(), wall_boundaries_.data(), faces_.data());
    }
    if (!symmetry_boundaries_.empty()) {
        SEM::Host::Meshes::copy_boundaries_error(symmetry_boundaries_.size(), elements_.data(), symmetry_boundaries_.data(), faces_.data());
    }
    if (!inflow_boundaries_.empty()) {
        SEM::Host::Meshes::copy_boundaries_error(inflow_boundaries_.size(), elements_.data(), inflow_boundaries_.data(), faces_.data());
    }
    if (!outflow_boundaries_.empty()) {
        SEM::Host::Meshes::copy_boundaries_error(outflow_boundaries_.size(), elements_.data(), outflow_boundaries_.data(), faces_.data());
    }

    size_t n_splitting_faces = 0;
    for (auto& face: faces_) {
        face.refine_ = false;
        if (elements_[face.elements_[0]].additional_nodes_[face.elements_side_[0]]) {
            const Element2D_t& element = elements_[face.elements_[0]];
            const size_t element_side = face.elements_side_[0];
            const size_t element_next_side = (element_side + 1 < element.nodes_.size()) ? element_side + 1 : size_t{0};
            
            const Vec2<hostFloat> new_node = (nodes_[element.nodes_[element_side]] + nodes_[element.nodes_[element_next_side]])/2;

            std::array<bool, 2> side_face {false, false};
            const std::array<Vec2<hostFloat>, 2> AB {
                new_node - nodes_[element.nodes_[element_side]],
                nodes_[element.nodes_[element_next_side]] - new_node
            };

            const std::array<hostFloat, 2> AB_dot_inv {
                1/AB[0].dot(AB[0]),
                1/AB[1].dot(AB[1])
            };
            
            const std::array<Vec2<hostFloat>, 2> AC {
                nodes_[face.nodes_[0]] - nodes_[element.nodes_[element_side]],
                nodes_[face.nodes_[0]] - new_node
            };
            const std::array<Vec2<hostFloat>, 2> AD {
                nodes_[face.nodes_[1]] - nodes_[element.nodes_[element_side]],
                nodes_[face.nodes_[1]] - new_node
            };

            const std::array<hostFloat, 2> C_proj {
                AC[0].dot(AB[0]) * AB_dot_inv[0],
                AC[1].dot(AB[1]) * AB_dot_inv[1]
            };
            const std::array<hostFloat, 2> D_proj {
                AD[0].dot(AB[0]) * AB_dot_inv[0],
                AD[1].dot(AB[1]) * AB_dot_inv[1]
            };

            // CHECK this projection is different than the one used for splitting, maybe wrong
            // The face is within the first element
            if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                side_face[0] = true;
            }
            // The face is within the second element
            if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                side_face[1] = true;
            }
            
            if (side_face[0] && side_face[1]) {
                face.refine_ = true;
                ++n_splitting_faces;
            }
        }
        else if (elements_[face.elements_[1]].additional_nodes_[face.elements_side_[1]] && !face.refine_) {
            const Element2D_t& element = elements_[face.elements_[1]];
            const size_t element_side = face.elements_side_[1];
            const size_t element_next_side = (element_side + 1 < element.nodes_.size()) ? element_side + 1 : size_t{0};

            const Vec2<hostFloat> new_node = (nodes_[element.nodes_[element_side]] + nodes_[element.nodes_[element_next_side]])/2;

            std::array<bool, 2> side_face {false, false};
            const std::array<Vec2<hostFloat>, 2> AB {
                new_node - nodes_[element.nodes_[element_side]],
                nodes_[element.nodes_[element_next_side]] - new_node
            };

            const std::array<hostFloat, 2> AB_dot_inv {
                1/AB[0].dot(AB[0]),
                1/AB[1].dot(AB[1])
            };

            const std::array<Vec2<hostFloat>, 2> AC {
                nodes_[face.nodes_[0]] - nodes_[element.nodes_[element_side]],
                nodes_[face.nodes_[0]] - new_node
            };
            const std::array<Vec2<hostFloat>, 2> AD {
                nodes_[face.nodes_[1]] - nodes_[element.nodes_[element_side]],
                nodes_[face.nodes_[1]] - new_node
            };

            const std::array<hostFloat, 2> C_proj {
                AC[0].dot(AB[0]) * AB_dot_inv[0],
                AC[1].dot(AB[1]) * AB_dot_inv[1]
            };
            const std::array<hostFloat, 2> D_proj {
                AD[0].dot(AB[0]) * AB_dot_inv[0],
                AD[1].dot(AB[1]) * AB_dot_inv[1]
            };

            // CHECK this projection is different than the one used for splitting, maybe wrong
            // The face is within the first element
            if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                side_face[0] = true;
            }
            // The face is within the second element
            if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                side_face[1] = true;
            }

            if (side_face[0] && side_face[1]) {
                face.refine_ = true;
                ++n_splitting_faces;
            }
        }
    }

    // Boundary and interfaces new sizes
    size_t n_splitting_wall_boundaries = 0;
    for (auto boundary_element_index : wall_boundaries_) {
        if (elements_[boundary_element_index].faces_[0].size() == 1) { // Should always be the case 
            n_splitting_wall_boundaries += faces_[elements_[boundary_element_index].faces_[0][0]].refine_;
        }
    }

    size_t n_splitting_symmetry_boundaries = 0;
    for (auto boundary_element_index : symmetry_boundaries_) {
        if (elements_[boundary_element_index].faces_[0].size() == 1) { // Should always be the case 
            n_splitting_symmetry_boundaries += faces_[elements_[boundary_element_index].faces_[0][0]].refine_;
        }
    }

    size_t n_splitting_inflow_boundaries = 0;
    for (auto boundary_element_index : inflow_boundaries_) {
        if (elements_[boundary_element_index].faces_[0].size() == 1) { // Should always be the case 
            n_splitting_inflow_boundaries += faces_[elements_[boundary_element_index].faces_[0][0]].refine_;
        }
    }

    size_t n_splitting_outflow_boundaries = 0;
    for (auto boundary_element_index : outflow_boundaries_) {
        if (elements_[boundary_element_index].faces_[0].size() == 1) { // Should always be the case 
            n_splitting_outflow_boundaries += faces_[elements_[boundary_element_index].faces_[0][0]].refine_;
        }
    }

    size_t n_splitting_interface_elements = 0;
    for (auto interface_element_index : interfaces_origin_) {
        n_splitting_interface_elements += elements_[interface_element_index].would_h_refine(max_split_level_);
    }

    // New arrays
    std::vector<Element2D_t> new_elements(elements_.size() + 3 * n_splitting_elements + n_splitting_wall_boundaries + n_splitting_symmetry_boundaries + n_splitting_inflow_boundaries + n_splitting_outflow_boundaries + n_splitting_interface_elements + n_splitting_mpi_interface_incoming_elements);
    const size_t old_n_nodes = nodes_.size();
    nodes_.resize(nodes_.size() + n_splitting_elements + n_splitting_faces + n_mpi_interface_new_nodes);
    std::vector<Face2D_t> new_faces(faces_.size() + 4 * n_splitting_elements + n_splitting_faces);

    std::vector<size_t> new_wall_boundaries(wall_boundaries_.size() + n_splitting_wall_boundaries);
    std::vector<size_t> new_symmetry_boundaries(symmetry_boundaries_.size() + n_splitting_symmetry_boundaries);
    std::vector<size_t> new_inflow_boundaries(inflow_boundaries_.size() + n_splitting_inflow_boundaries);
    std::vector<size_t> new_outflow_boundaries(outflow_boundaries_.size() + n_splitting_outflow_boundaries);

    std::vector<size_t> new_interfaces_origin(interfaces_origin_.size() + n_splitting_interface_elements);
    std::vector<size_t> new_interfaces_origin_side(interfaces_origin_side_.size() + n_splitting_interface_elements);
    std::vector<size_t> new_interfaces_destination(interfaces_destination_.size() + n_splitting_interface_elements);

    std::vector<size_t> new_mpi_interfaces_origin(mpi_interfaces_origin_.size() + n_splitting_mpi_interface_outgoing_elements);
    std::vector<size_t> new_mpi_interfaces_origin_side(mpi_interfaces_origin_side_.size() + n_splitting_mpi_interface_outgoing_elements);
    std::vector<size_t> new_mpi_interfaces_destination(mpi_interfaces_destination_.size() + n_splitting_mpi_interface_incoming_elements);

    std::vector<size_t> elements_new_indices(elements_.size());

    // Creating new entities and moving ond ones, adjusting as needed
    SEM::Host::Meshes::hp_adapt(n_elements_, faces_.size(), old_n_nodes, n_splitting_elements, elements_.data(), new_elements.data(), faces_.data(), new_faces.data(), max_split_level_, N_max, nodes_.data(), polynomial_nodes, barycentric_weights, elements_new_indices.data());
    
    if (!wall_boundaries_.empty()) {
        SEM::Host::Meshes::split_boundaries(wall_boundaries_.size(), faces_.size(), old_n_nodes, n_splitting_elements, n_elements_ + 3 * n_splitting_elements, elements_.data(), new_elements.data(), wall_boundaries_.data(), new_wall_boundaries.data(), faces_.data(), nodes_.data(), polynomial_nodes, elements_new_indices.data());
    }
    if (!symmetry_boundaries_.empty()) {
        SEM::Host::Meshes::split_boundaries(symmetry_boundaries_.size(), faces_.size(), old_n_nodes, n_splitting_elements, n_elements_ + 3 * n_splitting_elements + wall_boundaries_.size() + n_splitting_wall_boundaries, elements_.data(), new_elements.data(), symmetry_boundaries_.data(), new_symmetry_boundaries.data(), faces_.data(), nodes_.data(), polynomial_nodes, elements_new_indices.data());
    }
    if (!inflow_boundaries_.empty()) {
        SEM::Host::Meshes::split_boundaries(inflow_boundaries_.size(), faces_.size(), old_n_nodes, n_splitting_elements, n_elements_ + 3 * n_splitting_elements + wall_boundaries_.size() + n_splitting_wall_boundaries + symmetry_boundaries_.size() + n_splitting_symmetry_boundaries, elements_.data(), new_elements.data(), inflow_boundaries_.data(), new_inflow_boundaries.data(), faces_.data(), nodes_.data(), polynomial_nodes, elements_new_indices.data());
    }
    if (!outflow_boundaries_.empty()) {
        SEM::Host::Meshes::split_boundaries(outflow_boundaries_.size(), faces_.size(), old_n_nodes, n_splitting_elements, n_elements_ + 3 * n_splitting_elements + wall_boundaries_.size() + n_splitting_wall_boundaries + symmetry_boundaries_.size() + n_splitting_symmetry_boundaries + inflow_boundaries_.size() + n_splitting_inflow_boundaries, elements_.data(), new_elements.data(), outflow_boundaries_.data(), new_outflow_boundaries.data(), faces_.data(), nodes_.data(), polynomial_nodes, elements_new_indices.data());
    }

    if (!interfaces_origin_.empty()) {
        SEM::Host::Meshes::split_interfaces(interfaces_origin_.size(), faces_.size(), old_n_nodes, n_splitting_elements, n_elements_ + 3 * n_splitting_elements + wall_boundaries_.size() + n_splitting_wall_boundaries + symmetry_boundaries_.size() + n_splitting_symmetry_boundaries + inflow_boundaries_.size() + n_splitting_inflow_boundaries + outflow_boundaries_.size() + n_splitting_outflow_boundaries, elements_.data(), new_elements.data(), interfaces_origin_.data(), interfaces_origin_side_.data(), interfaces_destination_.data(), new_interfaces_origin.data(), new_interfaces_origin_side.data(), new_interfaces_destination.data(), faces_.data(), nodes_.data(), max_split_level_, N_max, polynomial_nodes, elements_new_indices.data());
    }

    if (!mpi_interfaces_process_.empty()) {       
        SEM::Host::Meshes::split_mpi_outgoing_interfaces(mpi_interfaces_origin_.size(), n_elements_, mpi_interfaces_process_.size(), elements_.data(), faces_.data(), nodes_.data(), mpi_interfaces_origin_.data(), mpi_interfaces_origin_side_.data(), mpi_interfaces_destination_.data(), new_mpi_interfaces_origin.data(), new_mpi_interfaces_origin_side.data(), mpi_interfaces_process_.data(), mpi_interfaces_outgoing_size_.data(), mpi_interfaces_incoming_size_.data(), mpi_interfaces_incoming_offset_.data(), interfaces_refine_.get(), interfaces_refine_without_splitting_.get(), max_split_level_);
        SEM::Host::Meshes::split_mpi_incoming_interfaces(mpi_interfaces_destination_.size(), faces_.size(), old_n_nodes, n_splitting_elements, n_splitting_faces, n_elements_ + 3 * n_splitting_elements + wall_boundaries_.size() + n_splitting_wall_boundaries + symmetry_boundaries_.size() + n_splitting_symmetry_boundaries + inflow_boundaries_.size() + n_splitting_inflow_boundaries + outflow_boundaries_.size() + n_splitting_outflow_boundaries + interfaces_origin_.size() + n_splitting_interface_elements, elements_.data(), new_elements.data(), mpi_interfaces_destination_.data(), new_mpi_interfaces_destination.data(), faces_.data(), nodes_.data(), polynomial_nodes, receiving_interfaces_N_.data(), receiving_interfaces_refine_.get(), receiving_interfaces_refine_without_splitting_.get(), receiving_interfaces_creating_node_.get(), elements_new_indices.data());
    }

    SEM::Host::Meshes::split_faces(faces_.size(), old_n_nodes, n_splitting_elements, faces_.data(), new_faces.data(), elements_.data(), nodes_.data(), max_split_level_, N_max, elements_new_indices.data());

    // Swapping out the old arrays for the new ones
    elements_ = std::move(new_elements);
    faces_ = std::move(new_faces);

    wall_boundaries_ = std::move(new_wall_boundaries);
    symmetry_boundaries_ = std::move(new_symmetry_boundaries);
    inflow_boundaries_ = std::move(new_inflow_boundaries);
    outflow_boundaries_ = std::move(new_outflow_boundaries);

    interfaces_origin_ = std::move(new_interfaces_origin);
    interfaces_origin_side_ = std::move(new_interfaces_origin_side);
    interfaces_destination_ = std::move(new_interfaces_destination);

    mpi_interfaces_origin_ = std::move(new_mpi_interfaces_origin);
    mpi_interfaces_origin_side_ = std::move(new_mpi_interfaces_origin_side);
    mpi_interfaces_destination_ = std::move(new_mpi_interfaces_destination);

    // Updating quantities
    if (!mpi_interfaces_origin_.empty()) {
        size_t interface_offset = 0;
        for (size_t i = 0; i < mpi_interfaces_incoming_size_.size(); ++i) {
            size_t splitting_incoming_elements = 0;
            for (size_t j = 0; j < mpi_interfaces_incoming_size_[i]; ++j) {
                splitting_incoming_elements += receiving_interfaces_refine_[mpi_interfaces_incoming_offset_[i] + j] && !receiving_interfaces_refine_without_splitting_[mpi_interfaces_incoming_offset_[i] + j];
            }
            mpi_interfaces_incoming_offset_[i] = interface_offset;
            mpi_interfaces_incoming_size_[i] += splitting_incoming_elements;
            interface_offset += mpi_interfaces_incoming_size_[i];
        }

        interface_offset = 0;
        for (size_t i = 0; i < mpi_interfaces_outgoing_size_.size(); ++i) {
            size_t splitting_incoming_elements = 0;
            for (size_t j = 0; j < mpi_interfaces_outgoing_size_[i]; ++j) {
                splitting_incoming_elements += interfaces_refine_[mpi_interfaces_outgoing_offset_[i] + j] && !interfaces_refine_without_splitting_[mpi_interfaces_outgoing_offset_[i] + j];
            }
            mpi_interfaces_outgoing_offset_[i] = interface_offset;
            mpi_interfaces_outgoing_size_[i] += splitting_incoming_elements;
            interface_offset += mpi_interfaces_outgoing_size_[i];
        }
    }

    n_elements_ += 3 * n_splitting_elements;

    // Output
    x_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    y_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    p_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    u_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
    v_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));

    // Boundary solution exchange
    interfaces_p_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
    interfaces_u_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
    interfaces_v_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
    interfaces_N_ = std::vector<int>(mpi_interfaces_origin_.size());
    interfaces_refine_ = std::make_unique<bool[]>(mpi_interfaces_origin_.size());
    interfaces_refine_without_splitting_ = std::make_unique<bool[]>(mpi_interfaces_origin_.size());

    receiving_interfaces_p_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
    receiving_interfaces_u_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
    receiving_interfaces_v_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
    receiving_interfaces_N_ = std::vector<int>(mpi_interfaces_destination_.size());
    receiving_interfaces_refine_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
    receiving_interfaces_refine_without_splitting_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
    receiving_interfaces_creating_node_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
}

auto SEM::Host::Meshes::Mesh2D_t::load_balance(const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    std::vector<size_t> n_elements_per_proc(global_size);

    const MPI_Datatype size_t_data_type = (sizeof(size_t) == sizeof(unsigned long long)) ? MPI_UNSIGNED_LONG_LONG : (sizeof(size_t) == sizeof(unsigned long)) ? MPI_UNSIGNED_LONG : MPI_UNSIGNED; // CHECK this is a bad way of doing this
    MPI_Allgather(&n_elements_, 1, size_t_data_type, n_elements_per_proc.data(), 1, size_t_data_type, MPI_COMM_WORLD);

    std::vector<size_t> global_element_offset_current(global_size, 0);
    std::vector<size_t> global_element_offset_end_current(global_size);
    global_element_offset_end_current[0] = n_elements_per_proc[0];

    size_t n_elements_global_new = n_elements_per_proc[0];
    for (int i = 1; i < global_size; ++i) {
        global_element_offset_current[i] = global_element_offset_current[i - 1] + n_elements_per_proc[i - 1];
        global_element_offset_end_current[i] = global_element_offset_current[i] + n_elements_per_proc[i];
        n_elements_global_new += n_elements_per_proc[i];
    }

    deviceFloat max_load_imbalance = n_elements_per_proc[0] * total_weight_/(n_elements_global_new * worker_weights_[0]);
    for (int i = 1; i < global_size; ++i) {
        max_load_imbalance = std::max(max_load_imbalance, n_elements_per_proc[i] * total_weight_/(n_elements_global_new * worker_weights_[i]));
    }

    // If there is little imbalance, do nothing.
    if (max_load_imbalance < load_balancing_threshold_) {
        return;
    }

    std::vector<size_t> global_element_offset_new(global_size); // This is the first element that is owned by the proc
    std::vector<size_t> global_element_offset_end_new(global_size); // This is the first element after elements owned by the proc
    std::vector<size_t> n_elements_new(global_size);
    std::vector<size_t> n_elements_send_left(global_size);
    std::vector<size_t> n_elements_recv_left(global_size);
    std::vector<size_t> n_elements_send_right(global_size);
    std::vector<size_t> n_elements_recv_right(global_size);

    const deviceFloat n_elements_per_weight = n_elements_global_new/total_weight_;
    size_t current_element_index = 0;
    for (int i = 0; i < global_size; ++i) {
        n_elements_new[i] = static_cast<size_t>(std::floor(n_elements_per_weight * worker_weights_[i]));
        current_element_index += n_elements_new[i];
    }

    const size_t elements_remainder = n_elements_global_new - current_element_index;
    current_element_index = 0;
    for (int i = 0; i < global_size; ++i) {
        if (i < elements_remainder) {
            ++n_elements_new[i];
        }

        global_element_offset_new[i] = current_element_index;
        global_element_offset_end_new[i] = global_element_offset_new[i] + n_elements_new[i];
        current_element_index = global_element_offset_end_new[i];

        n_elements_send_left[i] = (global_element_offset_new[i] > global_element_offset_current[i]) ? std::min(global_element_offset_new[i] - global_element_offset_current[i], n_elements_per_proc[i]) : size_t{0};
        n_elements_recv_left[i] = (global_element_offset_current[i] > global_element_offset_new[i]) ? std::min(global_element_offset_current[i] - global_element_offset_new[i], n_elements_new[i]) : size_t{0};
        n_elements_send_right[i] = (global_element_offset_end_current[i] > global_element_offset_end_new[i]) ? std::min(global_element_offset_end_current[i] - global_element_offset_end_new[i], n_elements_per_proc[i]) : size_t{0};
        n_elements_recv_right[i] = (global_element_offset_end_new[i] > global_element_offset_end_current[i]) ? std::min(global_element_offset_end_new[i] - global_element_offset_end_current[i], n_elements_new[i]) : size_t{0};
    }

    if (n_elements_send_left[global_rank] + n_elements_recv_left[global_rank] + n_elements_send_right[global_rank] + n_elements_recv_right[global_rank] > 0) {
        // MPI interfaces
        std::vector<int> mpi_interfaces_new_process_outgoing(mpi_interfaces_origin_.size(), global_rank);
        std::vector<size_t> mpi_interfaces_new_local_index_outgoing(mpi_interfaces_origin_);

        std::vector<int> mpi_interfaces_new_process_incoming(mpi_interfaces_destination_.size());
        std::vector<size_t> mpi_interfaces_new_local_index_incoming(mpi_interfaces_destination_.size());
        std::vector<size_t> mpi_interfaces_new_side_incoming(mpi_interfaces_destination_.size());

        for (size_t i = 0; i < mpi_interfaces_new_local_index_outgoing.size(); ++i) {
            if (mpi_interfaces_new_local_index_outgoing[i] < n_elements_send_left[global_rank] || mpi_interfaces_new_local_index_outgoing[i] >= n_elements_per_proc[global_rank] - n_elements_send_right[global_rank]) {
                const size_t element_global_index = mpi_interfaces_new_local_index_outgoing[i] + global_element_offset_current[global_rank];
                
                mpi_interfaces_new_process_outgoing[i] = -1;

                for (int process_index = 0; process_index < global_size; ++process_index) {
                    if (element_global_index >= global_element_offset_new[process_index] && element_global_index < global_element_offset_end_new[process_index]) {
                        mpi_interfaces_new_process_outgoing[i] = process_index;
                        mpi_interfaces_new_local_index_outgoing[i] = element_global_index - global_element_offset_new[process_index];
                        break;
                    }
                }

                if (mpi_interfaces_new_process_outgoing[i] == -1) {
                    std::cerr << "Error: MPI interfaces new local index outgoing, " << i << ", global element index " << element_global_index << " is not found in any process. Exiting." << std::endl;
                    exit(62);
                }
            }
            else {
                mpi_interfaces_new_local_index_outgoing[i] += n_elements_recv_left[global_rank];
                mpi_interfaces_new_local_index_outgoing[i] -= n_elements_send_left[global_rank];
            }
        }

        std::vector<MPI_Request> mpi_interfaces_requests(2 * mpi_load_balancing_interfaces_n_transfers * mpi_interfaces_process_.size());

        for (size_t i = 0; i < mpi_interfaces_process_.size(); ++i) {
            MPI_Isend(mpi_interfaces_new_process_outgoing.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]), MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * (mpi_interfaces_process_.size() + i)]);
            MPI_Irecv(mpi_interfaces_new_process_incoming.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank), MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * i]);

            MPI_Isend(mpi_interfaces_new_local_index_outgoing.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 1, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * (mpi_interfaces_process_.size() + i) + 1]);
            MPI_Irecv(mpi_interfaces_new_local_index_incoming.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 1, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * i + 1]);
        
            MPI_Isend(mpi_interfaces_origin_side_.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 2, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * (mpi_interfaces_process_.size() + i) + 2]);
            MPI_Irecv(mpi_interfaces_new_side_incoming.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 2, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * i + 2]);
        }

        std::vector<MPI_Status> mpi_interfaces_statuses(mpi_interfaces_requests.size());
        MPI_Waitall(mpi_interfaces_requests.size(), mpi_interfaces_requests.data(), mpi_interfaces_statuses.data());

        const MPI_Datatype float_data_type = (sizeof(hostFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
        Element2D_t::Datatype element_data_type;

        // Sending and receiving
        std::vector<Element2D_t> elements_send_left(n_elements_send_left[global_rank]);
        std::vector<Element2D_t> elements_send_right(n_elements_send_right[global_rank]);
        std::vector<hostFloat> solution_arrays_send_left(3 * n_elements_send_left[global_rank] * std::pow(maximum_N_ + 1, 2));
        std::vector<hostFloat> solution_arrays_send_right(3 * n_elements_send_right[global_rank] * std::pow(maximum_N_ + 1, 2));
        std::vector<size_t> n_neighbours_arrays_send_left(4 * n_elements_send_left[global_rank]); // CHECK this only works on quadrilaterals
        std::vector<size_t> n_neighbours_arrays_send_right(4 * n_elements_send_right[global_rank]); // CHECK this only works on quadrilaterals
        std::vector<hostFloat> nodes_arrays_send_left(8 * n_elements_send_left[global_rank]); // CHECK this only works on quadrilaterals
        std::vector<hostFloat> nodes_arrays_send_right(8 * n_elements_send_right[global_rank]); // CHECK this only works on quadrilaterals
        std::vector<int> destination_process_send_left(n_elements_send_left[global_rank]);
        std::vector<int> destination_process_send_right(n_elements_send_right[global_rank]);
        std::vector<size_t> destination_local_index_send_left(n_elements_send_left[global_rank]);
        std::vector<size_t> destination_local_index_send_right(n_elements_send_right[global_rank]);


        // Finding out where to send elements
        for (size_t i = 0; i < n_elements_send_left[global_rank]; ++i) {
            const size_t element_global_index = global_element_offset_current[global_rank] + i;
            destination_process_send_left[i] = -1;

            for (int process_index = 0; process_index < global_size; ++process_index) {
                if (element_global_index >= global_element_offset_new[process_index] && element_global_index < global_element_offset_end_new[process_index]) {
                    destination_process_send_left[i] = process_index;
                    destination_local_index_send_left[i] = element_global_index - global_element_offset_new[process_index];
                    break;
                }
            }

            if (destination_process_send_left[i] == -1) {
                std::cerr << "Error: Left element to send " << i << ", global element index " << element_global_index << " is not found in any process. Exiting." << std::endl;
                exit(63);
            }
        }

        for (size_t i = 0; i < n_elements_send_right[global_rank]; ++i) {
            const size_t element_global_index = global_element_offset_end_current[global_rank] - n_elements_send_right[global_rank] + i;
            destination_process_send_right[i] = -1;

            for (int process_index = 0; process_index < global_size; ++process_index) {
                if (element_global_index >= global_element_offset_new[process_index] && element_global_index < global_element_offset_end_new[process_index]) {
                    destination_process_send_right[i] = process_index;
                    destination_local_index_send_right[i] = element_global_index - global_element_offset_new[process_index];
                    break;
                }
            }

            if (destination_process_send_right[i] == -1) {
                std::cerr << "Error: Right element to send " << i << ", global element index " << element_global_index << " is not found in any process. Exiting." << std::endl;
                exit(64);
            }
        }

        size_t n_send_processes_left = 0;
        int current_process = global_rank;
        for (const auto send_rank : destination_process_send_left) {
            if (send_rank != current_process) {
                current_process = send_rank;
                ++n_send_processes_left;
            }
        }

        size_t n_send_processes_right = 0;
        current_process = global_rank;
        for (const auto send_rank : destination_process_send_right) {
            if (send_rank != current_process) {
                current_process = send_rank;
                ++n_send_processes_right;
            }
        }
        
        if (n_elements_send_left[global_rank] > 0) {
            for (size_t i = 0; i < n_elements_send_left[global_rank]; ++i) {
                elements_send_left[i] = elements_[i]; // CHECK unnecessary vector copy
            }
            
            SEM::Host::Meshes::get_transfer_solution(n_elements_send_left[global_rank], elements_.data(), maximum_N_, nodes_.data(), solution_arrays_send_left.data(), n_neighbours_arrays_send_left.data(), nodes_arrays_send_left.data());
        }

        if (n_elements_send_right[global_rank] > 0) {
            for (size_t i = 0; i < n_elements_send_right[global_rank]; ++i) {
                elements_send_right[i] = elements_[ n_elements_per_proc[global_rank] - n_elements_send_right[global_rank] + i]; // CHECK unnecessary vector copy
            }
            
            SEM::Host::Meshes::get_transfer_solution(n_elements_send_right[global_rank], elements_.data() + n_elements_per_proc[global_rank] - n_elements_send_right[global_rank], maximum_N_, nodes_.data(), solution_arrays_send_right.data(), n_neighbours_arrays_send_right.data(), nodes_arrays_send_right.data());
        }

        size_t n_neighbours_send_total_left = 0;
        size_t n_neighbours_send_total_right = 0;

        for (const auto n_neighbours : n_neighbours_arrays_send_left) {
            n_neighbours_send_total_left += n_neighbours;
        }
        for (const auto n_neighbours : n_neighbours_arrays_send_right) {
            n_neighbours_send_total_right += n_neighbours;
        }

        std::vector<size_t> neighbours_arrays_send_left(n_neighbours_send_total_left);
        std::vector<size_t> neighbours_arrays_send_right(n_neighbours_send_total_right);
        std::vector<hostFloat> neighbours_nodes_arrays_send_left(4 * n_neighbours_send_total_left);
        std::vector<hostFloat> neighbours_nodes_arrays_send_right(4 * n_neighbours_send_total_right);
        std::vector<int> neighbours_proc_arrays_send_left(n_neighbours_send_total_left);
        std::vector<int> neighbours_proc_arrays_send_right(n_neighbours_send_total_right);
        std::vector<size_t> neighbours_side_arrays_send_left(n_neighbours_send_total_left);
        std::vector<size_t> neighbours_side_arrays_send_right(n_neighbours_send_total_right);
        std::vector<int> neighbours_N_arrays_send_left(n_neighbours_send_total_left);
        std::vector<int> neighbours_N_arrays_send_right(n_neighbours_send_total_right);
        std::vector<size_t> process_n_neighbours_send_left(n_send_processes_left, 0);
        std::vector<size_t> process_n_neighbours_send_right(n_send_processes_right, 0);
        std::vector<int> destination_processes_left(n_send_processes_left);        
        std::vector<int> destination_processes_right(n_send_processes_right);
        std::vector<size_t> process_offset_send_left(n_send_processes_left, 0);
        std::vector<size_t> process_offset_send_right(n_send_processes_right, 0);
        std::vector<size_t> process_size_send_left(n_send_processes_left, 0);
        std::vector<size_t> process_size_send_right(n_send_processes_right, 0);
        std::vector<size_t> process_neighbour_offset_send_left(n_send_processes_left, 0);
        std::vector<size_t> process_neighbour_offset_send_right(n_send_processes_right, 0);

        // Fetching data to send
        std::vector<size_t> neighbour_offsets_send_left(n_elements_send_left[global_rank], 0);
        if (n_elements_send_left[global_rank] > 0) {
            size_t n_neighbours_total = 0;
            for (size_t i = 0; i < n_elements_send_left[global_rank]; ++i) {
                neighbour_offsets_send_left[i] = n_neighbours_total;
                n_neighbours_total += n_neighbours_arrays_send_left[4 * i] + n_neighbours_arrays_send_left[4 * i + 1] + n_neighbours_arrays_send_left[4 * i + 2] + n_neighbours_arrays_send_left[4 * i + 3];
            }

            SEM::Host::Meshes::get_neighbours(
                n_elements_send_left[global_rank], 
                0, 
                n_elements_per_proc[global_rank], 
                wall_boundaries_.size(), 
                symmetry_boundaries_.size(), 
                inflow_boundaries_.size(), 
                outflow_boundaries_.size(), 
                interfaces_origin_.size(), 
                mpi_interfaces_destination_.size(), 
                global_rank, 
                global_size, 
                elements_.data(), 
                faces_.data(), 
                nodes_.data(), 
                wall_boundaries_.data(),
                symmetry_boundaries_.data(), 
                inflow_boundaries_.data(), 
                outflow_boundaries_.data(), 
                interfaces_destination_.data(), 
                interfaces_origin_.data(), 
                mpi_interfaces_destination_.data(), 
                mpi_interfaces_new_process_incoming.data(), 
                mpi_interfaces_new_local_index_incoming.data(), 
                mpi_interfaces_new_side_incoming.data(), 
                neighbour_offsets_send_left.data(), 
                n_elements_recv_left.data(), 
                n_elements_send_left.data(), 
                n_elements_recv_right.data(), 
                n_elements_send_right.data(),
                global_element_offset_current.data(), 
                global_element_offset_new.data(), 
                global_element_offset_end_new.data(), 
                neighbours_arrays_send_left.data(), 
                neighbours_nodes_arrays_send_left.data(), 
                neighbours_proc_arrays_send_left.data(), 
                neighbours_side_arrays_send_left.data(), 
                neighbours_N_arrays_send_left.data());

            if (n_send_processes_left > 0) {
                destination_processes_left[0] = destination_process_send_left[0];
            }

            size_t process_index = 0;
            size_t process_current_offset = 0;
            size_t process_current_neighbour_offset = 0;
            for (size_t i = 0; i < n_elements_send_left[global_rank]; ++i) {
                const int send_rank = destination_process_send_left[i];
                if (send_rank != destination_processes_left[process_index]) {
                    process_current_neighbour_offset += process_n_neighbours_send_left[process_index];
                    process_current_offset += process_size_send_left[process_index];
                    ++process_index;
                    process_offset_send_left[process_index] = process_current_offset;
                    destination_processes_left[process_index] = send_rank;
                    process_neighbour_offset_send_left[process_index] = process_current_neighbour_offset;
                }

                ++process_size_send_left[process_index];
                process_n_neighbours_send_left[process_index] += n_neighbours_arrays_send_left[4 * i] + n_neighbours_arrays_send_left[4 * i + 1] + n_neighbours_arrays_send_left[4 * i + 2] + n_neighbours_arrays_send_left[4 * i + 3];
            }
        }       

        std::vector<size_t> neighbour_offsets_send_right(n_elements_send_right[global_rank], 0);
        if (n_elements_send_right[global_rank] > 0) {
            size_t n_neighbours_total = 0;
            for (size_t i = 0; i < n_elements_send_right[global_rank]; ++i) {
                neighbour_offsets_send_right[i] = n_neighbours_total;
                n_neighbours_total += n_neighbours_arrays_send_right[4 * i] + n_neighbours_arrays_send_right[4 * i + 1] + n_neighbours_arrays_send_right[4 * i + 2] + n_neighbours_arrays_send_right[4 * i + 3];
            }

            SEM::Host::Meshes::get_neighbours(
                n_elements_send_right[global_rank], 
                n_elements_per_proc[global_rank] - n_elements_send_right[global_rank], 
                n_elements_per_proc[global_rank], 
                wall_boundaries_.size(), 
                symmetry_boundaries_.size(), 
                inflow_boundaries_.size(), 
                outflow_boundaries_.size(), 
                interfaces_origin_.size(), 
                mpi_interfaces_destination_.size(), 
                global_rank, 
                global_size, 
                elements_.data(), 
                faces_.data(), 
                nodes_.data(), 
                wall_boundaries_.data(), 
                symmetry_boundaries_.data(), 
                inflow_boundaries_.data(), 
                outflow_boundaries_.data(), 
                interfaces_destination_.data(), 
                interfaces_origin_.data(), 
                mpi_interfaces_destination_.data(), 
                mpi_interfaces_new_process_incoming.data(), 
                mpi_interfaces_new_local_index_incoming.data(), 
                mpi_interfaces_new_side_incoming.data(), 
                neighbour_offsets_send_right.data(), 
                n_elements_recv_left.data(), 
                n_elements_send_left.data(), 
                n_elements_recv_right.data(), 
                n_elements_send_right.data(), 
                global_element_offset_current.data(), 
                global_element_offset_new.data(), 
                global_element_offset_end_new.data(), 
                neighbours_arrays_send_right.data(), 
                neighbours_nodes_arrays_send_right.data(), 
                neighbours_proc_arrays_send_right.data(), 
                neighbours_side_arrays_send_right.data(), 
                neighbours_N_arrays_send_right.data());

            // Compute the global indices of the elements using global_element_offset_end_current, and where they are going
            if (n_send_processes_right > 0) {
                destination_processes_right[0] = destination_process_send_right[0];
            }

            size_t process_index = 0;
            size_t process_current_offset = 0;
            size_t process_current_neighbour_offset = 0;
            for (size_t i = 0; i < n_elements_send_right[global_rank]; ++i) {
                const int send_rank = destination_process_send_right[i];
                if (send_rank != destination_processes_right[process_index]) {
                    process_current_neighbour_offset += process_n_neighbours_send_right[process_index];
                    process_current_offset += process_size_send_right[process_index];
                    ++process_index;
                    process_offset_send_right[process_index] = process_current_offset;
                    destination_processes_right[process_index] = send_rank;
                    process_neighbour_offset_send_right[process_index] = process_current_neighbour_offset;
                }

                ++process_size_send_right[process_index];
                process_n_neighbours_send_right[process_index] += n_neighbours_arrays_send_right[4 * i] + n_neighbours_arrays_send_right[4 * i + 1] + n_neighbours_arrays_send_right[4 * i + 2] + n_neighbours_arrays_send_right[4 * i + 3];
            }
        }

        // Setting up receive buffers
        std::vector<Element2D_t> elements_recv_left(n_elements_recv_left[global_rank]);
        std::vector<Element2D_t> elements_recv_right(n_elements_recv_right[global_rank]);
        std::vector<size_t> n_neighbours_arrays_recv_left(4 * n_elements_recv_left[global_rank]);
        std::vector<size_t> n_neighbours_arrays_recv_right(4 * n_elements_recv_right[global_rank]);
        std::vector<hostFloat> nodes_arrays_recv(8 * (n_elements_recv_left[global_rank] + n_elements_recv_right[global_rank]));
        std::vector<hostFloat> solution_arrays_recv_left(3 * n_elements_recv_left[global_rank] * std::pow(maximum_N_ + 1, 2));
        std::vector<hostFloat> solution_arrays_recv_right(3 * n_elements_recv_right[global_rank] * std::pow(maximum_N_ + 1, 2));

        std::vector<int> origin_process_recv_left(n_elements_recv_left[global_rank]);
        std::vector<int> origin_process_recv_right(n_elements_recv_right[global_rank]);
        size_t n_recv_processes_left = 0;
        size_t n_recv_processes_right = 0;

        if (n_elements_recv_left[global_rank] > 0) {
            // Compute the global indices of the elements using global_element_offset_new, and where they are coming from
            for (size_t i = 0; i < n_elements_recv_left[global_rank]; ++i) {
                const size_t global_element_indices_recv = global_element_offset_new[global_rank] + i;

                bool missing = true;
                for (int j = 0; j < global_size; ++j) {
                    if (global_element_indices_recv >= global_element_offset_current[j] && global_element_indices_recv < global_element_offset_end_current[j]) {
                        origin_process_recv_left[i] = j;
                        missing = false;
                        break;
                    }
                }
                if (missing) {
                    std::cerr << "Error: Process " << global_rank << " should receive element with global index " << global_element_indices_recv << " as its left received element #" << i << ", but the element is not within any process' range. Exiting." << std::endl;
                    exit(54);
                }
            }

            int current_process = global_rank;
            for (const auto recv_rank : origin_process_recv_left) {
                if (recv_rank != current_process) {
                    current_process = recv_rank;
                    ++n_recv_processes_left;
                }
            }
        }

        if (n_elements_recv_right[global_rank] > 0) {
            // Compute the global indices of the elements using global_element_offset_new, and where they are coming from
            for (size_t i = 0; i < n_elements_recv_right[global_rank]; ++i) {
                const size_t global_element_indices_recv = global_element_offset_end_new[global_rank] - n_elements_recv_right[global_rank] + i;

                bool missing = true;
                for (int j = 0; j < global_size; ++j) {
                    if (global_element_indices_recv >= global_element_offset_current[j] && global_element_indices_recv < global_element_offset_end_current[j]) {
                        origin_process_recv_right[i] = j;
                        missing = false;
                        break;
                    }
                }
                if (missing) {
                    std::cerr << "Error: Process " << global_rank << " should receive element with global index " << global_element_indices_recv << " as its right received element #" << i << ", but the element is not within any process' range. Exiting." << std::endl;
                    exit(55);
                }
            }

            int current_process = global_rank;
            for (const auto recv_rank : origin_process_recv_right) {
                if (recv_rank != current_process) {
                    current_process = recv_rank;
                    ++n_recv_processes_right;
                }
            }
        }

        std::vector<int> origin_processes_left = std::vector<int>(n_recv_processes_left);
        std::vector<int> origin_processes_right = std::vector<int>(n_recv_processes_right);
        std::vector<size_t> process_offset_left(origin_processes_left.size(), 0);            
        std::vector<size_t> process_offset_right(origin_processes_right.size(), 0);
        std::vector<size_t> process_size_left(origin_processes_left.size(), 0);
        std::vector<size_t> process_size_right(origin_processes_right.size(), 0);

        std::vector<size_t> process_n_neighbours_recv_left(origin_processes_left.size(), 0);
        std::vector<size_t> process_n_neighbours_recv_right(origin_processes_right.size(), 0);

        if (n_elements_recv_left[global_rank] > 0) {
            if (origin_processes_left.size() > 0) {
                origin_processes_left[0] = origin_process_recv_left[0];
            }

            size_t process_index = 0;
            size_t process_current_offset = 0;
            for (size_t i = 0; i < n_elements_recv_left[global_rank]; ++i) {
                const int recv_rank = origin_process_recv_left[i];
                if (recv_rank != origin_processes_left[process_index]) {
                    process_current_offset += process_size_left[process_index];
                    ++process_index;
                    process_offset_left[process_index] = process_current_offset;
                    origin_processes_left[process_index] = recv_rank;
                }

                ++process_size_left[process_index];
            }
        }

        if (n_elements_recv_right[global_rank] > 0) {
            if (origin_processes_right.size() > 0) {
                origin_processes_right[0] = origin_process_recv_right[0];
            }

            size_t process_index = 0;
            size_t process_current_offset = 0;
            for (size_t i = 0; i < n_elements_recv_right[global_rank]; ++i) {
                const int recv_rank = origin_process_recv_right[i];
                if (recv_rank != origin_processes_right[process_index]) {
                    process_current_offset += process_size_right[process_index];
                    ++process_index;
                    process_offset_right[process_index] = process_current_offset;
                    origin_processes_right[process_index] = recv_rank;
                }

                ++process_size_right[process_index];
            }
        }

        // Sending n neighbours
        std::vector<MPI_Request> mpi_interfaces_n_neighbours_requests(mpi_load_balancing_n_neighbours_n_transfers * (destination_processes_left.size() + destination_processes_right.size() + origin_processes_left.size() + origin_processes_right.size()));
        
        for (size_t i = 0; i < destination_processes_left.size(); ++i) {
            MPI_Isend(&process_n_neighbours_send_left[i], 1, size_t_data_type, destination_processes_left[i], mpi_load_balancing_n_neighbours_offset * global_size * global_size + mpi_load_balancing_n_neighbours_n_transfers * (global_size * global_rank + destination_processes_left[i]), MPI_COMM_WORLD, &mpi_interfaces_n_neighbours_requests[mpi_load_balancing_n_neighbours_n_transfers * (origin_processes_left.size() + origin_processes_right.size() + i)]);
        }

        for (size_t i = 0; i < destination_processes_right.size(); ++i) {
            MPI_Isend(&process_n_neighbours_send_right[i], 1, size_t_data_type, destination_processes_right[i], mpi_load_balancing_n_neighbours_offset * global_size * global_size + mpi_load_balancing_n_neighbours_n_transfers * (global_size * global_rank + destination_processes_right[i]), MPI_COMM_WORLD, &mpi_interfaces_n_neighbours_requests[mpi_load_balancing_n_neighbours_n_transfers * (origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size() + i)]);
        }

        // Receiving n neighbours
        for (size_t i = 0; i < origin_processes_left.size(); ++i) {
            MPI_Irecv(&process_n_neighbours_recv_left[i], 1, size_t_data_type, origin_processes_left[i], mpi_load_balancing_n_neighbours_offset * global_size * global_size + mpi_load_balancing_n_neighbours_n_transfers * (global_size * origin_processes_left[i] + global_rank), MPI_COMM_WORLD, &mpi_interfaces_n_neighbours_requests[mpi_load_balancing_n_neighbours_n_transfers * i]);
        }

        for (size_t i = 0; i < origin_processes_right.size(); ++i) {
            MPI_Irecv(&process_n_neighbours_recv_right[i], 1, size_t_data_type, origin_processes_right[i], mpi_load_balancing_n_neighbours_offset * global_size * global_size + mpi_load_balancing_n_neighbours_n_transfers * (global_size * origin_processes_right[i] + global_rank), MPI_COMM_WORLD, &mpi_interfaces_n_neighbours_requests[mpi_load_balancing_n_neighbours_n_transfers * (origin_processes_left.size() + i)]);
        }

        std::vector<MPI_Status> mpi_interfaces_n_neighbours_statuses(mpi_interfaces_n_neighbours_requests.size());
        MPI_Waitall(mpi_interfaces_n_neighbours_requests.size(), mpi_interfaces_n_neighbours_requests.data(), mpi_interfaces_n_neighbours_statuses.data());

        size_t n_neighbours_recv_total_left = 0;
        size_t n_neighbours_recv_total_right = 0;
        std::vector<size_t> process_neighbour_offset_left(origin_processes_left.size());
        std::vector<size_t> process_neighbour_offset_right(origin_processes_right.size());
        
        for (size_t i = 0; i < origin_processes_left.size(); ++i) {
            process_neighbour_offset_left[i] = n_neighbours_recv_total_left;
            n_neighbours_recv_total_left += process_n_neighbours_recv_left[i];
        }

        for (size_t i = 0; i < origin_processes_right.size(); ++i) {
            process_neighbour_offset_right[i] = n_neighbours_recv_total_right;
            n_neighbours_recv_total_right += process_n_neighbours_recv_right[i];
        }

        std::vector<size_t> neighbours_arrays_recv(n_neighbours_recv_total_left + n_neighbours_recv_total_right);
        std::vector<hostFloat> neighbours_nodes_arrays_recv(4 * neighbours_arrays_recv.size());
        std::vector<int> neighbours_proc_arrays_recv(neighbours_arrays_recv.size());
        std::vector<size_t> neighbours_side_arrays_recv(neighbours_arrays_recv.size());
        std::vector<int> neighbours_N_arrays_recv(neighbours_arrays_recv.size());

        // Sending data
        std::vector<MPI_Request> mpi_interfaces_solution_requests(mpi_load_balancing_solution_n_transfers * (destination_processes_left.size() + destination_processes_right.size() + origin_processes_left.size() + origin_processes_right.size()));
        
        for (size_t i = 0; i < destination_processes_left.size(); ++i) {
            // Neighbours
            MPI_Isend(n_neighbours_arrays_send_left.data() + 4 * process_offset_send_left[i], 4 * process_size_send_left[i], size_t_data_type, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]), MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size())]);
            MPI_Isend(neighbours_arrays_send_left.data() + process_neighbour_offset_send_left[i], process_n_neighbours_send_left[i], size_t_data_type, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 1, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 1]);
            MPI_Isend(neighbours_nodes_arrays_send_left.data() + 4 * process_neighbour_offset_send_left[i], 4 * process_n_neighbours_send_left[i], float_data_type, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 7, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 7]);
            MPI_Isend(neighbours_proc_arrays_send_left.data() + process_neighbour_offset_send_left[i], process_n_neighbours_send_left[i], MPI_INT, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 2, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 2]);
            MPI_Isend(neighbours_side_arrays_send_left.data() + process_neighbour_offset_send_left[i], process_n_neighbours_send_left[i], size_t_data_type, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 6, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 6]);
            MPI_Isend(neighbours_N_arrays_send_left.data() + process_neighbour_offset_send_left[i], process_n_neighbours_send_left[i], MPI_INT, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 8, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 8]);

            // Nodes
            MPI_Isend(nodes_arrays_send_left.data() + 8 * process_offset_send_left[i], 8 * process_size_send_left[i], float_data_type, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 3, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 3]);

            // Solution
            MPI_Isend(solution_arrays_send_left.data() + 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_offset_send_left[i], 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_size_send_left[i], float_data_type, destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 4, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 4]);

            // Elements
            MPI_Isend(elements_send_left.data() + process_offset_send_left[i], process_size_send_left[i], element_data_type.data(), destination_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_left[i]) + 5, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size()) + 5]);
        }

        for (size_t i = 0; i < destination_processes_right.size(); ++i) {
            // Neighbours
            MPI_Isend(n_neighbours_arrays_send_right.data() + 4 * process_offset_send_right[i], 4 * process_size_send_right[i], size_t_data_type, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]), MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size())]);
            MPI_Isend(neighbours_arrays_send_right.data() + process_neighbour_offset_send_right[i], process_n_neighbours_send_right[i], size_t_data_type, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 1, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 1]);
            MPI_Isend(neighbours_nodes_arrays_send_right.data() + 4 * process_neighbour_offset_send_right[i], 4 * process_n_neighbours_send_right[i], float_data_type, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 7, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 7]);
            MPI_Isend(neighbours_proc_arrays_send_right.data() + process_neighbour_offset_send_right[i], process_n_neighbours_send_right[i], MPI_INT, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 2, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 2]);
            MPI_Isend(neighbours_side_arrays_send_right.data() + process_neighbour_offset_send_right[i], process_n_neighbours_send_right[i], size_t_data_type, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 6, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 6]);
            MPI_Isend(neighbours_N_arrays_send_right.data() + process_neighbour_offset_send_right[i], process_n_neighbours_send_right[i], MPI_INT, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 8, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 8]);

            // Nodes
            MPI_Isend(nodes_arrays_send_right.data() + 8 * process_offset_send_right[i], 8 * process_size_send_right[i], float_data_type, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 3, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 3]);

            // Solution
            MPI_Isend(solution_arrays_send_right.data() + 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_offset_send_right[i], 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_size_send_right[i], float_data_type, destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 4, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 4]);

            // Elements
            MPI_Isend(elements_send_right.data() + process_offset_send_right[i], process_size_send_right[i], element_data_type.data(), destination_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * global_rank + destination_processes_right[i]) + 5, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size() + origin_processes_right.size() + destination_processes_left.size()) + 5]);
        }

        // Receiving data
        for (size_t i = 0; i < origin_processes_left.size(); ++i) {
            // Neighbours
            MPI_Irecv(n_neighbours_arrays_recv_left.data() + 4 * process_offset_left[i], 4 * process_size_left[i], size_t_data_type, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank), MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i]);

            // Nodes
            MPI_Irecv(nodes_arrays_recv.data() + 8 * process_offset_left[i], 8 * process_size_left[i], float_data_type, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 3, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 1]);

            // Solution
            MPI_Irecv(solution_arrays_recv_left.data() + 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_offset_left[i], 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_size_left[i], float_data_type, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 4, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 2]);

            // Elements
            MPI_Irecv(elements_recv_left.data() + process_offset_left[i], process_size_left[i], element_data_type.data(), origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 5, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 3]);
        
            // Neighbours
            MPI_Irecv(neighbours_arrays_recv.data() + process_neighbour_offset_left[i], process_n_neighbours_recv_left[i], size_t_data_type, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 1, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 4]);
            MPI_Irecv(neighbours_nodes_arrays_recv.data() + 4 * process_neighbour_offset_left[i], 4 * process_n_neighbours_recv_left[i], float_data_type, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 7, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 7]);
            MPI_Irecv(neighbours_proc_arrays_recv.data() + process_neighbour_offset_left[i], process_n_neighbours_recv_left[i], MPI_INT, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 2, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 5]);
            MPI_Irecv(neighbours_side_arrays_recv.data() + process_neighbour_offset_left[i], process_n_neighbours_recv_left[i], size_t_data_type, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 6, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 6]);
            MPI_Irecv(neighbours_N_arrays_recv.data() + process_neighbour_offset_left[i], process_n_neighbours_recv_left[i], MPI_INT, origin_processes_left[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_left[i] + global_rank) + 8, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * i + 8]);
        }

        for (size_t i = 0; i < origin_processes_right.size(); ++i) {
            // Neighbours
            MPI_Irecv(n_neighbours_arrays_recv_right.data() + 4 * process_offset_right[i], 4 * process_size_right[i], size_t_data_type, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank), MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size())]);

            // Nodes
            MPI_Irecv(nodes_arrays_recv.data() + 8 * (process_offset_right[i] + n_elements_recv_left[global_rank]), 8 * process_size_right[i], float_data_type, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 3, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size()) + 1]);

            // Solution
            MPI_Irecv(solution_arrays_recv_right.data() + 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_offset_right[i], 3 * (maximum_N_ + 1) * (maximum_N_ + 1) * process_size_right[i], float_data_type, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 4, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size()) + 2]);

            // Elements
            MPI_Irecv(elements_recv_right.data() + process_offset_right[i], process_size_right[i], element_data_type.data(), origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 5, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (i + origin_processes_left.size()) + 3]);
        
            // Neighbours
            MPI_Irecv(neighbours_arrays_recv.data() + n_neighbours_recv_total_left + process_neighbour_offset_right[i], process_n_neighbours_recv_right[i], size_t_data_type, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 1, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (origin_processes_left.size() + i) + 4]);
            MPI_Irecv(neighbours_nodes_arrays_recv.data() + 4 * (n_neighbours_recv_total_left + process_neighbour_offset_right[i]), 4 * process_n_neighbours_recv_right[i], float_data_type, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 7, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (origin_processes_left.size() + i) + 7]);
            MPI_Irecv(neighbours_proc_arrays_recv.data() + n_neighbours_recv_total_left + process_neighbour_offset_right[i], process_n_neighbours_recv_right[i], MPI_INT, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 2, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (origin_processes_left.size() + i) + 5]);
            MPI_Irecv(neighbours_side_arrays_recv.data() + n_neighbours_recv_total_left + process_neighbour_offset_right[i], process_n_neighbours_recv_right[i], size_t_data_type, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 6, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (origin_processes_left.size() + i) + 6]);
            MPI_Irecv(neighbours_N_arrays_recv.data() + n_neighbours_recv_total_left + process_neighbour_offset_right[i], process_n_neighbours_recv_right[i], MPI_INT, origin_processes_right[i], mpi_load_balancing_solution_offset * global_size * global_size + mpi_load_balancing_solution_n_transfers * (global_size * origin_processes_right[i] + global_rank) + 8, MPI_COMM_WORLD, &mpi_interfaces_solution_requests[mpi_load_balancing_solution_n_transfers * (origin_processes_left.size() + i) + 8]);
        }

        std::vector<MPI_Status> mpi_interfaces_solution_statuses(mpi_interfaces_solution_requests.size());
        MPI_Waitall(mpi_interfaces_solution_requests.size(), mpi_interfaces_solution_requests.data(), mpi_interfaces_solution_statuses.data());

        std::vector<size_t> neighbour_offsets_left(n_elements_recv_left[global_rank], 0);
        std::vector<size_t> neighbour_offsets_right(n_elements_recv_right[global_rank], 0);

        size_t neighbours_index = 0;
        for (size_t i = 0; i < n_elements_recv_left[global_rank]; ++i) {
            neighbour_offsets_left[i] = neighbours_index;
            neighbours_index += n_neighbours_arrays_recv_left[4 * i] + n_neighbours_arrays_recv_left[4 * i + 1] + n_neighbours_arrays_recv_left[4 * i + 2] + n_neighbours_arrays_recv_left[4 * i + 3];
        }

        for (size_t i = 0; i < n_elements_recv_right[global_rank]; ++i) {
            neighbour_offsets_right[i] = neighbours_index;
            neighbours_index += n_neighbours_arrays_recv_right[4 * i] + n_neighbours_arrays_recv_right[4 * i + 1] + n_neighbours_arrays_recv_right[4 * i + 2] + n_neighbours_arrays_recv_right[4 * i + 3];
        }
    
        // Now everything is received
        // We can do something about the nodes
        std::vector<size_t> received_node_indices(4 * (n_elements_recv_left[global_rank] + n_elements_recv_right[global_rank]));
        std::vector<size_t> received_neighbour_node_indices(2 * neighbours_arrays_recv.size());
        if (n_elements_recv_left[global_rank] + n_elements_recv_right[global_rank] > 0) {
            // Elements received nodes
            std::vector<bool> missing_nodes(received_node_indices.size());
            std::vector<bool> missing_received_nodes(received_node_indices.size());
            std::vector<size_t> received_node_received_indices(received_node_indices.size()); // Worst naming convention ever, these are nodes that have been found in other received nodes

            SEM::Host::Meshes::find_received_nodes(nodes_.size(), nodes_.data(), nodes_arrays_recv.data(), missing_nodes, missing_received_nodes, received_node_indices.data(), received_node_received_indices.data());

            // Neighbours received nodes
            std::vector<bool> missing_neighbour_nodes(received_neighbour_node_indices.size());
            std::vector<bool> missing_received_neighbour_nodes(received_neighbour_node_indices.size());
            std::vector<size_t> received_neighbour_node_received_indices(received_neighbour_node_indices.size()); // Worst naming convention ever, these are nodes that have been found in other received nodes

            SEM::Host::Meshes::find_received_neighbour_nodes(missing_nodes.size(), nodes_.size(), nodes_.data(), neighbours_nodes_arrays_recv.data(), nodes_arrays_recv.data(), missing_neighbour_nodes, missing_received_neighbour_nodes, received_neighbour_node_indices.data(), received_neighbour_node_received_indices.data());

            size_t n_missing_received_nodes = 0;
            for (auto i : missing_nodes) {
                n_missing_received_nodes += i;
            }
            const size_t n_created_received_nodes = n_missing_received_nodes;
            for (auto i : missing_neighbour_nodes) {
                n_missing_received_nodes += i;
            }

            const size_t old_n_nodes = nodes_.size();
            nodes_.resize(nodes_.size() + n_missing_received_nodes);
            SEM::Host::Meshes::add_new_received_nodes(old_n_nodes, nodes_.data(), nodes_arrays_recv.data(), missing_nodes, missing_received_nodes, received_node_indices.data(), received_node_received_indices.data());
        
            SEM::Host::Meshes::add_new_received_neighbour_nodes(old_n_nodes, n_created_received_nodes, nodes_.data(), neighbours_nodes_arrays_recv.data(), missing_neighbour_nodes, missing_nodes, missing_received_neighbour_nodes, received_neighbour_node_indices.data(), received_neighbour_node_received_indices.data());
        }
        // Now something similar for neighbours?

        // Faces
        // Faces to add
        size_t n_faces_to_add_recv_left = 0;
        size_t neighbour_index = 0;
        std::vector<size_t> face_offsets_left(n_elements_recv_left[global_rank]);
        for (size_t i = 0; i < n_elements_recv_left[global_rank]; ++i) {
            const size_t element_index = i;
            face_offsets_left[i] = n_faces_to_add_recv_left;
            for (size_t j = 0; j < 4; ++j) {
                const size_t side_n_neighbours = n_neighbours_arrays_recv_left[4 * i + j];
                for (size_t k = 0; k < n_neighbours_arrays_recv_left[4 * i + j]; ++k) {
                    const int neighbour_process = neighbours_proc_arrays_recv[neighbour_index + k];
                    const size_t neighbour_element_index = neighbours_arrays_recv[neighbour_index + k];

                    if (neighbour_process != global_rank
                            || neighbour_element_index < n_elements_recv_left[global_rank] && neighbour_element_index >= element_index
                            || neighbour_element_index >= n_elements_new[global_rank] - n_elements_recv_right[global_rank] && neighbour_element_index >= element_index) {
                        
                        ++n_faces_to_add_recv_left;
                    }
                }
                neighbour_index += side_n_neighbours;
            }
        }
        size_t n_faces_to_add_recv_right = 0;
        neighbour_index = n_neighbours_recv_total_left; // Maybe error check here? should already be there
        std::vector<size_t> face_offsets_right(n_elements_recv_right[global_rank]);
        for (size_t i = 0; i < n_elements_recv_right[global_rank]; ++i) {
            const size_t element_index = n_elements_new[global_rank] + 1 - n_elements_recv_right[global_rank] + i;
            face_offsets_right[i] = n_faces_to_add_recv_left + n_faces_to_add_recv_right;
            for (size_t j = 0; j < 4; ++j) {
                const size_t side_n_neighbours = n_neighbours_arrays_recv_right[4 * i + j];
                for (size_t k = 0; k < side_n_neighbours; ++k) {
                    const int neighbour_process = neighbours_proc_arrays_recv[neighbour_index + k];
                    const size_t neighbour_element_index = neighbours_arrays_recv[neighbour_index + k];

                    if (neighbour_process != global_rank
                            || neighbour_element_index < n_elements_recv_left[global_rank] && neighbour_element_index >= element_index
                            || neighbour_element_index >= n_elements_new[global_rank] - n_elements_recv_right[global_rank] && neighbour_element_index >= element_index) {

                        ++n_faces_to_add_recv_right;
                    }
                }
                neighbour_index += side_n_neighbours;
            }
        }
        const size_t n_faces_to_add = n_faces_to_add_recv_left + n_faces_to_add_recv_right;

         // Faces to delete
        std::vector<bool> faces_to_delete(faces_.size());
        size_t n_faces_to_delete = 0;
        if (n_elements_send_left[global_rank] + n_elements_send_right[global_rank] > 0) {
            SEM::Host::Meshes::find_faces_to_delete(n_elements_, n_elements_send_left[global_rank], n_elements_send_right[global_rank], faces_.data(), faces_to_delete);

            for (auto i : faces_to_delete) {
                n_faces_to_delete += i;
            }
        }
        else {
            SEM::Host::Meshes::no_faces_to_delete(faces_to_delete);
        }

        std::vector<bool> boundary_elements_to_delete(elements_.size() - n_elements_);
        SEM::Host::Meshes::find_boundary_elements_to_delete(elements_.size() - n_elements_, n_elements_, n_elements_send_left[global_rank], n_elements_send_right[global_rank], elements_.data(), faces_.data(), boundary_elements_to_delete, faces_to_delete);

        size_t n_wall_boundaries_to_add_recv = 0;
        size_t n_symmetry_boundaries_to_add_recv = 0;
        size_t n_inflow_boundaries_to_add_recv = 0;
        size_t n_outflow_boundaries_to_add_recv = 0;
        size_t n_mpi_destinations_to_add_recv = 0;

        if (n_elements_recv_left[global_rank] + n_elements_recv_right[global_rank] > 0) {
            for (int i = 0; i < neighbours_proc_arrays_recv.size(); ++i) {
                const int neighbour_proc = neighbours_proc_arrays_recv[i];
        
                switch (neighbour_proc) {
                    case SEM::Host::Meshes::Mesh2D_t::boundary_type::wall :
                        ++n_wall_boundaries_to_add_recv;
                        break;

                    case SEM::Host::Meshes::Mesh2D_t::boundary_type::symmetry :
                        ++n_symmetry_boundaries_to_add_recv;
                        break;

                    case SEM::Host::Meshes::Mesh2D_t::boundary_type::inflow :
                        ++n_inflow_boundaries_to_add_recv;
                        break;

                    case SEM::Host::Meshes::Mesh2D_t::boundary_type::outflow :
                        ++n_outflow_boundaries_to_add_recv;
                        break;

                    default:
                        if (neighbour_proc != global_rank) {
                            const size_t local_element_index = neighbours_arrays_recv[i];
                            const size_t element_side_index = neighbours_side_arrays_recv[i];

                            bool first_time = true;
                            for (size_t j = 0; j < mpi_interfaces_new_process_incoming.size(); ++j) {
                                if (mpi_interfaces_new_process_incoming[j] == neighbour_proc && mpi_interfaces_new_local_index_incoming[j] == local_element_index && mpi_interfaces_new_side_incoming[j] == element_side_index) {
                                    first_time = false;
                                    break;
                                }
                            }
                            if (first_time) {
                                for (size_t j = 0; j < i; ++j) {
                                    if (neighbours_proc_arrays_recv[j] == neighbour_proc && neighbours_arrays_recv[j] == local_element_index && neighbours_side_arrays_recv[j] == element_side_index) {
                                        first_time = false;
                                        break;
                                    }
                                }
                            }
                            if (first_time) {
                                ++n_mpi_destinations_to_add_recv;
                            }
                        }
                }
            }

            SEM::Host::Meshes::find_mpi_interface_elements_to_delete(mpi_interfaces_destination_.size(), n_elements_, global_rank, mpi_interfaces_destination_.data(), mpi_interfaces_new_process_incoming.data(), boundary_elements_to_delete);
            SEM::Host::Meshes::find_mpi_interface_elements_to_keep(mpi_interfaces_destination_.size(), neighbours_proc_arrays_recv.size(), global_rank, n_elements_, neighbours_proc_arrays_recv.data(), neighbours_arrays_recv.data(), neighbours_side_arrays_recv.data(), mpi_interfaces_new_process_incoming.data(), mpi_interfaces_new_local_index_incoming.data(), mpi_interfaces_new_side_incoming.data(), mpi_interfaces_destination_.data(), boundary_elements_to_delete);
        }

        size_t n_boundary_elements_to_delete = 0;
        for (auto i: boundary_elements_to_delete) {
            n_boundary_elements_to_delete += i;
        }

        // Boundary elements to add
        std::vector<size_t> elements_send_destinations_offset_left(n_elements_send_left[global_rank]);
        std::vector<size_t> elements_send_destinations_offset_right(n_elements_send_right[global_rank]);
        std::vector<int> elements_send_destinations_keep_left(4 * n_elements_send_left[global_rank], 0);
        std::vector<int> elements_send_destinations_keep_right(4 * n_elements_send_right[global_rank], 0);

        size_t n_mpi_destinations_to_add_send_left = 0;
        size_t neighbour_index_send = 0;
        for (size_t i = 0; i < n_elements_send_left[global_rank]; ++i) {
            elements_send_destinations_offset_left[i] = n_mpi_destinations_to_add_send_left;
            for (size_t j = 0; j < 4; ++j) {
                const size_t side_n_neighbours = n_neighbours_arrays_send_left[4 * i + j];
                for (size_t k = 0; k < side_n_neighbours; ++k) {
                    if (neighbours_proc_arrays_send_left[neighbour_index_send + k] == global_rank && neighbours_arrays_send_left[neighbour_index_send + k] >= n_elements_recv_left[global_rank] && neighbours_arrays_send_left[neighbour_index_send + k] < n_elements_new[global_rank] - n_elements_recv_right[global_rank]) {
                        ++n_mpi_destinations_to_add_send_left;
                        elements_send_destinations_keep_left[4 * i + j] = 1;
                        break;
                    }
                }
                neighbour_index_send += side_n_neighbours;
            }
        }

        size_t n_mpi_destinations_to_add_send_right = 0;
        neighbour_index_send = 0;
        for (size_t i = 0; i < n_elements_send_right[global_rank]; ++i) {
            elements_send_destinations_offset_right[i] = n_mpi_destinations_to_add_send_right;
            for (size_t j = 0; j < 4; ++j) {
                const size_t side_n_neighbours = n_neighbours_arrays_send_right[4 * i + j];
                for (size_t k = 0; k < side_n_neighbours; ++k) {
                    if (neighbours_proc_arrays_send_right[neighbour_index_send + k] == global_rank && neighbours_arrays_send_right[neighbour_index_send + k] >= n_elements_recv_left[global_rank] && neighbours_arrays_send_right[neighbour_index_send + k] < n_elements_new[global_rank] - n_elements_recv_right[global_rank]) {
                        ++n_mpi_destinations_to_add_send_right;
                        elements_send_destinations_keep_right[4 * i + j] = 1;
                        break;
                    }
                }
                neighbour_index_send += side_n_neighbours;
            }
        }

        // Moving faces
        std::vector<Face2D_t> new_faces;

        if (n_elements_send_left[global_rank] + n_elements_send_right[global_rank] > 0) {
            new_faces = std::vector<Face2D_t>(faces_.size() + n_faces_to_add - n_faces_to_delete);

            const size_t new_elements_offset_left = n_elements_new[global_rank] + elements_.size() - n_elements_ - n_boundary_elements_to_delete;
            const size_t new_elements_offset_right = n_elements_new[global_rank] + elements_.size() - n_elements_ - n_boundary_elements_to_delete + n_mpi_destinations_to_add_send_left;
            SEM::Host::Meshes::move_required_faces(n_elements_, n_elements_new[global_rank], n_elements_send_left[global_rank], n_elements_send_right[global_rank], n_elements_recv_left[global_rank], n_elements_recv_right[global_rank], new_elements_offset_left, new_elements_offset_right, mpi_interfaces_destination_.size(), global_rank, faces_.data(), new_faces.data(), faces_to_delete, boundary_elements_to_delete, elements_send_destinations_offset_left.data(), elements_send_destinations_offset_right.data(), elements_send_destinations_keep_left.data(), elements_send_destinations_keep_right.data(), mpi_interfaces_destination_.data(), mpi_interfaces_new_process_incoming.data(), mpi_interfaces_new_local_index_incoming.data(), mpi_interfaces_new_side_incoming.data());
        }
        else {
            new_faces = std::vector<Face2D_t>(faces_.size() + n_faces_to_add);

            SEM::Host::Meshes::move_faces(faces_.size(), n_elements_, n_elements_new[global_rank], n_elements_recv_left[global_rank], n_elements_recv_right[global_rank], mpi_interfaces_destination_.size(), global_rank, faces_.data(), new_faces.data(), boundary_elements_to_delete, mpi_interfaces_destination_.data(), mpi_interfaces_new_process_incoming.data(), mpi_interfaces_new_local_index_incoming.data(), mpi_interfaces_new_side_incoming.data());

        }

        // Boundary conditions
        const size_t n_wall_boundaries_to_add = n_wall_boundaries_to_add_recv;
        const size_t n_symmetry_boundaries_to_add = n_symmetry_boundaries_to_add_recv;
        const size_t n_inflow_boundaries_to_add = n_inflow_boundaries_to_add_recv;
        const size_t n_outflow_boundaries_to_add = n_outflow_boundaries_to_add_recv;
        const size_t n_mpi_destinations_to_add = n_mpi_destinations_to_add_recv + n_mpi_destinations_to_add_send_left + n_mpi_destinations_to_add_send_right;

        const size_t n_boundary_elements_to_add = n_wall_boundaries_to_add + n_symmetry_boundaries_to_add + n_inflow_boundaries_to_add + n_outflow_boundaries_to_add + n_mpi_destinations_to_add;

        std::vector<Element2D_t> new_elements(n_elements_new[global_rank] + elements_.size() - n_elements_ + n_boundary_elements_to_add - n_boundary_elements_to_delete);
        
        // Now we move carried over elements
        const size_t n_elements_move = n_elements_ - n_elements_send_left[global_rank] - n_elements_send_right[global_rank];
        SEM::Host::Meshes::move_elements(n_elements_move, n_elements_send_left[global_rank], n_elements_recv_left[global_rank], elements_.data(), new_elements.data(), faces_to_delete);
        SEM::Host::Meshes::move_boundary_elements(n_elements_, n_elements_new[global_rank], elements_.data(), new_elements.data(), boundary_elements_to_delete, faces_to_delete);

        // Boundaries
        // Boundaries to delete
        size_t n_wall_boundaries_to_delete = 0;
        size_t n_symmetry_boundaries_to_delete = 0;
        size_t n_inflow_boundaries_to_delete = 0;
        size_t n_outflow_boundaries_to_delete = 0;

        std::vector<size_t> new_wall_boundaries;
        std::vector<size_t> new_symmetry_boundaries;
        std::vector<size_t> new_inflow_boundaries;
        std::vector<size_t> new_outflow_boundaries;

        if (n_elements_send_left[global_rank] + n_elements_send_right[global_rank] > 0) {
            if (!wall_boundaries_.empty()) {
                std::vector<bool> wall_boundaries_to_delete(wall_boundaries_.size());
                SEM::Host::Meshes::find_boundaries_to_delete(wall_boundaries_, n_elements_, boundary_elements_to_delete, wall_boundaries_to_delete);
                
                for (int i = 0; i < wall_boundaries_to_delete.size(); ++i) {
                    n_wall_boundaries_to_delete += wall_boundaries_to_delete[i];
                }

                new_wall_boundaries = std::vector<size_t>(wall_boundaries_.size() + n_wall_boundaries_to_add - n_wall_boundaries_to_delete);

                SEM::Host::Meshes::move_required_boundaries(n_elements_, n_elements_new[global_rank], wall_boundaries_, new_wall_boundaries, wall_boundaries_to_delete, boundary_elements_to_delete);
            }
            else if (n_wall_boundaries_to_add > 0) {
                new_wall_boundaries = std::vector<size_t>(n_wall_boundaries_to_add);
            }

            if (!symmetry_boundaries_.empty()) {
                std::vector<bool> symmetry_boundaries_to_delete(symmetry_boundaries_.size());
                SEM::Host::Meshes::find_boundaries_to_delete(symmetry_boundaries_, n_elements_, boundary_elements_to_delete, symmetry_boundaries_to_delete);

                for (int i = 0; i < symmetry_boundaries_to_delete.size(); ++i) {
                    n_symmetry_boundaries_to_delete += symmetry_boundaries_to_delete[i];
                }

                new_symmetry_boundaries = std::vector<size_t>(symmetry_boundaries_.size() + n_symmetry_boundaries_to_add - n_symmetry_boundaries_to_delete);

                SEM::Host::Meshes::move_required_boundaries(n_elements_, n_elements_new[global_rank], symmetry_boundaries_, new_symmetry_boundaries, symmetry_boundaries_to_delete, boundary_elements_to_delete);
            }
            else if (n_symmetry_boundaries_to_add > 0) {
                new_symmetry_boundaries = std::vector<size_t>(n_symmetry_boundaries_to_add);
            }

            if (!inflow_boundaries_.empty()) {
                std::vector<bool> inflow_boundaries_to_delete(inflow_boundaries_.size());
                SEM::Host::Meshes::find_boundaries_to_delete(inflow_boundaries_, n_elements_, boundary_elements_to_delete, inflow_boundaries_to_delete);
                
                for (int i = 0; i < inflow_boundaries_to_delete.size(); ++i) {
                    n_inflow_boundaries_to_delete += inflow_boundaries_to_delete[i];
                }
                
                new_inflow_boundaries = std::vector<size_t>(inflow_boundaries_.size() + n_inflow_boundaries_to_add - n_inflow_boundaries_to_delete);

                SEM::Host::Meshes::move_required_boundaries(n_elements_, n_elements_new[global_rank], inflow_boundaries_, new_inflow_boundaries, inflow_boundaries_to_delete, boundary_elements_to_delete);
            }
            else if (n_inflow_boundaries_to_add > 0) {
                new_inflow_boundaries = std::vector<size_t>(n_inflow_boundaries_to_add);
            }

            if (!outflow_boundaries_.empty()) {
                std::vector<bool> outflow_boundaries_to_delete(outflow_boundaries_.size());
                SEM::Host::Meshes::find_boundaries_to_delete(outflow_boundaries_, n_elements_, boundary_elements_to_delete, outflow_boundaries_to_delete);

                for (int i = 0; i < outflow_boundaries_to_delete.size(); ++i) {
                    n_outflow_boundaries_to_delete += outflow_boundaries_to_delete[i];
                }

                new_outflow_boundaries = std::vector<size_t>(outflow_boundaries_.size() + n_outflow_boundaries_to_add - n_outflow_boundaries_to_delete);

                SEM::Host::Meshes::move_required_boundaries(n_elements_, n_elements_new[global_rank], outflow_boundaries_, new_outflow_boundaries, outflow_boundaries_to_delete, boundary_elements_to_delete);
            }
            else if (n_outflow_boundaries_to_add > 0) {
                new_outflow_boundaries = std::vector<size_t>(n_outflow_boundaries_to_add);
            } 
        }
        else {
            if (!wall_boundaries_.empty()) {
                new_wall_boundaries = std::vector<size_t>(wall_boundaries_.size() + n_wall_boundaries_to_add);

                SEM::Host::Meshes::move_all_boundaries(n_elements_, n_elements_new[global_rank], wall_boundaries_, new_wall_boundaries, boundary_elements_to_delete);
            }
            else if (n_wall_boundaries_to_add > 0) {
                new_wall_boundaries = std::vector<size_t>(n_wall_boundaries_to_add);
            }

            if (!symmetry_boundaries_.empty()) {
                new_symmetry_boundaries = std::vector<size_t>(symmetry_boundaries_.size() + n_symmetry_boundaries_to_add);

                SEM::Host::Meshes::move_all_boundaries(n_elements_, n_elements_new[global_rank], symmetry_boundaries_, new_symmetry_boundaries, boundary_elements_to_delete);
            }
            else if (n_symmetry_boundaries_to_add > 0) {
                new_symmetry_boundaries = std::vector<size_t>(n_symmetry_boundaries_to_add);
            }

            if (!inflow_boundaries_.empty()) {
                new_inflow_boundaries = std::vector<size_t>(inflow_boundaries_.size() + n_inflow_boundaries_to_add);

                SEM::Host::Meshes::move_all_boundaries(n_elements_, n_elements_new[global_rank], inflow_boundaries_, new_inflow_boundaries, boundary_elements_to_delete);
            }
            else if (n_inflow_boundaries_to_add > 0) {
                new_inflow_boundaries = std::vector<size_t>(n_inflow_boundaries_to_add);
            }

            if (!outflow_boundaries_.empty()) {
                new_outflow_boundaries = std::vector<size_t>(outflow_boundaries_.size() + n_outflow_boundaries_to_add);

                SEM::Host::Meshes::move_all_boundaries(n_elements_, n_elements_new[global_rank], outflow_boundaries_, new_outflow_boundaries, boundary_elements_to_delete);
            }
            else if (n_outflow_boundaries_to_add > 0) {
                new_outflow_boundaries = std::vector<size_t>(n_outflow_boundaries_to_add);
            }
        }

        // Interfaces
        // MPI incoming interfaces to delete
        size_t n_mpi_destinations_to_delete = 0;

        std::vector<size_t> new_mpi_interfaces_destination;
        std::vector<int> new_mpi_interfaces_destination_process;
        std::vector<size_t> new_mpi_interfaces_destination_local_index;
        std::vector<size_t> new_mpi_interfaces_destination_side;

        if (!mpi_interfaces_destination_.empty()) {
            std::vector<bool> mpi_destinations_to_delete(mpi_interfaces_destination_.size());
            SEM::Host::Meshes::find_boundaries_to_delete(mpi_interfaces_destination_, n_elements_, boundary_elements_to_delete, mpi_destinations_to_delete);

            for (auto i : mpi_destinations_to_delete) {
                n_mpi_destinations_to_delete += i;
            }

            new_mpi_interfaces_destination = std::vector<size_t>(mpi_interfaces_destination_.size() + n_mpi_destinations_to_add - n_mpi_destinations_to_delete);
            new_mpi_interfaces_destination_process = std::vector<int>(new_mpi_interfaces_destination.size());
            new_mpi_interfaces_destination_local_index = std::vector<size_t>(new_mpi_interfaces_destination.size());
            new_mpi_interfaces_destination_side = std::vector<size_t>(new_mpi_interfaces_destination.size());

            SEM::Host::Meshes::move_required_boundaries(n_elements_, n_elements_new[global_rank], mpi_interfaces_destination_, new_mpi_interfaces_destination, new_mpi_interfaces_destination_process, new_mpi_interfaces_destination_local_index, new_mpi_interfaces_destination_side, mpi_interfaces_new_process_incoming, mpi_interfaces_new_local_index_incoming, mpi_interfaces_new_side_incoming, mpi_destinations_to_delete, boundary_elements_to_delete);
        }
        else if (n_mpi_destinations_to_add > 0) {
            new_mpi_interfaces_destination = std::vector<size_t>(n_mpi_destinations_to_add);
            new_mpi_interfaces_destination_process = std::vector<int>(new_mpi_interfaces_destination.size());
            new_mpi_interfaces_destination_local_index = std::vector<size_t>(new_mpi_interfaces_destination.size());
            new_mpi_interfaces_destination_side = std::vector<size_t>(new_mpi_interfaces_destination.size());
        }

        // CHECK the same for self interfaces?
        // Self interfaces to add
        size_t n_self_interfaces_to_add = 0; // Wow

        // Self interfaces to delete
        size_t n_self_interfaces_to_delete = 0; // Wow 

        // Creating new boundary elements
        if (n_elements_send_left[global_rank] > 0) {
            SEM::Host::Meshes::create_sent_mpi_boundaries_destinations(n_elements_send_left[global_rank], 0, n_elements_new[global_rank] + elements_.size() - n_elements_ - n_boundary_elements_to_delete, mpi_interfaces_destination_.size() - n_mpi_destinations_to_delete, elements_.data(), new_elements.data(), nodes_.data(), elements_send_destinations_offset_left.data(), elements_send_destinations_keep_left.data(), destination_process_send_left.data(), destination_local_index_send_left.data(), new_mpi_interfaces_destination.data(), new_mpi_interfaces_destination_process.data(), new_mpi_interfaces_destination_local_index.data(), new_mpi_interfaces_destination_side.data(), polynomial_nodes, faces_to_delete);
        }

        if (n_elements_send_right[global_rank] > 0) {
            SEM::Host::Meshes::create_sent_mpi_boundaries_destinations(n_elements_send_right[global_rank], n_elements_ - n_elements_send_right[global_rank], n_elements_new[global_rank] + elements_.size() - n_elements_ - n_boundary_elements_to_delete + n_mpi_destinations_to_add_send_left, mpi_interfaces_destination_.size() - n_mpi_destinations_to_delete + n_mpi_destinations_to_add_send_left, elements_.data(), new_elements.data(), nodes_.data(), elements_send_destinations_offset_right.data(), elements_send_destinations_keep_right.data(), destination_process_send_right.data(), destination_local_index_send_right.data(), new_mpi_interfaces_destination.data(), new_mpi_interfaces_destination_process.data(), new_mpi_interfaces_destination_local_index.data(), new_mpi_interfaces_destination_side.data(), polynomial_nodes, faces_to_delete);
        }

        // We create received boundary elements for mpi destinations, and boundary conditions
        std::vector<size_t> neighbours_element_indices(neighbours_arrays_recv.size()); // We can build this above I think, as a side effect of computing the numbers to add
        std::vector<size_t> neighbours_element_side(neighbours_side_arrays_recv.size()); // We can build this above I think, as a side effect of computing the numbers to add
        if (n_elements_recv_left[global_rank] + n_elements_recv_right[global_rank] > 0) {
            const size_t new_elements_start_index = n_elements_new[global_rank] + elements_.size() - n_elements_ - n_boundary_elements_to_delete + n_mpi_destinations_to_add_send_left + n_mpi_destinations_to_add_send_right;
            const size_t wall_boundaries_start_index = wall_boundaries_.size() - n_wall_boundaries_to_delete;
            const size_t symmetry_boundaries_start_index = symmetry_boundaries_.size() - n_symmetry_boundaries_to_delete;
            const size_t inflow_boundaries_start_index = inflow_boundaries_.size() - n_inflow_boundaries_to_delete;
            const size_t outflow_boundaries_start_index = outflow_boundaries_.size() - n_outflow_boundaries_to_delete;
            const size_t mpi_destinations_start_index = mpi_interfaces_destination_.size() + n_mpi_destinations_to_add_send_left + n_mpi_destinations_to_add_send_right - n_mpi_destinations_to_delete;
            // What about faces?
            SEM::Host::Meshes::create_received_neighbours(
                neighbours_arrays_recv.size(), 
                global_rank, 
                mpi_interfaces_new_process_incoming.size(), 
                new_elements_start_index, 
                wall_boundaries_start_index, 
                symmetry_boundaries_start_index, 
                inflow_boundaries_start_index, 
                outflow_boundaries_start_index, 
                mpi_destinations_start_index, 
                n_wall_boundaries_to_add, 
                n_symmetry_boundaries_to_add, 
                n_inflow_boundaries_to_add, 
                n_outflow_boundaries_to_add, 
                n_mpi_destinations_to_add_recv, 
                n_elements_,
                n_elements_new[global_rank],
                neighbours_arrays_recv.data(), 
                neighbours_proc_arrays_recv.data(), 
                neighbours_side_arrays_recv.data(), 
                neighbours_N_arrays_recv.data(), 
                received_neighbour_node_indices.data(), 
                mpi_interfaces_new_local_index_incoming.data(), 
                mpi_interfaces_new_process_incoming.data(), 
                mpi_interfaces_new_side_incoming.data(), 
                new_elements.data(), 
                nodes_.data(), 
                mpi_interfaces_destination_.data(),
                neighbours_element_indices.data(), 
                neighbours_element_side.data(), 
                new_wall_boundaries.data(), 
                new_symmetry_boundaries.data(), 
                new_inflow_boundaries.data(), 
                new_outflow_boundaries.data(), 
                new_mpi_interfaces_destination.data(), 
                new_mpi_interfaces_destination_process.data(),
                new_mpi_interfaces_destination_local_index.data(),
                new_mpi_interfaces_destination_side.data(),
                polynomial_nodes,
                boundary_elements_to_delete);
        }

        if (n_elements_recv_left[global_rank] > 0) {
            // Now we must store these elements
            for (size_t i = 0; i < n_elements_recv_left[global_rank]; ++i) {
                new_elements[i] = std::move(elements_recv_left[i]);
            }

            SEM::Host::Meshes::fill_received_elements(n_elements_recv_left[global_rank], 0, new_elements.data(), maximum_N_, nodes_.data(), solution_arrays_recv_left.data(), received_node_indices.data(), polynomial_nodes);
        }                     

        if (n_elements_recv_right[global_rank] > 0) {
            // Now we must store these elements
            for (size_t i = 0; i < n_elements_recv_right[global_rank]; ++i) {
                new_elements[n_elements_new[global_rank] - n_elements_recv_right[global_rank] + i] = std::move(elements_recv_right[i]);
            }
            
            SEM::Host::Meshes::fill_received_elements(n_elements_recv_right[global_rank], n_elements_new[global_rank] - n_elements_recv_right[global_rank], new_elements.data(), maximum_N_, nodes_.data(), solution_arrays_recv_right.data(), received_node_indices.data() + 4 * n_elements_recv_left[global_rank], polynomial_nodes);
        }

        if (n_elements_recv_left[global_rank] > 0) {
            SEM::Host::Meshes::fill_received_elements_faces(
                n_elements_recv_left[global_rank], 
                0, 
                faces_.size() - n_faces_to_delete, 
                n_elements_new[global_rank], 
                n_elements_recv_left[global_rank], 
                n_elements_recv_right[global_rank], 
                new_elements.data(), 
                new_faces.data(), 
                nodes_.data(), 
                n_neighbours_arrays_recv_left.data(), 
                n_neighbours_arrays_recv_left.data(), 
                n_neighbours_arrays_recv_right.data(), 
                neighbour_offsets_left.data(), 
                neighbour_offsets_left.data(),
                neighbour_offsets_right.data(),
                neighbours_element_indices.data(), 
                neighbours_element_side.data(), 
                neighbours_proc_arrays_recv.data(),
                face_offsets_left.data(), 
                face_offsets_left.data(), 
                face_offsets_right.data());
        }

        if (n_elements_recv_right[global_rank] > 0) {
            SEM::Host::Meshes::fill_received_elements_faces(
                n_elements_recv_right[global_rank], 
                n_elements_new[global_rank] - n_elements_recv_right[global_rank], 
                faces_.size() - n_faces_to_delete, 
                n_elements_new[global_rank], 
                n_elements_recv_left[global_rank], 
                n_elements_recv_right[global_rank], 
                new_elements.data(), 
                new_faces.data(), 
                nodes_.data(), 
                n_neighbours_arrays_recv_right.data(), 
                n_neighbours_arrays_recv_left.data(), 
                n_neighbours_arrays_recv_right.data(), 
                neighbour_offsets_right.data(), 
                neighbour_offsets_left.data(),
                neighbour_offsets_right.data(),
                neighbours_element_indices.data(), 
                neighbours_element_side.data(), 
                neighbours_proc_arrays_recv.data(),
                face_offsets_right.data(), 
                face_offsets_left.data(), 
                face_offsets_right.data());
        }

        // MPI destinations faces
        // This is a bad way of doing this
        if (n_elements_recv_left[global_rank] + n_elements_recv_right[global_rank] > 0 && new_mpi_interfaces_destination.size() > 0) {
            SEM::Host::Meshes::add_new_faces_to_mpi_destination_elements(new_mpi_interfaces_destination.size(), faces_.size() - n_faces_to_delete, n_faces_to_add, new_mpi_interfaces_destination.data(), new_elements.data(), new_faces.data());
        }

        // Sizing mpi destinations
        std::vector<int> new_mpi_interfaces_process;
        std::vector<size_t> new_mpi_interfaces_incoming_size;

        for (size_t i = 0; i < new_mpi_interfaces_destination_process.size(); ++i) {
            bool missing = true;
            for (size_t j = 0; j < new_mpi_interfaces_process.size(); ++j) {
                if (new_mpi_interfaces_process[j] == new_mpi_interfaces_destination_process[i]) {
                    ++new_mpi_interfaces_incoming_size[j];
                    missing = false;
                    break;
                }
            }

            if (missing) {
                new_mpi_interfaces_process.push_back(new_mpi_interfaces_destination_process[i]);
                new_mpi_interfaces_incoming_size.push_back(1);
            }
        }

        // Sorting mpi destinations
        std::vector<size_t> new_mpi_interfaces_incoming_offset(new_mpi_interfaces_process.size());
        size_t mpi_interface_offset = 0;
        for (size_t i = 0; i < new_mpi_interfaces_incoming_offset.size(); ++i) {
            new_mpi_interfaces_incoming_offset[i] = mpi_interface_offset;
            mpi_interface_offset += new_mpi_interfaces_incoming_size[i];
        }
        if (mpi_interface_offset != new_mpi_interfaces_destination_process.size()) {
            std::cerr << "Error: Process " << global_rank << " had " << new_mpi_interfaces_destination_process.size() << " mpi interface elements, but now has " << mpi_interface_offset << ". Exiting." << std::endl;
            exit(67);
        }

        std::vector<size_t> new_mpi_interfaces_destination_ordered(new_mpi_interfaces_destination_process.size());
        std::vector<size_t> new_mpi_interfaces_destination_local_index_ordered(new_mpi_interfaces_destination_process.size());
        std::vector<size_t> new_mpi_interfaces_destination_side_ordered(new_mpi_interfaces_destination_process.size());
        std::vector<size_t> mpi_interfaces_destination_indices(new_mpi_interfaces_process.size(), 0);

        for (size_t i = 0; i < new_mpi_interfaces_destination_process.size(); ++i) {
            bool missing = true;
            for (size_t j = 0; j < new_mpi_interfaces_process.size(); ++j) {
                if (new_mpi_interfaces_process[j] == new_mpi_interfaces_destination_process[i]) {
                    new_mpi_interfaces_destination_ordered[new_mpi_interfaces_incoming_offset[j] + mpi_interfaces_destination_indices[j]] = new_mpi_interfaces_destination[i];
                    new_mpi_interfaces_destination_local_index_ordered[new_mpi_interfaces_incoming_offset[j] + mpi_interfaces_destination_indices[j]] = new_mpi_interfaces_destination_local_index[i];
                    new_mpi_interfaces_destination_side_ordered[new_mpi_interfaces_incoming_offset[j] + mpi_interfaces_destination_indices[j]] = new_mpi_interfaces_destination_side[i];
                    ++mpi_interfaces_destination_indices[j];

                    missing = false;
                    break;
                }
            }

            if (missing) {
                std::cerr << "Error: Process " << global_rank << " got sent new process " << new_mpi_interfaces_destination_process[i] << " for mpi interface element " << i << ", local id " << new_mpi_interfaces_destination[i] << ", but the process is not in this process' mpi interfaces destinations. Exiting." << std::endl;
                exit(68);
            }
        }

        new_mpi_interfaces_destination = std::move(new_mpi_interfaces_destination_ordered);

        // Rebuilding mpi origins
        std::vector<size_t> new_mpi_interfaces_outgoing_size(new_mpi_interfaces_process.size(), 0);

        std::vector<MPI_Request> mpi_incoming_requests(2 * mpi_load_balancing_incoming_n_transfers * new_mpi_interfaces_process.size());

        for (size_t i = 0; i < new_mpi_interfaces_process.size(); ++i) {
            MPI_Isend(&new_mpi_interfaces_incoming_size[i], 1, size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_incoming_offset * global_size * global_size + mpi_load_balancing_incoming_n_transfers * (global_size * global_rank + new_mpi_interfaces_process[i]), MPI_COMM_WORLD, &mpi_incoming_requests[mpi_load_balancing_incoming_n_transfers * (new_mpi_interfaces_process.size() + i)]);
            MPI_Irecv(&new_mpi_interfaces_outgoing_size[i], 1, size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_incoming_offset * global_size * global_size + mpi_load_balancing_incoming_n_transfers * (global_size * new_mpi_interfaces_process[i] + global_rank), MPI_COMM_WORLD, &mpi_incoming_requests[mpi_load_balancing_incoming_n_transfers * i]);
        }

        std::vector<MPI_Status> mpi_incoming_statuses(mpi_incoming_requests.size());
        MPI_Waitall(mpi_incoming_requests.size(), mpi_incoming_requests.data(), mpi_incoming_statuses.data());

        std::vector<size_t> new_mpi_interfaces_outgoing_offset(new_mpi_interfaces_process.size());
        size_t mpi_outgoing_interface_size_total = 0;
        for (size_t i = 0; i < new_mpi_interfaces_outgoing_offset.size(); ++i) {
            new_mpi_interfaces_outgoing_offset[i] = mpi_outgoing_interface_size_total;
            mpi_outgoing_interface_size_total += new_mpi_interfaces_outgoing_size[i];
        }

        std::vector<size_t> new_mpi_interfaces_origin(mpi_outgoing_interface_size_total);
        std::vector<size_t> new_mpi_interfaces_origin_side(new_mpi_interfaces_origin.size());

        std::vector<MPI_Request> mpi_origins_requests(2 * mpi_load_balancing_origins_n_transfers * new_mpi_interfaces_process.size());

        for (size_t i = 0; i < new_mpi_interfaces_process.size(); ++i) {
            MPI_Isend(new_mpi_interfaces_destination_local_index_ordered.data() + new_mpi_interfaces_incoming_offset[i], new_mpi_interfaces_incoming_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * global_rank + new_mpi_interfaces_process[i]), MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * (new_mpi_interfaces_process.size() + i)]);
            MPI_Irecv(new_mpi_interfaces_origin.data() + new_mpi_interfaces_outgoing_offset[i], new_mpi_interfaces_outgoing_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * new_mpi_interfaces_process[i] + global_rank), MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * i]);

            MPI_Isend(new_mpi_interfaces_destination_side_ordered.data() + new_mpi_interfaces_incoming_offset[i], new_mpi_interfaces_incoming_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * global_rank + new_mpi_interfaces_process[i]) + 1, MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * (new_mpi_interfaces_process.size() + i) + 1]);
            MPI_Irecv(new_mpi_interfaces_origin_side.data() + new_mpi_interfaces_outgoing_offset[i], new_mpi_interfaces_outgoing_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * new_mpi_interfaces_process[i] + global_rank) + 1, MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * i + 1]);
        }

        std::vector<MPI_Status> mpi_origins_statuses(mpi_origins_requests.size());
        MPI_Waitall(mpi_origins_requests.size(), mpi_origins_requests.data(), mpi_origins_statuses.data());        

        // Swapping out the old arrays for the new ones
        elements_ = std::move(new_elements);
        faces_ = std::move(new_faces);

        wall_boundaries_ = std::move(new_wall_boundaries);
        symmetry_boundaries_ = std::move(new_symmetry_boundaries);
        inflow_boundaries_ = std::move(new_inflow_boundaries);
        outflow_boundaries_ = std::move(new_outflow_boundaries);

        mpi_interfaces_destination_ = std::move(new_mpi_interfaces_destination);
        mpi_interfaces_origin_ = std::move(new_mpi_interfaces_origin);
        mpi_interfaces_origin_side_ = std::move(new_mpi_interfaces_origin_side);

        mpi_interfaces_outgoing_size_ = std::move(new_mpi_interfaces_outgoing_size);
        mpi_interfaces_incoming_size_ = std::move(new_mpi_interfaces_incoming_size);
        mpi_interfaces_outgoing_offset_ = std::move(new_mpi_interfaces_outgoing_offset);
        mpi_interfaces_incoming_offset_ = std::move(new_mpi_interfaces_incoming_offset);
        mpi_interfaces_process_ = std::move(new_mpi_interfaces_process);

        requests_ = std::vector<MPI_Request>(mpi_interfaces_process_.size() * 2 * mpi_boundaries_n_transfers);
        statuses_ = std::vector<MPI_Status>(requests_.size());
        requests_adaptivity_ = std::vector<MPI_Request>(mpi_interfaces_process_.size() * 2 * mpi_adaptivity_split_n_transfers);
        statuses_adaptivity_ = std::vector<MPI_Status>(requests_adaptivity_.size());

        // Updating quantities
        n_elements_ = n_elements_new[global_rank];

        // Output
        x_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
        y_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
        p_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
        u_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));
        v_output_ = std::vector<hostFloat>(n_elements_ * std::pow(n_interpolation_points_, 2));

        // Boundary solution exchange
        interfaces_p_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
        interfaces_u_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
        interfaces_v_ = std::vector<hostFloat>(mpi_interfaces_origin_.size() * (maximum_N_ + 1));
        interfaces_N_ = std::vector<int>(mpi_interfaces_origin_.size());
        interfaces_refine_ = std::make_unique<bool[]>(mpi_interfaces_origin_.size());
        interfaces_refine_without_splitting_ = std::make_unique<bool[]>(mpi_interfaces_origin_.size());

        receiving_interfaces_p_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
        receiving_interfaces_u_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
        receiving_interfaces_v_ = std::vector<hostFloat>(mpi_interfaces_destination_.size() * (maximum_N_ + 1));
        receiving_interfaces_N_ = std::vector<int>(mpi_interfaces_destination_.size());
        receiving_interfaces_refine_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
        receiving_interfaces_refine_without_splitting_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
        receiving_interfaces_creating_node_ = std::make_unique<bool[]>(mpi_interfaces_destination_.size());
    }
    else {
        // MPI interfaces
        std::vector<int> mpi_interfaces_new_process_outgoing(mpi_interfaces_origin_.size(), global_rank);

        std::vector<int> mpi_interfaces_new_process_incoming(mpi_interfaces_destination_.size());
        std::vector<size_t> mpi_interfaces_new_local_index_incoming(mpi_interfaces_destination_.size());
        std::vector<size_t> mpi_interfaces_new_side_incoming(mpi_interfaces_destination_.size());

        std::vector<MPI_Request> mpi_interfaces_requests(2 * mpi_load_balancing_interfaces_n_transfers * mpi_interfaces_process_.size());

        for (size_t i = 0; i < mpi_interfaces_process_.size(); ++i) {
            MPI_Isend(mpi_interfaces_new_process_outgoing.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]), MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * (mpi_interfaces_process_.size() + i)]);
            MPI_Irecv(mpi_interfaces_new_process_incoming.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], MPI_INT, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank), MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * i]);

            MPI_Isend(mpi_interfaces_origin_.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 1, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * (mpi_interfaces_process_.size() + i) + 1]);
            MPI_Irecv(mpi_interfaces_new_local_index_incoming.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 1, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * i + 1]);
        
            MPI_Isend(mpi_interfaces_origin_side_.data() + mpi_interfaces_outgoing_offset_[i], mpi_interfaces_outgoing_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 2, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * (mpi_interfaces_process_.size() + i) + 2]);
            MPI_Irecv(mpi_interfaces_new_side_incoming.data() + mpi_interfaces_incoming_offset_[i], mpi_interfaces_incoming_size_[i], size_t_data_type, mpi_interfaces_process_[i], mpi_load_balancing_interfaces_offset * global_size * global_size + mpi_load_balancing_interfaces_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 2, MPI_COMM_WORLD, &mpi_interfaces_requests[mpi_load_balancing_interfaces_n_transfers * i + 2]);
        }

        std::vector<MPI_Status> mpi_interfaces_statuses(mpi_interfaces_requests.size());
        MPI_Waitall(mpi_interfaces_requests.size(), mpi_interfaces_requests.data(), mpi_interfaces_statuses.data());

        std::vector<int> new_mpi_interfaces_process;
        std::vector<size_t> new_mpi_interfaces_incoming_size;

        // Rebuilding incoming interfaces
        for (size_t i = 0; i < mpi_interfaces_new_process_incoming.size(); ++i) {
            bool missing = true;
            for (size_t j = 0; j < new_mpi_interfaces_process.size(); ++j) {
                if (new_mpi_interfaces_process[j] == mpi_interfaces_new_process_incoming[i]) {
                    ++new_mpi_interfaces_incoming_size[j];
                    missing = false;
                    break;
                }
            }

            if (missing) {
                new_mpi_interfaces_process.push_back(mpi_interfaces_new_process_incoming[i]);
                new_mpi_interfaces_incoming_size.push_back(1);
            }
        }

        // Recalculate incoming offsets
        std::vector<size_t> new_mpi_interfaces_incoming_offset(new_mpi_interfaces_process.size());
        size_t mpi_interface_offset = 0;
        for (size_t i = 0; i < new_mpi_interfaces_incoming_offset.size(); ++i) {
            new_mpi_interfaces_incoming_offset[i] = mpi_interface_offset;
            mpi_interface_offset += new_mpi_interfaces_incoming_size[i];
        }
        const size_t n_mpi_interface_elements = mpi_interface_offset;
        if (n_mpi_interface_elements != mpi_interfaces_new_process_incoming.size()) {
            std::cerr << "Error: Process " << global_rank << " had " << mpi_interfaces_new_process_incoming.size() << " mpi interface elements, but now has " << n_mpi_interface_elements << ". Exiting." << std::endl;
            exit(56);
        }

        // Filling rebuilt incoming interfaces
        std::vector<size_t> new_mpi_interfaces_destination(n_mpi_interface_elements);
        std::vector<size_t> new_mpi_interfaces_destination_local_indices(n_mpi_interface_elements);
        std::vector<size_t> new_mpi_interfaces_destination_side(n_mpi_interface_elements);
        std::vector<size_t> mpi_interfaces_destination_indices(new_mpi_interfaces_process.size(), 0);

        for (size_t i = 0; i < mpi_interfaces_new_process_incoming.size(); ++i) {
            bool missing = true;
            for (size_t j = 0; j < new_mpi_interfaces_process.size(); ++j) {
                if (new_mpi_interfaces_process[j] == mpi_interfaces_new_process_incoming[i]) {
                    new_mpi_interfaces_destination[new_mpi_interfaces_incoming_offset[j] + mpi_interfaces_destination_indices[j]] = mpi_interfaces_destination_[i];
                    new_mpi_interfaces_destination_local_indices[new_mpi_interfaces_incoming_offset[j] + mpi_interfaces_destination_indices[j]] = mpi_interfaces_new_local_index_incoming[i];
                    new_mpi_interfaces_destination_side[new_mpi_interfaces_incoming_offset[j] + mpi_interfaces_destination_indices[j]] = mpi_interfaces_new_side_incoming[i];
                    ++mpi_interfaces_destination_indices[j];

                    missing = false;
                    break;
                }
            }

            if (missing) {
                std::cerr << "Error: Process " << global_rank << " got sent new process " << mpi_interfaces_new_process_incoming[i] << " for mpi interface element " << i << ", local id " << mpi_interfaces_destination_[i] << ", but the process is not in this process' mpi interfaces destinations. Exiting." << std::endl;
                exit(53);
            }
        }

        // Rebuilding mpi origins
        std::vector<size_t> new_mpi_interfaces_outgoing_size(new_mpi_interfaces_process.size(), 0);

        std::vector<MPI_Request> mpi_incoming_requests(2 * mpi_load_balancing_incoming_n_transfers * new_mpi_interfaces_process.size());

        for (size_t i = 0; i < new_mpi_interfaces_process.size(); ++i) {
            MPI_Isend(&new_mpi_interfaces_incoming_size[i], 1, size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_incoming_offset * global_size * global_size + mpi_load_balancing_incoming_n_transfers * (global_size * global_rank + new_mpi_interfaces_process[i]), MPI_COMM_WORLD, &mpi_incoming_requests[mpi_load_balancing_incoming_n_transfers * (new_mpi_interfaces_process.size() + i)]);
            MPI_Irecv(&new_mpi_interfaces_outgoing_size[i], 1, size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_incoming_offset * global_size * global_size + mpi_load_balancing_incoming_n_transfers * (global_size * new_mpi_interfaces_process[i] + global_rank), MPI_COMM_WORLD, &mpi_incoming_requests[mpi_load_balancing_incoming_n_transfers * i]);
        }

        std::vector<MPI_Status> mpi_incoming_statuses(mpi_incoming_requests.size());
        MPI_Waitall(mpi_incoming_requests.size(), mpi_incoming_requests.data(), mpi_incoming_statuses.data());

        std::vector<size_t> new_mpi_interfaces_outgoing_offset(new_mpi_interfaces_process.size());
        size_t mpi_outgoing_interface_size_total = 0;
        for (size_t i = 0; i < new_mpi_interfaces_outgoing_offset.size(); ++i) {
            new_mpi_interfaces_outgoing_offset[i] = mpi_outgoing_interface_size_total;
            mpi_outgoing_interface_size_total += new_mpi_interfaces_outgoing_size[i];
        }

        std::vector<size_t> new_mpi_interfaces_origin(mpi_outgoing_interface_size_total);
        std::vector<size_t> new_mpi_interfaces_origin_side(new_mpi_interfaces_origin.size());

        std::vector<MPI_Request> mpi_origins_requests(2 * mpi_load_balancing_origins_n_transfers * new_mpi_interfaces_process.size());

        for (size_t i = 0; i < new_mpi_interfaces_process.size(); ++i) {
            MPI_Isend(new_mpi_interfaces_destination_local_indices.data() + new_mpi_interfaces_incoming_offset[i], new_mpi_interfaces_incoming_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * global_rank + new_mpi_interfaces_process[i]), MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * (new_mpi_interfaces_process.size() + i)]);
            MPI_Irecv(new_mpi_interfaces_origin.data() + new_mpi_interfaces_outgoing_offset[i], new_mpi_interfaces_outgoing_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * new_mpi_interfaces_process[i] + global_rank), MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * i]);

            MPI_Isend(new_mpi_interfaces_destination_side.data() + new_mpi_interfaces_incoming_offset[i], new_mpi_interfaces_incoming_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * global_rank + new_mpi_interfaces_process[i]) + 1, MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * (new_mpi_interfaces_process.size() + i) + 1]);
            MPI_Irecv(new_mpi_interfaces_origin_side.data() + new_mpi_interfaces_outgoing_offset[i], new_mpi_interfaces_outgoing_size[i], size_t_data_type, new_mpi_interfaces_process[i], mpi_load_balancing_origins_offset * global_size * global_size + mpi_load_balancing_origins_n_transfers * (global_size * new_mpi_interfaces_process[i] + global_rank) + 1, MPI_COMM_WORLD, &mpi_origins_requests[mpi_load_balancing_origins_n_transfers * i + 1]);
        }

        std::vector<MPI_Status> mpi_origins_statuses(mpi_origins_requests.size());
        MPI_Waitall(mpi_origins_requests.size(), mpi_origins_requests.data(), mpi_origins_statuses.data());

        // Recalculate block sizes etc.
        interfaces_p_ = std::vector<hostFloat>(new_mpi_interfaces_origin.size() * (maximum_N_ + 1));
        interfaces_u_ = std::vector<hostFloat>(new_mpi_interfaces_origin.size() * (maximum_N_ + 1));
        interfaces_v_ = std::vector<hostFloat>(new_mpi_interfaces_origin.size() * (maximum_N_ + 1));
        interfaces_N_ = std::vector<int>(new_mpi_interfaces_origin.size());
        interfaces_refine_ = std::make_unique<bool[]>(new_mpi_interfaces_origin.size());
        interfaces_refine_without_splitting_ = std::make_unique<bool[]>(new_mpi_interfaces_origin.size());

        mpi_interfaces_destination_ = std::move(new_mpi_interfaces_destination);
        mpi_interfaces_origin_ = std::move(new_mpi_interfaces_origin);
        mpi_interfaces_origin_side_ = std::move(new_mpi_interfaces_origin_side);

        mpi_interfaces_process_ = std::move(new_mpi_interfaces_process);
        mpi_interfaces_incoming_size_ = std::move(new_mpi_interfaces_incoming_size);
        mpi_interfaces_incoming_offset_ = std::move(new_mpi_interfaces_incoming_offset);
        mpi_interfaces_outgoing_size_ = std::move(new_mpi_interfaces_outgoing_size);
        mpi_interfaces_outgoing_offset_ = std::move(new_mpi_interfaces_outgoing_offset);

        requests_ = std::vector<MPI_Request>(mpi_interfaces_process_.size() * 2 * mpi_boundaries_n_transfers);
        statuses_ = std::vector<MPI_Status>(requests_.size());
        requests_adaptivity_ = std::vector<MPI_Request>(mpi_interfaces_process_.size() * 2 * mpi_adaptivity_split_n_transfers);
        statuses_adaptivity_ = std::vector<MPI_Status>(requests_adaptivity_.size());
    }

    n_elements_global_ = n_elements_global_new;
    global_element_offset_ = global_element_offset_new[global_rank];

    // Adjust boundaries etc, maybe send new index and process to everyone?
}

auto SEM::Host::Meshes::Mesh2D_t::boundary_conditions(
        hostFloat t, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& weights, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    // Boundary conditions
    if (!wall_boundaries_.empty()) {
        SEM::Host::Meshes::compute_wall_boundaries(wall_boundaries_.size(), elements_.data(), wall_boundaries_.data(), faces_.data(), polynomial_nodes, weights, barycentric_weights);
    }
    if (!symmetry_boundaries_.empty()) {
        SEM::Host::Meshes::compute_symmetry_boundaries(symmetry_boundaries_.size(), elements_.data(), symmetry_boundaries_.data(), faces_.data(), polynomial_nodes, weights, barycentric_weights);
    }
    if (!inflow_boundaries_.empty()) {
        SEM::Host::Meshes::compute_inflow_boundaries(inflow_boundaries_.size(), elements_.data(), inflow_boundaries_.data(), faces_.data(), t, nodes_.data(), polynomial_nodes);
    }
    if (!outflow_boundaries_.empty()) {
        SEM::Host::Meshes::compute_outflow_boundaries(outflow_boundaries_.size(), elements_.data(), outflow_boundaries_.data(), faces_.data(), polynomial_nodes, weights, barycentric_weights);
    }

    // Interfaces
    if (!interfaces_origin_.empty()) {
        SEM::Host::Meshes::local_interfaces(interfaces_origin_.size(), elements_.data(), interfaces_origin_.data(), interfaces_origin_side_.data(), interfaces_destination_.data());
    }

    if (!mpi_interfaces_origin_.empty()) {
        int global_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
        int global_size;
        MPI_Comm_size(MPI_COMM_WORLD, &global_size);

        SEM::Host::Meshes::get_MPI_interfaces(mpi_interfaces_origin_.size(), elements_.data(), mpi_interfaces_origin_.data(), mpi_interfaces_origin_side_.data(), maximum_N_, interfaces_p_.data(), interfaces_u_.data(), interfaces_v_.data());
        
        const MPI_Datatype float_data_type = (sizeof(deviceFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
        for (size_t i = 0; i < mpi_interfaces_process_.size(); ++i) {
            MPI_Isend(interfaces_p_.data() + mpi_interfaces_outgoing_offset_[i] * (maximum_N_ + 1), mpi_interfaces_outgoing_size_[i] * (maximum_N_ + 1), float_data_type, mpi_interfaces_process_[i], mpi_boundaries_offset * global_size * global_size + mpi_boundaries_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]), MPI_COMM_WORLD, &requests_[mpi_boundaries_n_transfers * (mpi_interfaces_process_.size() + i)]);
            MPI_Irecv(receiving_interfaces_p_.data() + mpi_interfaces_incoming_offset_[i] * (maximum_N_ + 1), mpi_interfaces_incoming_size_[i] * (maximum_N_ + 1), float_data_type, mpi_interfaces_process_[i], mpi_boundaries_offset * global_size * global_size + mpi_boundaries_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank), MPI_COMM_WORLD, &requests_[mpi_boundaries_n_transfers * i]);

            MPI_Isend(interfaces_u_.data() + mpi_interfaces_outgoing_offset_[i] * (maximum_N_ + 1), mpi_interfaces_outgoing_size_[i] * (maximum_N_ + 1), float_data_type, mpi_interfaces_process_[i], mpi_boundaries_offset * global_size * global_size + mpi_boundaries_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 1, MPI_COMM_WORLD, &requests_[mpi_boundaries_n_transfers * (mpi_interfaces_process_.size() + i) + 1]);
            MPI_Irecv(receiving_interfaces_u_.data() + mpi_interfaces_incoming_offset_[i] * (maximum_N_ + 1), mpi_interfaces_incoming_size_[i] * (maximum_N_ + 1), float_data_type, mpi_interfaces_process_[i], mpi_boundaries_offset * global_size * global_size + mpi_boundaries_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 1, MPI_COMM_WORLD, &requests_[mpi_boundaries_n_transfers * i + 1]);

            MPI_Isend(interfaces_v_.data() + mpi_interfaces_outgoing_offset_[i] * (maximum_N_ + 1), mpi_interfaces_outgoing_size_[i] * (maximum_N_ + 1), float_data_type, mpi_interfaces_process_[i], mpi_boundaries_offset * global_size * global_size + mpi_boundaries_n_transfers * (global_size * global_rank + mpi_interfaces_process_[i]) + 2, MPI_COMM_WORLD, &requests_[mpi_boundaries_n_transfers * (mpi_interfaces_process_.size() + i) + 2]);
            MPI_Irecv(receiving_interfaces_v_.data() + mpi_interfaces_incoming_offset_[i] * (maximum_N_ + 1), mpi_interfaces_incoming_size_[i] * (maximum_N_ + 1), float_data_type, mpi_interfaces_process_[i], mpi_boundaries_offset * global_size * global_size + mpi_boundaries_n_transfers * (global_size * mpi_interfaces_process_[i] + global_rank) + 2, MPI_COMM_WORLD, &requests_[mpi_boundaries_n_transfers * i + 2]);
        }

        MPI_Waitall(mpi_boundaries_n_transfers * mpi_interfaces_process_.size(), requests_.data(), statuses_.data());

        SEM::Host::Meshes::put_MPI_interfaces(mpi_interfaces_destination_.size(), elements_.data(), mpi_interfaces_destination_.data(), maximum_N_, receiving_interfaces_p_.data(), receiving_interfaces_u_.data(), receiving_interfaces_v_.data());

        MPI_Waitall(mpi_boundaries_n_transfers * mpi_interfaces_process_.size(), requests_.data() + mpi_boundaries_n_transfers * mpi_interfaces_process_.size(), statuses_.data() + mpi_boundaries_n_transfers * mpi_interfaces_process_.size());
    }
}

// From cppreference.com
auto SEM::Host::Meshes::Mesh2D_t::almost_equal(hostFloat x, hostFloat y) -> bool {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<hostFloat>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<hostFloat>::min();
}

auto SEM::Host::Meshes::Mesh2D_t::interpolate_to_boundaries(
        const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, 
        const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) -> void {
    
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        elements_[element_index].interpolate_to_boundaries(lagrange_interpolant_left[elements_[element_index].N_], lagrange_interpolant_right[elements_[element_index].N_]);
    }
}

auto SEM::Host::Meshes::Mesh2D_t::project_to_faces(
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    for (size_t face_index = 0; face_index < faces_.size(); ++face_index) {
        Face2D_t& face = faces_[face_index];

        // Getting element solution
        const Element2D_t& element_L = elements_[face.elements_[0]];
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
            for (int i = 0; i <= face.N_; ++i) {
                const hostFloat coordinate = (polynomial_nodes[face.N_][i] + 1) * face.scale_[0] + 2 * face.offset_[0] - 1;

                hostFloat p_numerator = 0.0;
                hostFloat u_numerator = 0.0;
                hostFloat v_numerator = 0.0;
                hostFloat denominator = 0.0;

                for (int j = 0; j <= element_L.N_; ++j) {
                    if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element_L.N_][j])) {
                        p_numerator = element_L.p_extrapolated_[face.elements_side_[0]][j];
                        u_numerator = element_L.u_extrapolated_[face.elements_side_[0]][j];
                        v_numerator = element_L.v_extrapolated_[face.elements_side_[0]][j];
                        denominator = 1.0;
                        break;
                    }

                    const hostFloat t = barycentric_weights[element_L.N_][j]/(coordinate - polynomial_nodes[element_L.N_][j]);
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

        const Element2D_t& element_R = elements_[face.elements_[1]];
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
            for (int i = 0; i <= face.N_; ++i) {
                const hostFloat coordinate = (polynomial_nodes[face.N_][face.N_ - i] + 1) * face.scale_[1] + 2 * face.offset_[1] - 1;

                hostFloat p_numerator = 0.0;
                hostFloat u_numerator = 0.0;
                hostFloat v_numerator = 0.0;
                hostFloat denominator = 0.0;

                for (int j = 0; j <= element_R.N_; ++j) {
                    if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element_R.N_][j])) {
                        p_numerator = element_R.p_extrapolated_[face.elements_side_[1]][j];
                        u_numerator = element_R.u_extrapolated_[face.elements_side_[1]][j];
                        v_numerator = element_R.v_extrapolated_[face.elements_side_[1]][j];
                        denominator = 1.0;
                        break;
                    }

                    const hostFloat t = barycentric_weights[element_R.N_][j]/(coordinate - polynomial_nodes[element_R.N_][j]);
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

auto SEM::Host::Meshes::Mesh2D_t::project_to_elements(
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& weights, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        Element2D_t& element = elements_[element_index];

        for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
            // Conforming, forward
            if ((element.faces_[side_index].size() == 1)
                    && (faces_[element.faces_[side_index][0]].N_ == element.N_)  
                    && (faces_[element.faces_[side_index][0]].nodes_[0] == element.nodes_[side_index]) 
                    && (faces_[element.faces_[side_index][0]].nodes_[1] == element.nodes_[(side_index + 1) * !(side_index + 1 == (element.faces_.size()))])) {

                const Face2D_t& face = faces_[element.faces_[side_index][0]];
                for (int j = 0; j <= face.N_; ++j) {
                    element.p_flux_extrapolated_[side_index][j] = face.p_flux_[j] * element.scaling_factor_[side_index][j];
                    element.u_flux_extrapolated_[side_index][j] = face.u_flux_[j] * element.scaling_factor_[side_index][j];
                    element.v_flux_extrapolated_[side_index][j] = face.v_flux_[j] * element.scaling_factor_[side_index][j];
                }
            }
            // Conforming, backwards
            else if ((element.faces_[side_index].size() == 1)
                    && (faces_[element.faces_[side_index][0]].N_ == element.N_) 
                    && (faces_[element.faces_[side_index][0]].nodes_[1] == element.nodes_[side_index]) 
                    && (faces_[element.faces_[side_index][0]].nodes_[0] == element.nodes_[(side_index + 1) * !(side_index + 1 == (element.faces_.size()))])) {

                const Face2D_t& face = faces_[element.faces_[side_index][0]];
                for (int j = 0; j <= face.N_; ++j) {
                    element.p_flux_extrapolated_[side_index][face.N_ - j] = -face.p_flux_[j] * element.scaling_factor_[side_index][j];
                    element.u_flux_extrapolated_[side_index][face.N_ - j] = -face.u_flux_[j] * element.scaling_factor_[side_index][j];
                    element.v_flux_extrapolated_[side_index][face.N_ - j] = -face.v_flux_[j] * element.scaling_factor_[side_index][j];
                }
            }
            else { // We need to interpolate
                for (int i = 0; i <= element.N_; ++i) {
                    element.p_flux_extrapolated_[side_index][i] = 0.0;
                    element.u_flux_extrapolated_[side_index][i] = 0.0;
                    element.v_flux_extrapolated_[side_index][i] = 0.0;
                }

                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    const Face2D_t& face = faces_[element.faces_[side_index][face_index]];

                    // Non-conforming, forward
                    if (element_index == face.elements_[0]) {
                        for (int j = 0; j <= face.N_; ++j) {
                            const hostFloat coordinate = (polynomial_nodes[face.N_][j] + 1) * face.scale_[0] + 2 * face.offset_[0] - 1;
                            bool found_row = false;
                            
                            for (int i = 0; i <= element.N_; ++i) {
                                if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                                    element.p_flux_extrapolated_[side_index][i] += weights[face.N_][j]/weights[element.N_][i] * face.p_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[0];
                                    element.u_flux_extrapolated_[side_index][i] += weights[face.N_][j]/weights[element.N_][i] * face.u_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[0];
                                    element.v_flux_extrapolated_[side_index][i] += weights[face.N_][j]/weights[element.N_][i] * face.v_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[0];
                                    found_row = true;
                                    break;
                                }
                            }

                            if (!found_row) {
                                double s = 0.0;
                                for (int i = 0; i <= element.N_; ++i) {
                                    s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                                }
                                for (int i = 0; i <= element.N_; ++i) {
                                    const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                                    element.p_flux_extrapolated_[side_index][i] += T * weights[face.N_][j]/weights[element.N_][i] * face.p_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[0];
                                    element.u_flux_extrapolated_[side_index][i] += T * weights[face.N_][j]/weights[element.N_][i] * face.u_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[0];
                                    element.v_flux_extrapolated_[side_index][i] += T * weights[face.N_][j]/weights[element.N_][i] * face.v_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[0];
                                }
                            }
                        }
                    }
                    // Non-conforming, backwards
                    else {
                        for (int j = 0; j <= face.N_; ++j) {
                            const hostFloat coordinate = (polynomial_nodes[face.N_][face.N_ - j] + 1) * face.scale_[1] + 2 * face.offset_[1] - 1;
                            bool found_row = false;
                            
                            for (int i = 0; i <= element.N_; ++i) {
                                if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                                    element.p_flux_extrapolated_[side_index][i] += -weights[face.N_][j]/weights[element.N_][i] * face.p_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[1];
                                    element.u_flux_extrapolated_[side_index][i] += -weights[face.N_][j]/weights[element.N_][i] * face.u_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[1];
                                    element.v_flux_extrapolated_[side_index][i] += -weights[face.N_][j]/weights[element.N_][i] * face.v_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[1];
                                    found_row = true;
                                    break;
                                }
                            }

                            if (!found_row) {
                                double s = 0.0;
                                for (int i = 0; i <= element.N_; ++i) {
                                    s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                                }
                                for (int i = 0; i <= element.N_; ++i) {
                                    const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                                    element.p_flux_extrapolated_[side_index][i] += -T * weights[face.N_][j]/weights[element.N_][i] * face.p_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[1];
                                    element.u_flux_extrapolated_[side_index][i] += -T * weights[face.N_][j]/weights[element.N_][i] * face.u_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[1];
                                    element.v_flux_extrapolated_[side_index][i] += -T * weights[face.N_][j]/weights[element.N_][i] * face.v_flux_[j] * element.scaling_factor_[side_index][i] * face.scale_[1];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

auto SEM::Host::Meshes::Mesh2D_t::estimate_error(const std::vector<std::vector<hostFloat>>& polynomials) -> void {
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        elements_[element_index].estimate_p_error(tolerance_min_, tolerance_max_, polynomials[elements_[element_index].N_]);
    }
}

auto SEM::Host::Meshes::allocate_element_storage(size_t n_elements, std::vector<Element2D_t>& elements) -> void {
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        elements[element_index].allocate_storage();
    }
}

auto SEM::Host::Meshes::allocate_boundary_storage(size_t n_domain_elements, std::vector<Element2D_t>& elements) -> void {
    for (size_t element_index = n_domain_elements; element_index < elements.size(); ++element_index ) {
        elements[element_index].allocate_boundary_storage();
    }
}

auto SEM::Host::Meshes::compute_element_geometry(
        size_t n_elements, std::vector<Element2D_t>& elements, 
        const std::vector<Vec2<hostFloat>>& nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void {
    
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const std::array<Vec2<hostFloat>, 4> points {nodes[elements[element_index].nodes_[0]],
                                                       nodes[elements[element_index].nodes_[1]],
                                                       nodes[elements[element_index].nodes_[2]],
                                                       nodes[elements[element_index].nodes_[3]]};
        elements[element_index].compute_geometry(points, polynomial_nodes[elements[element_index].N_]);  
    }
}

auto SEM::Host::Meshes::compute_boundary_geometry(
        size_t n_domain_elements, 
        std::vector<Element2D_t>& elements, 
        const std::vector<Vec2<hostFloat>>& nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void {
    
    for (size_t element_index = n_domain_elements; element_index < elements.size(); ++element_index) {
        const std::array<Vec2<hostFloat>, 4> points {nodes[elements[element_index].nodes_[0]],
                                                       nodes[elements[element_index].nodes_[1]],
                                                       nodes[elements[element_index].nodes_[2]],
                                                       nodes[elements[element_index].nodes_[3]]};
        elements[element_index].compute_boundary_geometry(points, polynomial_nodes[elements[element_index].N_]);  
    }
}

auto SEM::Host::Meshes::compute_element_status(
        size_t n_elements, 
        std::vector<Element2D_t>& elements, 
        const std::vector<Vec2<hostFloat>>& nodes) -> void {
    
    constexpr std::array<hostFloat, 4> targets {-3*pi/4, -pi/4, pi/4, 3*pi/4};

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        long rotation = 0;
        for (size_t i = 0; i < elements[element_index].nodes_.size(); ++i) {
            const Vec2<hostFloat> delta = nodes[elements[element_index].nodes_[i]] - elements[element_index].center_;
            const hostFloat angle = std::atan2(delta.y(), delta.x());
            const hostFloat offset = std::atan2(std::sin(angle - targets[i]), std::cos(angle - targets[i]));
            const long n_turns = std::lround(offset * 2/pi + 4 * (offset < 0));
            rotation += n_turns * (n_turns < 4);
        }
        rotation /= elements[element_index].nodes_.size();

        elements[element_index].rotation_ = rotation;

        if (n_elements == 1) {
            elements[element_index].status_ = Hilbert::Status::H;
        }
        else if (element_index == 0) {
            const size_t next_element_index = element_index + 1;

            const Vec2<hostFloat> delta_next = elements[next_element_index].center_ - elements[element_index].center_;

            size_t outgoing_side = 0;
            if (std::abs(delta_next.y()) > std::abs(delta_next.x())) {
                if (delta_next.y() < 0) {
                    outgoing_side = 0;
                }
                else {
                    outgoing_side = 2;
                }
            }
            else {
                if (delta_next.x() < 0) {
                    outgoing_side = 3;
                }
                else {
                    outgoing_side = 1;
                }

            }

            elements[element_index].status_ = Hilbert::deduct_first_element_status(outgoing_side);
        }
        else if (element_index == n_elements - 1) {
            const size_t previous_element_index = element_index - 1;

            const Vec2<hostFloat> delta_previous = elements[previous_element_index].center_ - elements[element_index].center_;

            size_t incoming_side = 0;
            if (std::abs(delta_previous.y()) > std::abs(delta_previous.x())) {
                if (delta_previous.y() < 0) {
                    incoming_side = 0;
                }
                else {
                    incoming_side = 2;
                }
            }
            else {
                if (delta_previous.x() < 0) {
                    incoming_side = 3;
                }
                else {
                    incoming_side = 1;
                }

            }

            elements[element_index].status_ = Hilbert::deduct_last_element_status(incoming_side);
        }
        else {
            const size_t previous_element_index = element_index - 1;
            const size_t next_element_index = element_index + 1;

            const Vec2<hostFloat> delta_previous = elements[previous_element_index].center_ - elements[element_index].center_;
            const Vec2<hostFloat> delta_next = elements[next_element_index].center_ - elements[element_index].center_;

            size_t incoming_side = 0;
            if (std::abs(delta_previous.y()) > std::abs(delta_previous.x())) {
                if (delta_previous.y() < 0) {
                    incoming_side = 0;
                }
                else {
                    incoming_side = 2;
                }
            }
            else {
                if (delta_previous.x() < 0) {
                    incoming_side = 3;
                }
                else {
                    incoming_side = 1;
                }

            }

            size_t outgoing_side = 0;
            if (std::abs(delta_next.y()) > std::abs(delta_next.x())) {
                if (delta_next.y() < 0) {
                    outgoing_side = 0;
                }
                else {
                    outgoing_side = 2;
                }
            }
            else {
                if (delta_next.x() < 0) {
                    outgoing_side = 3;
                }
                else {
                    outgoing_side = 1;
                }

            }

            elements[element_index].status_ = Hilbert::deduct_element_status(incoming_side, outgoing_side);
        }
    }
}

auto SEM::Host::Meshes::allocate_face_storage(std::vector<Face2D_t>& faces) -> void {
    for (auto& face: faces) {
        face.allocate_storage();
    }
}

auto SEM::Host::Meshes::fill_element_faces(
        size_t n_elements, 
        std::vector<Element2D_t>& elements, 
        const std::vector<std::array<size_t, 4>>& element_to_face) -> void {
    
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        for (size_t j = 0; j < elements[element_index].faces_.size(); ++j) {
            elements[element_index].faces_[j][0] = element_to_face[element_index][j];
        }
    }
}

auto SEM::Host::Meshes::fill_boundary_element_faces(
        size_t n_domain_elements, 
        std::vector<Element2D_t>& elements, 
        const std::vector<std::array<size_t, 4>>& element_to_face) -> void {
    
    for (size_t element_index = n_domain_elements; element_index < elements.size(); ++element_index) {
        elements[element_index].faces_[0][0] = element_to_face[element_index][0];
    }
}

auto SEM::Host::Meshes::compute_face_geometry(
        std::vector<Face2D_t>& faces, 
        const std::vector<Element2D_t>& elements, 
        const std::vector<Vec2<hostFloat>>& nodes) -> void {
    
    for (auto& face: faces) {
        const std::array<Vec2<hostFloat>, 2> face_nodes {nodes[face.nodes_[0]], nodes[face.nodes_[1]]};

        const size_t element_L_index = face.elements_[0];
        const size_t element_R_index = face.elements_[1];
        const size_t element_L_side = face.elements_side_[0];
        const size_t element_R_side = face.elements_side_[1];
        const size_t element_L_next_side = (element_L_side + 1 < elements[element_L_index].nodes_.size()) ? element_L_side + 1 : size_t{0};
        const size_t element_R_next_side = (element_R_side + 1 < elements[element_R_index].nodes_.size()) ? element_R_side + 1 : size_t{0};
        const Element2D_t& element_L = elements[element_L_index];
        const Element2D_t& element_R = elements[element_R_index];
        const std::array<std::array<Vec2<hostFloat>, 2>, 2> elements_nodes {
            std::array<Vec2<hostFloat>, 2>{nodes[element_L.nodes_[element_L_side]], nodes[element_L.nodes_[element_L_next_side]]},
            std::array<Vec2<hostFloat>, 2>{nodes[element_R.nodes_[element_R_side]], nodes[element_R.nodes_[element_R_next_side]]}
        };
        const std::array<Vec2<hostFloat>, 2> elements_centres {
            element_L.center_, 
            element_R.center_
        };

        face.compute_geometry(elements_centres, face_nodes, elements_nodes);
    }
}

auto SEM::Host::Meshes::Mesh2D_t::get_solution(const std::vector<std::vector<hostFloat>>& interpolation_matrices) -> void {
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        const Element2D_t& element = elements_[element_index];
        const size_t offset_interp_2D = element_index * n_interpolation_points_ * n_interpolation_points_;

        const std::array<Vec2<hostFloat>, 4> points {nodes_[element.nodes_[0]],
                                                     nodes_[element.nodes_[1]],
                                                     nodes_[element.nodes_[2]],
                                                     nodes_[element.nodes_[3]]};

        element.interpolate_solution(n_interpolation_points_, offset_interp_2D, points, interpolation_matrices[element.N_], x_output_, y_output_, p_output_, u_output_, v_output_);
    }
}

auto SEM::Host::Meshes::Mesh2D_t::get_complete_solution(
        hostFloat time, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& interpolation_matrices, 
        std::vector<int>& N, 
        std::vector<hostFloat>& dp_dt, 
        std::vector<hostFloat>& du_dt, 
        std::vector<hostFloat>& dv_dt, 
        std::vector<hostFloat>& p_error, 
        std::vector<hostFloat>& u_error, 
        std::vector<hostFloat>& v_error, 
        std::vector<hostFloat>& p_sigma, 
        std::vector<hostFloat>& u_sigma, 
        std::vector<hostFloat>& v_sigma, 
        std::vector<int>& refine, 
        std::vector<int>& coarsen, 
        std::vector<int>& split_level, 
        std::vector<hostFloat>& p_analytical_error, 
        std::vector<hostFloat>& u_analytical_error, 
        std::vector<hostFloat>& v_analytical_error, 
        std::vector<int>& status, std::vector<int>& rotation) -> void {
    
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        const Element2D_t& element = elements_[element_index];
        const size_t offset_interp_2D = element_index * n_interpolation_points_ * n_interpolation_points_;

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
        status[element_index] = element.status_;
        rotation[element_index] = element.rotation_;
        const std::array<Vec2<hostFloat>, 4> points {nodes_[element.nodes_[0]],
                                                     nodes_[element.nodes_[1]],
                                                     nodes_[element.nodes_[2]],
                                                     nodes_[element.nodes_[3]]};

        element.interpolate_complete_solution(n_interpolation_points_, time, offset_interp_2D, points, polynomial_nodes[element.N_], interpolation_matrices[element.N_], x_output_, y_output_, p_output_, u_output_, v_output_, dp_dt, du_dt, dv_dt, p_analytical_error, u_analytical_error, v_analytical_error);
    }
}

auto SEM::Host::Meshes::compute_wall_boundaries(
        size_t n_wall_boundaries, 
        Element2D_t* elements, 
        const size_t* wall_boundaries, 
        const Face2D_t* faces, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& weights, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_wall_boundaries; ++boundary_index) {
        const size_t element_index = wall_boundaries[boundary_index];
        Element2D_t& element = elements[element_index];

        if (element.faces_[0].size() == 0) { // Only one neighbour
            const Face2D_t& face = faces[element.faces_[0][0]];
            const int face_side = face.elements_[0] == element_index;
            const Element2D_t& neighbour = elements[face.elements_[face_side]];
            const size_t neighbour_side = face.elements_side_[face_side];
            const Vec2<hostFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<hostFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};

            if (element.N_ == neighbour.N_) { // Conforming
                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<hostFloat> neighbour_velocity {neighbour.u_extrapolated_[neighbour_side][neighbour.N_ - k], neighbour.v_extrapolated_[neighbour_side][neighbour.N_ - k]};
                    const Vec2<hostFloat> local_velocity {-neighbour_velocity.dot(face.normal_), neighbour_velocity.dot(face.tangent_)};
                    //local_velocity.x() = (2 * neighbour.p_extrapolated_[neighbour_side][neighbour.N_ - k] + SEM::Host::Constants::c * local_velocity.x()) / SEM::Host::Constants::c;
                
                    element.p_extrapolated_[0][k] = neighbour.p_extrapolated_[neighbour_side][neighbour.N_ - k];
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
            else {
                for (int i = 0; i <= element.N_; ++i) {
                    element.p_extrapolated_[0][i] = 0.0;
                    element.u_extrapolated_[0][i] = 0.0;
                    element.v_extrapolated_[0][i] = 0.0;
                }

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const hostFloat coordinate = (polynomial_nodes[neighbour.N_][neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                            element.p_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }

                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<hostFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                    const Vec2<hostFloat> local_velocity {-neighbour_velocity.dot(face.normal_), neighbour_velocity.dot(face.tangent_)};
                    //local_velocity.x() = (2 * element.p_extrapolated_[0][k] + SEM::Host::Constants::c * local_velocity.x()) / SEM::Host::Constants::c;
                
                    //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
        }
        else {
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
                const Vec2<hostFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
                const Vec2<hostFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const hostFloat coordinate = (polynomial_nodes[neighbour.N_][neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                            element.p_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }

            const Face2D_t& face = faces[element.faces_[0][0]]; // CHECK this is kinda wrong, but we only use the normal and tangent so let's assume all the faces on a side have the same
            const Vec2<hostFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<hostFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};
            for (int k = 0; k <= element.N_; ++k) {
                const Vec2<hostFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                const Vec2<hostFloat> local_velocity {-neighbour_velocity.dot(face.normal_), neighbour_velocity.dot(face.tangent_)};
                //local_velocity.x() = (2 * element.p_extrapolated_[0][k] + SEM::Host::Constants::c * local_velocity.x()) / SEM::Host::Constants::c;
            
                //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
            }
        }
    }
}

auto SEM::Host::Meshes::compute_symmetry_boundaries(
        size_t n_symmetry_boundaries, 
        Element2D_t* elements, 
        const size_t* symmetry_boundaries, 
        const Face2D_t* faces, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& weights, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_symmetry_boundaries; ++boundary_index) {
        const size_t element_index = symmetry_boundaries[boundary_index];
        Element2D_t& element = elements[element_index];

        if (element.faces_[0].size() == 0) { // Only one neighbour
            const Face2D_t& face = faces[element.faces_[0][0]];
            const int face_side = face.elements_[0] == element_index;
            const Element2D_t& neighbour = elements[face.elements_[face_side]];
            const size_t neighbour_side = face.elements_side_[face_side];
            const Vec2<hostFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<hostFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};

            if (element.N_ == neighbour.N_) { // Conforming
                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<hostFloat> neighbour_velocity {neighbour.u_extrapolated_[neighbour_side][neighbour.N_ - k], neighbour.v_extrapolated_[neighbour_side][neighbour.N_ - k]};
                    const Vec2<hostFloat> local_velocity {-(neighbour_velocity.dot(face.normal_)), neighbour_velocity.dot(face.tangent_)};
                
                    element.p_extrapolated_[0][k] = neighbour.p_extrapolated_[neighbour_side][neighbour.N_ - k];
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
            else {
                for (int i = 0; i <= element.N_; ++i) {
                    element.p_extrapolated_[0][i] = 0.0;
                    element.u_extrapolated_[0][i] = 0.0;
                    element.v_extrapolated_[0][i] = 0.0;
                }

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const hostFloat coordinate = (polynomial_nodes[neighbour.N_][neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                            element.p_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }

                for (int k = 0; k <= element.N_; ++k) {
                    const Vec2<hostFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                    const Vec2<hostFloat> local_velocity {-(neighbour_velocity.dot(face.normal_)), neighbour_velocity.dot(face.tangent_)};
                
                    //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                    element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                    element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
                }
            }
        }
        else {
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
                const Vec2<hostFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
                const Vec2<hostFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const hostFloat coordinate = (polynomial_nodes[neighbour.N_][neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                            element.p_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }

            const Face2D_t& face = faces[element.faces_[0][0]]; // CHECK this is kinda wrong, but we only use the normal and tangent so let's assume all the faces on a side have the same
            const Vec2<hostFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
            const Vec2<hostFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};
            for (int k = 0; k <= element.N_; ++k) {
                const Vec2<hostFloat> neighbour_velocity {element.u_extrapolated_[0][k], element.v_extrapolated_[0][k]};
                const Vec2<hostFloat> local_velocity {-(neighbour_velocity.dot(face.normal_)), neighbour_velocity.dot(face.tangent_)};
            
                //element.p_extrapolated_[0][k] = element.p_extrapolated_[0][k]; // Does nothing
                element.u_extrapolated_[0][k] = normal_inv.dot(local_velocity);
                element.v_extrapolated_[0][k] = tangent_inv.dot(local_velocity);
            }
        }
    }
}

auto SEM::Host::Meshes::compute_inflow_boundaries(
        size_t n_inflow_boundaries, 
        Element2D_t* elements, 
        const size_t* inflow_boundaries, 
        const Face2D_t* faces, 
        hostFloat t, 
        const Vec2<hostFloat>* nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_inflow_boundaries; ++boundary_index) {
        Element2D_t& element = elements[inflow_boundaries[boundary_index]];
        const std::array<Vec2<hostFloat>, 2> points{nodes[element.nodes_[0]], nodes[element.nodes_[1]]};

        for (int k = 0; k <= element.N_; ++k) {
            const hostFloat interp = (polynomial_nodes[element.N_][k] + 1)/2;
            const Vec2<hostFloat> global_coordinates = points[1] * interp + points[0] * (1 - interp);

            const std::array<hostFloat, 3> state = SEM::Host::g(global_coordinates, t);

            element.p_extrapolated_[0][k] = state[0];
            element.u_extrapolated_[0][k] = state[1];
            element.v_extrapolated_[0][k] = state[2];
        }
    }
}

auto SEM::Host::Meshes::compute_outflow_boundaries(
        size_t n_outflow_boundaries, 
        Element2D_t* elements, 
        const size_t* outflow_boundaries, 
        const Face2D_t* faces, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& weights, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_outflow_boundaries; ++boundary_index) {
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
                for (int i = 0; i <= element.N_; ++i) {
                    element.p_extrapolated_[0][i] = 0.0;
                    element.u_extrapolated_[0][i] = 0.0;
                    element.v_extrapolated_[0][i] = 0.0;
                }

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const hostFloat coordinate = (polynomial_nodes[neighbour.N_][neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                            element.p_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }
        }
        else {
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
                const Vec2<hostFloat> normal_inv {face.normal_.x(), face.tangent_.x()};
                const Vec2<hostFloat> tangent_inv {face.normal_.y(), face.tangent_.y()};

                for (int j = 0; j <= neighbour.N_; ++j) {
                    const hostFloat coordinate = (polynomial_nodes[neighbour.N_][neighbour.N_ - j] + 1) * face.scale_[face_side] + 2 * face.offset_[face_side] - 1;
                    bool found_row = false;
                    
                    for (int i = 0; i <= element.N_; ++i) {
                        if (SEM::Host::Meshes::Mesh2D_t::almost_equal(coordinate, polynomial_nodes[element.N_][i])) {
                            element.p_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            found_row = true;
                            break;
                        }
                    }

                    if (!found_row) {
                        double s = 0.0;
                        for (int i = 0; i <= element.N_; ++i) {
                            s += barycentric_weights[element.N_][i]/(coordinate - polynomial_nodes[element.N_][i]);
                        }
                        for (int i = 0; i <= element.N_; ++i) {
                            const hostFloat T = barycentric_weights[element.N_][i]/((coordinate - polynomial_nodes[element.N_][i]) * s);

                            element.p_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.p_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.u_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.u_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                            element.v_extrapolated_[0][i] += T * weights[neighbour.N_][j]/weights[element.N_][i] * neighbour.v_extrapolated_[neighbour_side][j] * element.scaling_factor_[0][i];
                        }
                    }
                }
            }
        }
    }
}

auto SEM::Host::Meshes::local_interfaces(
        size_t n_local_interfaces, 
        Element2D_t* elements, 
        const size_t* local_interfaces_origin, 
        const size_t* local_interfaces_origin_side, 
        const size_t* local_interfaces_destination) -> void {
    
    for (size_t interface_index = 0; interface_index < n_local_interfaces; ++interface_index) {
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

auto SEM::Host::Meshes::get_MPI_interfaces(
        size_t n_MPI_interface_elements, 
        const Element2D_t* elements, 
        const size_t* MPI_interfaces_origin, 
        const size_t* MPI_interfaces_origin_side, 
        int maximum_N, 
        hostFloat* p, 
        hostFloat* u, 
        hostFloat* v) -> void {
    
    for (size_t interface_index = 0; interface_index < n_MPI_interface_elements; ++interface_index) {
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

auto SEM::Host::Meshes::get_MPI_interfaces_N(
        size_t n_MPI_interface_elements, 
        int N_max, 
        const Element2D_t* elements, 
        const size_t* MPI_interfaces_origin, 
        int* N) -> void {
    
    for (size_t interface_index = 0; interface_index < n_MPI_interface_elements; ++interface_index) {
        const size_t element_index = MPI_interfaces_origin[interface_index];
        
        N[interface_index] = elements[element_index].N_ + 2 * elements[element_index].would_p_refine(N_max);
    }
}

auto SEM::Host::Meshes::get_MPI_interfaces_adaptivity(
        size_t n_MPI_interface_elements, 
        size_t n_domain_elements, 
        size_t n_processes, 
        const Element2D_t* elements, 
        const Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        const size_t* MPI_interfaces_origin, 
        const size_t* MPI_interfaces_origin_side, 
        const size_t* MPI_interfaces_destination, 
        const int* MPI_process, 
        const size_t* origin_process_size, 
        const size_t* destination_process_size, 
        const size_t* destination_process_offset, 
        int* N, 
        bool* elements_splitting, 
        bool* elements_refining_without_splitting, 
        int max_split_level, 
        int N_max) -> void {

    for (size_t i = 0; i < n_MPI_interface_elements; ++i) {
        const size_t element_index = MPI_interfaces_origin[i];
        const Element2D_t& element = elements[element_index];

        N[i] = element.N_ + 2 * element.would_p_refine(N_max);
        elements_splitting[i] = element.would_h_refine(max_split_level);

        elements_refining_without_splitting[i] = false;
        if (elements_splitting[i]) {
            size_t process_offset = 0;
            size_t process_index = static_cast<size_t>(-1);
            int process = -1;
            for (size_t j = 0; j < n_processes; ++j) {
                process_offset += origin_process_size[j];
                if (process_offset > i) {
                    process = MPI_process[j];
                    process_index = j;
                    break;
                }
            }
            if (process == -1) {
                std::cout << "Error: MPI origin element " << i << ", index " << element_index << ", cannot be found with mpi_interfaces_outgoing_size_. Results are undefined." << std::endl;
            }

            const size_t side = MPI_interfaces_origin_side[i];
            const size_t next_side = (side + 1 < element.faces_.size()) ? side + 1 : size_t{0};
            const Vec2<hostFloat> new_node = (nodes[element.nodes_[side]] + nodes[element.nodes_[next_side]])/2;

            std::array<size_t, 2> n_side_faces {0, 0};
            const std::array<Vec2<hostFloat>, 2> AB {
                new_node - nodes[element.nodes_[side]],
                nodes[element.nodes_[next_side]] - new_node
            };

            const std::array<hostFloat, 2> AB_dot_inv {
                1/AB[0].dot(AB[0]),
                1/AB[1].dot(AB[1])
            };

            for (size_t j = 0; j < element.faces_[side].size(); ++j) {
                const Face2D_t& face = faces[element.faces_[side][j]];
                const size_t face_side = face.elements_[0] == element_index;
                const size_t other_element_index = face.elements_[face_side];
                if (other_element_index >= n_domain_elements) {
                    for (size_t k = destination_process_offset[process_index]; k < destination_process_offset[process_index] + destination_process_size[process_index]; ++k) {
                        if (MPI_interfaces_destination[k] == other_element_index) {
                            const std::array<Vec2<hostFloat>, 2> AC {
                                nodes[face.nodes_[0]] - nodes[element.nodes_[side]],
                                nodes[face.nodes_[0]] - new_node
                            };
                            const std::array<Vec2<hostFloat>, 2> AD {
                                nodes[face.nodes_[1]] - nodes[element.nodes_[side]],
                                nodes[face.nodes_[1]] - new_node
                            };

                            const std::array<hostFloat, 2> C_proj {
                                AC[0].dot(AB[0]) * AB_dot_inv[0],
                                AC[1].dot(AB[1]) * AB_dot_inv[1]
                            };
                            const std::array<hostFloat, 2> D_proj {
                                AD[0].dot(AB[0]) * AB_dot_inv[0],
                                AD[1].dot(AB[1]) * AB_dot_inv[1]
                            };

                            // CHECK this projection is different than the one used for splitting, maybe wrong
                            // The face is within the first element
                            if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                                && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                                || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                                && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                                ++n_side_faces[0];
                            }
                            // The face is within the second element
                            if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                                && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                                || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                                && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                                ++n_side_faces[1];
                            }

                            break;
                        }
                    }
                }
            }

            if (n_side_faces[0] == 0 || n_side_faces[1] == 0) {
                elements_refining_without_splitting[i] = true;
            }
        }
    }
}

auto SEM::Host::Meshes::put_MPI_interfaces(
        size_t n_MPI_interface_elements, 
        Element2D_t* elements, 
        const size_t* MPI_interfaces_destination, 
        int maximum_N, 
        const hostFloat* p, 
        const hostFloat* u, 
        const hostFloat* v) -> void {
    
    for (size_t interface_index = 0; interface_index < n_MPI_interface_elements; ++interface_index) {
        Element2D_t& destination_element = elements[MPI_interfaces_destination[interface_index]];
        const size_t boundary_offset = interface_index * (maximum_N + 1);

        for (int k = 0; k <= destination_element.N_; ++k) {
            destination_element.p_extrapolated_[0][k] = p[boundary_offset + k];
            destination_element.u_extrapolated_[0][k] = u[boundary_offset + k];
            destination_element.v_extrapolated_[0][k] = v[boundary_offset + k];
        }
    }
}

auto SEM::Host::Meshes::adjust_MPI_incoming_interfaces(
        size_t n_MPI_interface_elements, 
        size_t nodes_offset, 
        Element2D_t* elements, 
        const Face2D_t* faces, 
        const size_t* MPI_interfaces_destination, 
        const int* N, 
        Vec2<hostFloat>* nodes, 
        const bool* refine, 
        const bool* refine_without_splitting, 
        const bool* creating_node, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void {
    
    for (size_t i = 0; i < n_MPI_interface_elements; ++i) {
        Element2D_t& destination_element = elements[MPI_interfaces_destination[i]];

        if (refine[i] && refine_without_splitting[i]) {
            const Vec2<hostFloat> new_node = (nodes[destination_element.nodes_[0]] + nodes[destination_element.nodes_[1]])/2;
            size_t new_node_index = static_cast<size_t>(-1);

            if (creating_node[i]) {
                new_node_index = nodes_offset;
                for (size_t j = 0; j < i; ++j) {
                    new_node_index += creating_node[j];
                }
                nodes[new_node_index] = new_node;
            }
            else {
                for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                    const Face2D_t& face = faces[destination_element.faces_[0][face_index]];
                    if (nodes[face.nodes_[0]].almost_equal(new_node)) {
                        new_node_index = face.nodes_[0];
                        break;
                    }
                    if (nodes[face.nodes_[1]].almost_equal(new_node)) {
                        new_node_index = face.nodes_[1];
                        break;
                    }
                }
            }

            std::array<size_t, 2> n_side_faces {0, 0};
            const std::array<Vec2<hostFloat>, 2> AB {
                new_node - nodes[destination_element.nodes_[0]],
                nodes[destination_element.nodes_[1]] - new_node
            };

            const std::array<hostFloat, 2> AB_dot_inv {
                1/AB[0].dot(AB[0]),
                1/AB[1].dot(AB[1])
            };
            for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                const size_t face_index = destination_element.faces_[0][side_face_index];
                const Face2D_t& face = faces[face_index];
                
                const std::array<Vec2<hostFloat>, 2> AC {
                    nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                    nodes[face.nodes_[0]] - new_node
                };
                const std::array<Vec2<hostFloat>, 2> AD {
                    nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                    nodes[face.nodes_[1]] - new_node
                };

                const std::array<hostFloat, 2> C_proj {
                    AC[0].dot(AB[0]) * AB_dot_inv[0],
                    AC[1].dot(AB[1]) * AB_dot_inv[1]
                };
                const std::array<hostFloat, 2> D_proj {
                    AD[0].dot(AB[0]) * AB_dot_inv[0],
                    AD[1].dot(AB[1]) * AB_dot_inv[1]
                };

                // CHECK this projection is different than the one used for splitting, maybe wrong
                // The face is within the first element
                if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                    && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                    || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                    ++n_side_faces[0];
                }
                // The face is within the second element
                if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                    || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                    ++n_side_faces[1];
                }
            }

            ++destination_element.split_level_;
            if (n_side_faces[0] != 0) {
                destination_element.nodes_[1] = new_node_index;
                destination_element.nodes_[2] = new_node_index;
            }
            else {
                destination_element.nodes_[0] = new_node_index;
                destination_element.nodes_[3] = new_node_index;
            }

            const std::array<Vec2<hostFloat>, 4> points {nodes[destination_element.nodes_[0]],
                                                           nodes[destination_element.nodes_[1]],
                                                           nodes[destination_element.nodes_[2]],
                                                           nodes[destination_element.nodes_[3]]};

            destination_element.compute_boundary_geometry(points, polynomial_nodes[destination_element.N_]);

        }
        else if (destination_element.N_ != N[i]) {
            destination_element.resize_boundary_storage(N[i]);
            const std::array<Vec2<hostFloat>, 4> points {nodes[destination_element.nodes_[0]],
                                                           nodes[destination_element.nodes_[1]],
                                                           nodes[destination_element.nodes_[2]],
                                                           nodes[destination_element.nodes_[3]]};
            destination_element.compute_boundary_geometry(points, polynomial_nodes[destination_element.N_]);
        }
    }
}

auto SEM::Host::Meshes::p_adapt(
        size_t n_elements, 
        Element2D_t* elements, 
        int N_max, 
        const Vec2<hostFloat>* nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights) -> void {
    
    for (size_t i = 0; i < n_elements; ++i) {
        if (elements[i].would_p_refine(N_max)) {
            Element2D_t new_element(elements[i].N_ + 2, elements[i].split_level_, elements[i].status_, elements[i].rotation_, elements[i].faces_, elements[i].nodes_);

            new_element.interpolate_from(elements[i], polynomial_nodes, barycentric_weights);

            const std::array<Vec2<hostFloat>, 4> points {nodes[new_element.nodes_[0]],
                                                           nodes[new_element.nodes_[1]],
                                                           nodes[new_element.nodes_[2]],
                                                           nodes[new_element.nodes_[3]]};
            new_element.compute_geometry(points, polynomial_nodes[new_element.N_]); 

            elements[i] = std::move(new_element);
        }
    }
}

auto SEM::Host::Meshes::p_adapt_move(
        size_t n_elements, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        int N_max, 
        const Vec2<hostFloat>* nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights, 
        size_t* elements_new_indices) -> void {
    
    for (size_t i = 0; i < n_elements; ++i) {
        elements_new_indices[i] = i;

        if (elements[i].would_p_refine(N_max)) {
            new_elements[i] = Element2D_t(elements[i].N_ + 2, elements[i].split_level_, elements[i].status_,  elements[i].rotation_, elements[i].faces_, elements[i].nodes_);

            new_elements[i].interpolate_from(elements[i], polynomial_nodes, barycentric_weights);

            const std::array<Vec2<hostFloat>, 4> points {nodes[new_elements[i].nodes_[0]],
                                                           nodes[new_elements[i].nodes_[1]],
                                                           nodes[new_elements[i].nodes_[2]],
                                                           nodes[new_elements[i].nodes_[3]]};
            new_elements[i].compute_geometry(points, polynomial_nodes[new_elements[i].N_]); 
        }
        else {
            new_elements[i] = std::move(elements[i]);
        }
    }
}

auto SEM::Host::Meshes::p_adapt_split_faces(
        size_t n_elements, 
        size_t n_faces, 
        size_t n_nodes, 
        size_t n_splitting_elements, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const Face2D_t* faces, 
        int N_max, 
        const Vec2<hostFloat>* nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights, 
        size_t* elements_new_indices) -> void {
    
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        Element2D_t& element = elements[element_index];
        elements_new_indices[element_index] = element_index;

        if (elements[element_index].would_p_refine(N_max)) {
           new_elements[element_index] = Element2D_t(element.N_ + 2, element.split_level_, element.status_, element.rotation_, element.faces_, element.nodes_);

            // Adjusting faces for splitting elements
            for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                size_t side_n_splitting_faces = 0;
                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    side_n_splitting_faces += faces[element.faces_[side_index][face_index]].refine_;
                }

                if (side_n_splitting_faces > 0) {
                    std::vector<size_t> side_new_faces(element.faces_[side_index].size() + side_n_splitting_faces);

                    size_t side_new_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        if (faces[face_index].refine_) {
                            side_new_faces[side_new_face_index] = face_index;

                            size_t new_face_index = n_faces + 4 * n_splitting_elements;
                            for (size_t j = 0; j < face_index; ++j) {
                                new_face_index += faces[j].refine_;
                            }

                            side_new_faces[side_new_face_index + 1] = new_face_index;

                            side_new_face_index += 2;
                        }
                        else {
                            side_new_faces[side_new_face_index] = face_index;
                            ++side_new_face_index;
                        }
                    }

                    new_elements[element_index].faces_[side_index] = std::move(side_new_faces);
                }
            }

            new_elements[element_index].interpolate_from(element, polynomial_nodes, barycentric_weights);

            const std::array<Vec2<hostFloat>, 4> points {nodes[new_elements[element_index].nodes_[0]],
                                                           nodes[new_elements[element_index].nodes_[1]],
                                                           nodes[new_elements[element_index].nodes_[2]],
                                                           nodes[new_elements[element_index].nodes_[3]]};
            new_elements[element_index].compute_geometry(points, polynomial_nodes[new_elements[element_index].N_]); 
        }
        else {
            // Adjusting faces for splitting elements
            for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                size_t side_n_splitting_faces = 0;
                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    side_n_splitting_faces += faces[element.faces_[side_index][face_index]].refine_;
                }

                if (side_n_splitting_faces > 0) {
                    std::vector<size_t> side_new_faces(element.faces_[side_index].size() + side_n_splitting_faces);

                    size_t side_new_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        if (faces[face_index].refine_) {
                            side_new_faces[side_new_face_index] = face_index;

                            size_t new_face_index = n_faces + 4 * n_splitting_elements;
                            for (size_t j = 0; j < face_index; ++j) {
                                new_face_index += faces[j].refine_;
                            }

                            side_new_faces[side_new_face_index + 1] = new_face_index;

                            side_new_face_index += 2;
                        }
                        else {
                            side_new_faces[side_new_face_index] = face_index;
                            ++side_new_face_index;
                        }
                    }

                    element.faces_[side_index] = std::move(side_new_faces);
                }
            }

            new_elements[element_index] = std::move(element);
        }
    }
}

auto SEM::Host::Meshes::hp_adapt(
        size_t n_elements, 
        size_t n_faces, 
        size_t n_nodes, 
        size_t n_splitting_elements, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const Face2D_t* faces, 
        Face2D_t* new_faces, 
        int max_split_level, 
        int N_max, 
        Vec2<hostFloat>* nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<std::vector<hostFloat>>& barycentric_weights, 
        size_t* elements_new_indices) -> void {
    
    for (size_t i = 0; i < n_elements; ++i) {
        Element2D_t& element = elements[i];

        size_t n_splitting_elements_before = 0;
        for (size_t j = 0; j < i; ++j) {
            n_splitting_elements_before += elements[j].would_h_refine(max_split_level);
        }
        const size_t element_index = i + 3 * n_splitting_elements_before;

        elements_new_indices[i] = element_index;

        // h refinement
        if (element.would_h_refine(max_split_level)) {
            const size_t new_node_index = n_nodes + n_splitting_elements_before;

            std::array<size_t, 4> new_node_indices {static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)}; // CHECK this won't work with elements with more than 4 sides
            std::array<Vec2<hostFloat>, 4> new_nodes {}; // CHECK this won't work with elements with more than 4 sides
            nodes[new_node_index] = Vec2<hostFloat>{0};
            for (size_t side_index = 0; side_index < element.nodes_.size(); ++side_index) {
                nodes[new_node_index] += nodes[element.nodes_[side_index]];

                if (element.additional_nodes_[side_index]) {
                    const size_t face_index = element.faces_[side_index][0];
                    new_nodes[side_index] = (nodes[faces[face_index].nodes_[0]] + nodes[faces[face_index].nodes_[1]])/2;

                    new_node_indices[side_index] = n_nodes + n_splitting_elements;
                    for (size_t j = 0; j < face_index; ++j) {
                        new_node_indices[side_index] += faces[j].refine_;
                    }
                }
                else {
                    const std::array<Vec2<hostFloat>, 2> side_nodes = {nodes[element.nodes_[side_index]], (side_index + 1 < element.faces_.size()) ? nodes[element.nodes_[side_index + 1]] : nodes[element.nodes_[0]]};
                    const Vec2<hostFloat> new_node = (side_nodes[0] + side_nodes[1])/2;

                    for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                        const Face2D_t& face = faces[element.faces_[side_index][face_index]];
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

            const std::array<Vec2<hostFloat>, 4> element_nodes = {nodes[element.nodes_[0]], nodes[element.nodes_[1]], nodes[element.nodes_[2]], nodes[element.nodes_[3]]}; // CHECK this won't work with elements with more than 4 sides

            // We follow the Hilbert curve
            const std::array<size_t, 4> child_order = Hilbert::child_order(element.status_, element.rotation_);
            const std::array<Hilbert::Status, 4> child_statuses = Hilbert::child_statuses(element.status_, element.rotation_);
            const size_t new_face_index = n_faces + 4 * n_splitting_elements_before;

            for (size_t side_index = 0; side_index < element.nodes_.size(); ++side_index) {
                const size_t previous_side_index = (side_index > 0) ? side_index - 1 : element.nodes_.size() - 1;
                const size_t next_side_index = (side_index + 1 < element.nodes_.size()) ? side_index + 1 : size_t{0};
                const size_t opposite_side_index = (side_index + 2 < element.nodes_.size()) ? side_index + 2 : side_index + 2 - element.nodes_.size();

                const std::array<size_t, 2> side_element_indices {element_index + child_order[side_index], element_index + child_order[next_side_index]};
                const std::array<size_t, 2> side_element_sides {next_side_index, previous_side_index};
                
                new_faces[new_face_index + side_index] = Face2D_t{element.N_, {new_node_indices[side_index], new_node_index}, side_element_indices, side_element_sides};
            
                std::array<std::vector<size_t>, 4> new_element_faces {std::vector<size_t>(1), std::vector<size_t>(1), std::vector<size_t>(1), std::vector<size_t>(1)};
                
                new_element_faces[next_side_index][0] = new_face_index + side_index;
                new_element_faces[opposite_side_index][0] = new_face_index + previous_side_index;
                
                if (element.additional_nodes_[side_index]) {
                    const size_t face_index = element.faces_[side_index][0];
    
                    size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                    for (size_t j = 0; j < face_index; ++j) {
                        splitting_face_index += faces[j].refine_;
                    }
    
                    if (i == faces[face_index].elements_[0]) { // forward
                        new_element_faces[side_index][0] = face_index;
                    }
                    else { // backward
                        new_element_faces[side_index][0] = splitting_face_index;
                    }
                }
                else {
                    size_t n_side_faces = 0;
                    const Vec2<hostFloat> new_node = (nodes[element.nodes_[side_index]] + nodes[element.nodes_[next_side_index]])/2;
    
                    const Vec2<hostFloat> AB = new_node - nodes[element.nodes_[side_index]];
    
                    const hostFloat AB_dot_inv  = 1/AB.dot(AB);
    
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<hostFloat> AD = face_new_node - nodes[element.nodes_[side_index]];
                            const Vec2<hostFloat> AE = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const hostFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && E_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && E_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                        else {
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<hostFloat> AD = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                    }
    
                    new_element_faces[side_index] = std::vector<size_t>(n_side_faces);
    
                    size_t new_element_side_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                            for (size_t j = 0; j < face_index; ++j) {
                                splitting_face_index += faces[j].refine_;
                            }
                            const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<hostFloat> AD = face_new_node - nodes[element.nodes_[side_index]];
                            const Vec2<hostFloat> AE = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const hostFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                new_element_faces[side_index][new_element_side_face_index] = face_index;
                                ++new_element_side_face_index;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && E_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && E_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                new_element_faces[side_index][new_element_side_face_index] = splitting_face_index;
                                ++new_element_side_face_index;
                            }
                        }
                        else {
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - nodes[element.nodes_[side_index]];
                            const Vec2<hostFloat> AD = nodes[face.nodes_[1]] - nodes[element.nodes_[side_index]];
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                new_element_faces[side_index][new_element_side_face_index] = face_index;
                                ++new_element_side_face_index;
                            }
                        }
                    }
                }

                if (element.additional_nodes_[previous_side_index]) {
                    const size_t face_index = element.faces_[previous_side_index][0];
    
                    size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                    for (size_t j = 0; j < face_index; ++j) {
                        splitting_face_index += faces[j].refine_;
                    }
    
                    if (i == faces[face_index].elements_[0]) { // forward
                        new_element_faces[previous_side_index][0] = splitting_face_index;
                    }
                    else { // backward
                        new_element_faces[previous_side_index][0] = face_index;
                    }
                }
                else {
                    size_t n_side_faces = 0;
                    const Vec2<hostFloat> new_node = (nodes[element.nodes_[previous_side_index]] + nodes[element.nodes_[side_index]])/2;
    
                    const Vec2<hostFloat> AB = nodes[element.nodes_[side_index]] - new_node;
    
                    const hostFloat AB_dot_inv  = 1/AB.dot(AB);
    
                    for (size_t side_face_index = 0; side_face_index < element.faces_[previous_side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[previous_side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<hostFloat> AD = face_new_node - new_node;
                            const Vec2<hostFloat> AE = nodes[face.nodes_[1]] - new_node;
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const hostFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && E_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && E_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                        else {
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<hostFloat> AD = nodes[face.nodes_[1]] - new_node;
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                ++n_side_faces;
                            }
                        }
                    }
    
                    new_element_faces[previous_side_index] = std::vector<size_t>(n_side_faces);
    
                    size_t new_element_side_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[previous_side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[previous_side_index][side_face_index];
                        const Face2D_t& face = faces[face_index];
                        if (face.refine_) {
                            size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                            for (size_t j = 0; j < face_index; ++j) {
                                splitting_face_index += faces[j].refine_;
                            }
                            const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<hostFloat> AD = face_new_node - new_node;
                            const Vec2<hostFloat> AE = nodes[face.nodes_[1]] - new_node;
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
                            const hostFloat E_proj = AE.dot(AB) * AB_dot_inv;
    
                            // The first half of the face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                new_element_faces[previous_side_index][new_element_side_face_index] = face_index;
                                ++new_element_side_face_index;
                            }
                            // The second half of the face is within the element
                            if (D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && E_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && E_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
                                new_element_faces[previous_side_index][new_element_side_face_index] = splitting_face_index;
                                ++new_element_side_face_index;
                            }
                        }
                        else {
                            const Vec2<hostFloat> AC = nodes[face.nodes_[0]] - new_node;
                            const Vec2<hostFloat> AD = nodes[face.nodes_[1]] - new_node;
    
                            const hostFloat C_proj = AC.dot(AB) * AB_dot_inv;
                            const hostFloat D_proj = AD.dot(AB) * AB_dot_inv;
    
                            // The face is within the element
                            if (C_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && C_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                                && D_proj + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                                && D_proj <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
        
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

                new_elements[element_index + child_order[side_index]] = Element2D_t(element.N_, element.split_level_ + 1, child_statuses[side_index], element.rotation_, new_element_faces, new_element_node_indices);

                std::array<Vec2<hostFloat>, 4> new_element_nodes {};
                new_element_nodes[side_index] = nodes[element.nodes_[side_index]];
                new_element_nodes[next_side_index] = new_nodes[side_index];
                new_element_nodes[opposite_side_index] = nodes[new_node_index];
                new_element_nodes[previous_side_index] = new_nodes[previous_side_index];

                new_elements[element_index + child_order[side_index]].interpolate_from(new_element_nodes, element_nodes, element, polynomial_nodes, barycentric_weights);
                new_elements[element_index + child_order[side_index]].compute_geometry(new_element_nodes, polynomial_nodes[new_elements[element_index + child_order[side_index]].N_]);
            }

            for (size_t side_index = 0; side_index < element.nodes_.size(); ++side_index) {
                const size_t next_side_index = (side_index + 1 < element.nodes_.size()) ? side_index + 1 : size_t{0};
                const Vec2<hostFloat> new_node = (nodes[element.nodes_[side_index]] + nodes[element.nodes_[next_side_index]])/2;
                const std::array<Vec2<hostFloat>, 2> face_nodes {
                    new_node,
                    nodes[new_node_index]
                };
                const std::array<std::array<Vec2<hostFloat>, 2>, 2> elements_nodes {
                    std::array<Vec2<hostFloat>, 2>{new_node, nodes[new_node_index]}, // Not the same new node... this is confusing
                    std::array<Vec2<hostFloat>, 2>{nodes[new_node_index], new_node}
                };
                const std::array<Vec2<hostFloat>, 2> elements_centres {
                    new_elements[element_index + child_order[side_index]].center_,
                    new_elements[element_index + child_order[next_side_index]].center_
                };

                new_faces[new_face_index + side_index].compute_geometry(elements_centres, face_nodes, elements_nodes);
            }
        }
        // p refinement
        else if (element.would_p_refine(N_max)) {
            new_elements[element_index] = Element2D_t(element.N_ + 2, element.split_level_, element.status_, element.rotation_, element.faces_, element.nodes_);

            // Adjusting faces for splitting elements
            for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                size_t side_n_splitting_faces = 0;
                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    side_n_splitting_faces += faces[element.faces_[side_index][face_index]].refine_;
                }

                if (side_n_splitting_faces > 0) {
                    std::vector<size_t> side_new_faces(element.faces_[side_index].size() + side_n_splitting_faces);

                    size_t side_new_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        if (faces[face_index].refine_) {
                            side_new_faces[side_new_face_index] = face_index;

                            size_t new_face_index = n_faces + 4 * n_splitting_elements;
                            for (size_t j = 0; j < face_index; ++j) {
                                new_face_index += faces[j].refine_;
                            }

                            side_new_faces[side_new_face_index + 1] = new_face_index;

                            side_new_face_index += 2;
                        }
                        else {
                            side_new_faces[side_new_face_index] = face_index;
                            ++side_new_face_index;
                        }
                    }

                    new_elements[element_index].faces_[side_index] = std::move(side_new_faces);
                }
            }

            new_elements[element_index].interpolate_from(element, polynomial_nodes, barycentric_weights);

            const std::array<Vec2<hostFloat>, 4> points {nodes[new_elements[element_index].nodes_[0]],
                                                           nodes[new_elements[element_index].nodes_[1]],
                                                           nodes[new_elements[element_index].nodes_[2]],
                                                           nodes[new_elements[element_index].nodes_[3]]};
            new_elements[element_index].compute_geometry(points, polynomial_nodes[new_elements[element_index].N_]); 
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
                    std::vector<size_t> side_new_faces(element.faces_[side_index].size() + side_n_splitting_faces);

                    size_t side_new_face_index = 0;
                    for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                        const size_t face_index = element.faces_[side_index][side_face_index];
                        if (faces[face_index].refine_) {
                            side_new_faces[side_new_face_index] = face_index;

                            size_t new_face_index = n_faces + 4 * n_splitting_elements;
                            for (size_t j = 0; j < face_index; ++j) {
                                new_face_index += faces[j].refine_;
                            }

                            side_new_faces[side_new_face_index + 1] = new_face_index;

                            side_new_face_index += 2;
                        }
                        else {
                            side_new_faces[side_new_face_index] = face_index;
                            ++side_new_face_index;
                        }
                    }

                    element.faces_[side_index] = std::move(side_new_faces);
                }
            }

            new_elements[element_index] = std::move(element);
        }
    }
}

auto SEM::Host::Meshes::split_faces(
        size_t n_faces, 
        size_t n_nodes, 
        size_t n_splitting_elements, 
        Face2D_t* faces, 
        Face2D_t* new_faces, 
        const Element2D_t* elements, 
        Vec2<hostFloat>* nodes, 
        int max_split_level, 
        int N_max, 
        const size_t* elements_new_indices) -> void {
    
    for (size_t face_index = 0; face_index < n_faces; ++face_index) {
        Face2D_t& face = faces[face_index];
        const size_t element_L_index = face.elements_[0];
        const size_t element_R_index = face.elements_[1];
        const size_t element_L_side_index = face.elements_side_[0];
        const size_t element_R_side_index = face.elements_side_[1];
        const Element2D_t& element_L = elements[element_L_index];
        const Element2D_t& element_R = elements[element_R_index];
        const size_t element_L_new_index = elements_new_indices[element_L_index];
        const size_t element_R_new_index = elements_new_indices[element_R_index];

        if (face.refine_) {
            const size_t element_L_next_side_index = (element_L_side_index + 1 < element_L.nodes_.size()) ? element_L_side_index + 1 : size_t{0};
            const size_t element_R_next_side_index = (element_R_side_index + 1 < element_R.nodes_.size()) ? element_R_side_index + 1 : size_t{0};

            size_t new_node_index = n_nodes + n_splitting_elements;
            size_t new_face_index = n_faces + 4 * n_splitting_elements;
            for (size_t j = 0; j < face_index; ++j) {
                new_node_index += faces[j].refine_;
                new_face_index += faces[j].refine_;
            }

            const Vec2<hostFloat> new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
            nodes[new_node_index] = new_node;

            std::array<std::array<size_t, 2>, 2> new_element_indices {
                std::array<size_t, 2>{element_L_new_index, element_R_new_index},
                std::array<size_t, 2>{element_L_new_index, element_R_new_index}
            };

            std::array<std::array<Vec2<hostFloat>, 2>, 2> elements_centres {
                std::array<Vec2<hostFloat>, 2>{element_L.center_, element_R.center_},
                std::array<Vec2<hostFloat>, 2>{element_L.center_, element_R.center_}
            };

            std::array<std::array<std::array<Vec2<hostFloat>, 2>, 2>, 2> elements_nodes {
                std::array<std::array<Vec2<hostFloat>, 2>, 2>{
                    std::array<Vec2<hostFloat>, 2>{nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[element_L_next_side_index]]},
                    std::array<Vec2<hostFloat>, 2>{nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[element_R_next_side_index]]}
                },
                std::array<std::array<Vec2<hostFloat>, 2>, 2>{
                    std::array<Vec2<hostFloat>, 2>{nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[element_L_next_side_index]]},
                    std::array<Vec2<hostFloat>, 2>{nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[element_R_next_side_index]]}
                }
            };

            if (element_L.would_h_refine(max_split_level)) {
                const std::array<size_t, 4> child_order_L = Hilbert::child_order(element_L.status_, element_L.rotation_);

                const std::array<Vec2<hostFloat>, 2> element_nodes {nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[element_L_next_side_index]]};
                const Vec2<hostFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;

                const std::array<Vec2<hostFloat>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                const std::array<hostFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                const std::array<Vec2<hostFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                const std::array<Vec2<hostFloat>, 2> AD {new_node - element_nodes[0], new_node - new_element_node};
                const std::array<Vec2<hostFloat>, 2> AE {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                const std::array<hostFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<hostFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<hostFloat, 2> E_proj {AE[0].dot(AB[0]) * AB_dot_inv[0], AE[1].dot(AB[1]) * AB_dot_inv[1]};

                // The first face is within the first element
                if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t previous_side_index = (element_L_side_index > 0) ? element_L_side_index - 1 : element_L.nodes_.size() - 1;
                    const Vec2<deviceFloat> previous_new_element_node = (element_nodes[0] + nodes[element_L.nodes_[previous_side_index]])/2;
                    
                    new_element_indices[0][0] += child_order_L[element_L_side_index];
                    elements_centres[0][0] = (element_nodes[0] + element_L.center_ + new_element_node + previous_new_element_node)/4;
                    elements_nodes[0][0] = {element_nodes[0], new_element_node};
                }
                // The second face is within the first element
                if (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && E_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && E_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t previous_side_index = (element_L_side_index > 0) ? element_L_side_index - 1 : element_L.nodes_.size() - 1;
                    const Vec2<deviceFloat> previous_new_element_node = (element_nodes[0] + nodes[element_L.nodes_[previous_side_index]])/2;

                    new_element_indices[1][0] += child_order_L[element_L_side_index];
                    elements_centres[1][0] = (element_nodes[0] + element_L.center_ + new_element_node + previous_new_element_node)/4;
                    elements_nodes[1][0] = {element_nodes[0], new_element_node};
                }
                // The first face is within the second element
                if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t opposite_side_index = (element_L_side_index + 2 < element_L.nodes_.size()) ? element_L_side_index + 2 : element_L_side_index + 2 - element_L.nodes_.size();
                    const Vec2<deviceFloat> opposite_new_element_node = (element_nodes[1] + nodes[element_L.nodes_[opposite_side_index]])/2;

                    new_element_indices[0][0] += child_order_L[element_L_next_side_index];
                    elements_centres[0][0] = (element_nodes[1] + element_L.center_ + new_element_node + opposite_new_element_node)/4;
                    elements_nodes[0][0] = {new_element_node, element_nodes[1]};
                }
                // The second face is within the second element
                if (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && E_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && E_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t opposite_side_index = (element_L_side_index + 2 < element_L.nodes_.size()) ? element_L_side_index + 2 : element_L_side_index + 2 - element_L.nodes_.size();
                    const Vec2<deviceFloat> opposite_new_element_node = (element_nodes[1] + nodes[element_L.nodes_[opposite_side_index]])/2;

                    new_element_indices[1][0] += child_order_L[element_L_next_side_index];
                    elements_centres[1][0] = (element_nodes[1] + element_L.center_ + new_element_node + opposite_new_element_node)/4;
                    elements_nodes[1][0] = {new_element_node, element_nodes[1]};
                }
            }
            if (element_R.would_h_refine(max_split_level)) {
                const std::array<size_t, 4> child_order_R = Hilbert::child_order(element_R.status_, element_R.rotation_);

                const std::array<Vec2<hostFloat>, 2> element_nodes {nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[element_R_next_side_index]]};
                const Vec2<hostFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;

                const std::array<Vec2<hostFloat>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                const std::array<hostFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                const std::array<Vec2<hostFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                const std::array<Vec2<hostFloat>, 2> AD {new_node - element_nodes[0], new_node - new_element_node};
                const std::array<Vec2<hostFloat>, 2> AE {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                const std::array<hostFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<hostFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<hostFloat, 2> E_proj {AE[0].dot(AB[0]) * AB_dot_inv[0], AE[1].dot(AB[1]) * AB_dot_inv[1]};
                
                // The first face is within the first element
                if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t previous_side_index = (element_R_side_index > 0) ? element_R_side_index - 1 : element_R.nodes_.size() - 1;
                    const Vec2<deviceFloat> previous_new_element_node = (element_nodes[0] + nodes[element_R.nodes_[previous_side_index]])/2;
                    
                    new_element_indices[0][1] += child_order_R[element_R_side_index];
                    elements_centres[0][1] = (element_nodes[0] + element_R.center_ + new_element_node + previous_new_element_node)/4;
                    elements_nodes[0][1] = {element_nodes[0], new_element_node};
                }
                // The second face is within the first element
                if (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && E_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && E_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t previous_side_index = (element_R_side_index > 0) ? element_R_side_index - 1 : element_R.nodes_.size() - 1;
                    const Vec2<deviceFloat> previous_new_element_node = (element_nodes[0] + nodes[element_R.nodes_[previous_side_index]])/2;
                    
                    new_element_indices[1][1] += child_order_R[element_R_side_index];
                    elements_centres[1][1] = (element_nodes[0] + element_R.center_ + new_element_node + previous_new_element_node)/4;
                    elements_nodes[1][1] = {element_nodes[0], new_element_node};
                }
                // The first face is within the second element
                if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t opposite_side_index = (element_R_side_index + 2 < element_R.nodes_.size()) ? element_R_side_index + 2 : element_R_side_index + 2 - element_R.nodes_.size();
                    const Vec2<deviceFloat> opposite_new_element_node = (element_nodes[1] + nodes[element_R.nodes_[opposite_side_index]])/2;

                    new_element_indices[0][1] += child_order_R[element_R_next_side_index];
                    elements_centres[0][1] = (element_nodes[1] + element_R.center_ + new_element_node + opposite_new_element_node)/4;
                    elements_nodes[0][1] = {new_element_node, element_nodes[1]};
                }
                // The second face is within the second element
                if (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && E_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && E_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    const size_t opposite_side_index = (element_R_side_index + 2 < element_R.nodes_.size()) ? element_R_side_index + 2 : element_R_side_index + 2 - element_R.nodes_.size();
                    const Vec2<deviceFloat> opposite_new_element_node = (element_nodes[1] + nodes[element_R.nodes_[opposite_side_index]])/2;

                    new_element_indices[1][1] += child_order_R[element_R_next_side_index];
                    elements_centres[1][1] = (element_nodes[1] + element_R.center_ + new_element_node + opposite_new_element_node)/4;
                    elements_nodes[1][1] = {new_element_node, element_nodes[1]};
                }
            }

            const int face_N = std::max(element_L.N_ + 2 * element_L.would_p_refine(N_max), element_R.N_ + 2 * element_R.would_p_refine(N_max));
            
            new_faces[face_index] = Face2D_t(face_N, {face.nodes_[0], new_node_index}, new_element_indices[0], face.elements_side_);
            new_faces[new_face_index] = Face2D_t(face_N, {new_node_index, face.nodes_[1]}, new_element_indices[1], face.elements_side_);

            const std::array<std::array<Vec2<hostFloat>, 2>, 2> face_nodes {
                std::array<Vec2<hostFloat>, 2>{nodes[face.nodes_[0]], new_node},
                std::array<Vec2<hostFloat>, 2>{new_node, nodes[face.nodes_[1]]}
            };

            new_faces[face_index].compute_geometry(elements_centres[0], face_nodes[0], elements_nodes[0]);
            new_faces[new_face_index].compute_geometry(elements_centres[1], face_nodes[1], elements_nodes[1]);
        }
        else {
            const int face_N = std::max(element_L.N_ + 2 * element_L.would_p_refine(N_max), element_R.N_ + 2 * element_R.would_p_refine(N_max));

            if (face.N_ != face_N) {
                face.resize_storage(face_N);
            }

            std::array<size_t, 2> face_new_element_indices = {element_L_new_index, element_R_new_index};
            if (element_L.would_h_refine(max_split_level) || element_R.would_h_refine(max_split_level)) {
                const size_t element_L_next_side_index = (element_L_side_index + 1 < element_L.nodes_.size()) ? element_L_side_index + 1 : size_t{0};
                const size_t element_R_next_side_index = (element_R_side_index + 1 < element_R.nodes_.size()) ? element_R_side_index + 1 : size_t{0};

                std::array<Vec2<hostFloat>, 2> elements_centres {element_L.center_, element_R.center_};
                const std::array<Vec2<hostFloat>, 2> face_nodes {nodes[face.nodes_[0]], nodes[face.nodes_[1]]};
                std::array<std::array<Vec2<hostFloat>, 2>, 2> element_geometry_nodes {
                    std::array<Vec2<hostFloat>, 2>{nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[element_L_next_side_index]]},
                    std::array<Vec2<hostFloat>, 2>{nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[element_R_next_side_index]]}
                };

                if (element_L.would_h_refine(max_split_level)) {
                    const std::array<size_t, 4> child_order_L = Hilbert::child_order(element_L.status_, element_L.rotation_);

                    const std::array<Vec2<hostFloat>, 2> element_nodes {nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[element_L_next_side_index]]};
                    const Vec2<hostFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;

                    const std::array<Vec2<hostFloat>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                    const std::array<hostFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                    const std::array<Vec2<hostFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                    const std::array<Vec2<hostFloat>, 2> AD {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                    const std::array<hostFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                    const std::array<hostFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};

                    // The face is within the first element
                    if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                        && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                        face_new_element_indices[0] += child_order_L[element_L_side_index];
                        element_geometry_nodes[0] = {element_nodes[0], new_element_node};

                        const size_t previous_side_index = (element_L_side_index > 0) ? element_L_side_index - 1 : element_L.nodes_.size() - 1;
                        const Vec2<deviceFloat> previous_new_element_node = (element_nodes[0] + nodes[element_L.nodes_[previous_side_index]])/2;
                        elements_centres[0] = (element_nodes[0] + new_element_node + element_L.center_ + previous_new_element_node)/4; // CHECK only works on quadrilaterals
                    }
                    // The face is within the second element
                    if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                        && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                        face_new_element_indices[0] += child_order_L[element_L_next_side_index];
                        element_geometry_nodes[0] = {new_element_node, element_nodes[1]};

                        const size_t opposite_side_index = (element_L_side_index + 2 < element_L.nodes_.size()) ? element_L_side_index + 2 : element_L_side_index + 2 - element_L.nodes_.size();
                        const Vec2<deviceFloat> opposite_new_element_node = (element_nodes[1] + nodes[element_L.nodes_[opposite_side_index]])/2;
                        elements_centres[0] = (new_element_node + element_nodes[1] + element_L.center_ + opposite_new_element_node)/4; // CHECK only works on quadrilaterals
                    }
                }
                if (element_R.would_h_refine(max_split_level)) {
                    const std::array<size_t, 4> child_order_R = Hilbert::child_order(element_R.status_, element_R.rotation_);

                    const std::array<Vec2<hostFloat>, 2> element_nodes {nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[element_R_next_side_index]]};
                    const Vec2<hostFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;

                    const std::array<Vec2<hostFloat>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                    const std::array<hostFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                    const std::array<Vec2<hostFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                    const std::array<Vec2<hostFloat>, 2> AD {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                    const std::array<hostFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                    const std::array<hostFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};

                    // The face is within the first element
                    if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                        && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                        face_new_element_indices[1] += child_order_R[element_R_side_index];
                        element_geometry_nodes[1] = {element_nodes[0], new_element_node};
                        
                        const size_t previous_side_index = (element_R_side_index > 0) ? element_R_side_index - 1 : element_R.nodes_.size() - 1;
                        const Vec2<deviceFloat> previous_new_element_node = (element_nodes[0] + nodes[element_R.nodes_[previous_side_index]])/2;
                        elements_centres[1] = (element_nodes[0] + new_element_node + element_R.center_ + previous_new_element_node)/4; // CHECK only works on quadrilaterals
                    }
                    // The face is within the second element
                    if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                        && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                        && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                        face_new_element_indices[1] += child_order_R[element_R_next_side_index];
                        element_geometry_nodes[1] = {new_element_node, element_nodes[1]};
                        
                        const size_t opposite_side_index = (element_R_side_index + 2 < element_R.nodes_.size()) ? element_R_side_index + 2 : element_R_side_index + 2 - element_R.nodes_.size();
                        const Vec2<deviceFloat> opposite_new_element_node = (element_nodes[1] + nodes[element_R.nodes_[opposite_side_index]])/2;
                        elements_centres[1] = (new_element_node + element_nodes[1] + element_R.center_ + opposite_new_element_node)/4; // CHECK only works on quadrilaterals
                    }
                }

                face.compute_geometry(elements_centres, face_nodes, element_geometry_nodes);
            }

            face.elements_ = face_new_element_indices;

            new_faces[face_index] = std::move(face);
        }
    }
}

auto SEM::Host::Meshes::find_nodes(
        size_t n_elements, 
        Element2D_t* elements, 
        const Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        int max_split_level) -> void {
    
    for (size_t i = 0; i < n_elements; ++i) {
        Element2D_t& element = elements[i];
        element.additional_nodes_ = {false, false, false, false};
        if (element.would_h_refine(max_split_level)) {
            element.additional_nodes_ = {true, true, true, true};
            for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
                const std::array<Vec2<hostFloat>, 2> side_nodes = {nodes[element.nodes_[side_index]], (side_index + 1 < element.faces_.size()) ? nodes[element.nodes_[side_index + 1]] : nodes[element.nodes_[0]]};
                const Vec2<hostFloat> new_node = (side_nodes[0] + side_nodes[1])/2;

                // Here we check if the new node already exists
                for (size_t face_index = 0; face_index < element.faces_[side_index].size(); ++face_index) {
                    const Face2D_t& face = faces[element.faces_[side_index][face_index]];
                    if (nodes[face.nodes_[0]].almost_equal(new_node) || nodes[face.nodes_[1]].almost_equal(new_node)) {
                        element.additional_nodes_[side_index] = false;
                        break;
                    }
                }
            }
        }
    }
}

auto SEM::Host::Meshes::no_new_nodes(
        size_t n_elements, 
        Element2D_t* elements) -> void {
    
    for (size_t i = 0; i < n_elements; ++i) {
        elements[i].additional_nodes_ = {false, false, false, false};
    }
}

auto SEM::Host::Meshes::copy_boundaries_error(
        size_t n_boundaries, 
        Element2D_t* elements, 
        const size_t* boundaries, 
        const Face2D_t* faces) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_boundaries; ++boundary_index) {
        const size_t destination_element_index = boundaries[boundary_index];
        Element2D_t& destination_element = elements[destination_element_index];

        if (destination_element.faces_[0].size() == 1) { // Should always be the case 
            const Face2D_t& face = faces[destination_element.faces_[0][0]];
            const size_t face_side = face.elements_[0] == destination_element_index;
            const Element2D_t& source_element = elements[face.elements_[face_side]];
            const size_t element_side = face.elements_side_[face_side];

            destination_element.additional_nodes_[0] = source_element.additional_nodes_[element_side];
            destination_element.refine_ = source_element.refine_;
            destination_element.coarsen_ = source_element.coarsen_;
            destination_element.p_error_ = source_element.p_error_;
            destination_element.u_error_ = source_element.u_error_;
            destination_element.v_error_ = source_element.v_error_;
            destination_element.p_sigma_ = source_element.p_sigma_;
            destination_element.u_sigma_ = source_element.u_sigma_;
            destination_element.v_sigma_ = source_element.v_sigma_;
        }
    }
}

auto SEM::Host::Meshes::copy_interfaces_error(
        size_t n_local_interfaces, 
        Element2D_t* elements, 
        const size_t* local_interfaces_origin, 
        const size_t* local_interfaces_origin_side, 
        const size_t* local_interfaces_destination) -> void {
    
    for (size_t interface_index = 0; interface_index < n_local_interfaces; ++interface_index) {
        const Element2D_t& source_element = elements[local_interfaces_origin[interface_index]];
        Element2D_t& destination_element = elements[local_interfaces_destination[interface_index]];
        const size_t element_side = local_interfaces_origin_side[interface_index];

        destination_element.additional_nodes_[0] = source_element.additional_nodes_[element_side];
        destination_element.refine_ = source_element.refine_;
        destination_element.coarsen_ = source_element.coarsen_;
        destination_element.p_error_ = source_element.p_error_;
        destination_element.u_error_ = source_element.u_error_;
        destination_element.v_error_ = source_element.v_error_;
        destination_element.p_sigma_ = source_element.p_sigma_;
        destination_element.u_sigma_ = source_element.u_sigma_;
        destination_element.v_sigma_ = source_element.v_sigma_;
    }
}

auto SEM::Host::Meshes::copy_mpi_interfaces_error(
        size_t n_MPI_interface_elements, 
        Element2D_t* elements, 
        Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        const size_t* MPI_interfaces_destination, 
        const int* N, 
        const bool* elements_splitting, 
        bool* elements_refining_without_splitting, 
        bool* elements_creating_node) -> void {
    
    for (size_t i = 0; i < n_MPI_interface_elements; ++i) {
        Element2D_t& element = elements[MPI_interfaces_destination[i]];

        if (elements_splitting[i]) {            
            const std::array<Vec2<hostFloat>, 2> side_nodes = {nodes[element.nodes_[0]], nodes[element.nodes_[1]]};
            const Vec2<hostFloat> new_node = (side_nodes[0] + side_nodes[1])/2;

            // Here we check if the new node already exists
            element.additional_nodes_[0] = true;
            elements_creating_node[i] = true;
            for (auto face_index: element.faces_[0]) {
                const Face2D_t& face = faces[face_index];
                if (nodes[face.nodes_[0]].almost_equal(new_node) || nodes[face.nodes_[1]].almost_equal(new_node)) {
                    element.additional_nodes_[0] = false;
                    elements_creating_node[i] = false;
                    break;
                }
            }
            if (element.additional_nodes_[0]) {
                for (auto face_index: element.faces_[0]) {
                    const Face2D_t& face = faces[face_index];
                    if (((nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2).almost_equal(new_node)) {
                        elements_creating_node[i] = false;
                        break;
                    }
                }
            }

            std::array<size_t, 2> n_side_faces {0, 0};
            const std::array<Vec2<hostFloat>, 2> AB {
                new_node - side_nodes[0],
                side_nodes[1] - new_node
            };

            const std::array<hostFloat, 2> AB_dot_inv {
                1/AB[0].dot(AB[0]),
                1/AB[1].dot(AB[1])
            };
            for (size_t side_face_index = 0; side_face_index < element.faces_[0].size(); ++side_face_index) {
                const size_t face_index = element.faces_[0][side_face_index];
                const Face2D_t& face = faces[face_index];
                
                const std::array<Vec2<hostFloat>, 2> AC {
                    nodes[face.nodes_[0]] - side_nodes[0],
                    nodes[face.nodes_[0]] - new_node
                };
                const std::array<Vec2<hostFloat>, 2> AD {
                    nodes[face.nodes_[1]] - side_nodes[0],
                    nodes[face.nodes_[1]] - new_node
                };

                const std::array<hostFloat, 2> C_proj {
                    AC[0].dot(AB[0]) * AB_dot_inv[0],
                    AC[1].dot(AB[1]) * AB_dot_inv[1]
                };
                const std::array<hostFloat, 2> D_proj {
                    AD[0].dot(AB[0]) * AB_dot_inv[0],
                    AD[1].dot(AB[1]) * AB_dot_inv[1]
                };

                // CHECK this projection is different than the one used for splitting, maybe wrong
                // The face is within the first element
                if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                    && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                    || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                    ++n_side_faces[0];
                }
                // The face is within the second element
                if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                    || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                    ++n_side_faces[1];
                }
            }

            if (n_side_faces[0] == 0 || n_side_faces[1] == 0) {
                elements_refining_without_splitting[i] = true;
                element.refine_ = false;
                element.coarsen_ = false;

                // This is needed because faces won't compute their geometry again if an mpi element refines without splitting, as would_h_refine returns false
                // CHECK this is sketchy. Creates a race condition if two MPI interface elements link to the same face. 
                // Geometry fails if it is another ghost element.
                for (size_t side_face_index = 0; side_face_index < element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = element.faces_[0][side_face_index];
                    Face2D_t& face = faces[face_index];

                    const size_t other_element_face_side_index = face.elements_[0] == MPI_interfaces_destination[i];
                    const size_t other_element_index = face.elements_[other_element_face_side_index];
                    const Element2D_t& other_element = elements[other_element_index];
                    const size_t other_element_side_index = face.elements_side_[other_element_face_side_index];
                    const size_t other_element_next_side_index = (other_element_side_index + 1 < other_element.nodes_.size()) ? other_element_side_index + 1 : size_t{0};

                    Vec2<hostFloat> other_element_centre {0};
                    for (size_t element_side_index = 0; element_side_index < other_element.nodes_.size(); ++element_side_index) {
                        other_element_centre += nodes[other_element.nodes_[element_side_index]];
                    }
                    other_element_centre /= other_element.nodes_.size();

                    const std::array<Vec2<hostFloat>, 2> this_element_nodes = (n_side_faces[0] == 0) ? std::array<Vec2<hostFloat>, 2>{new_node, side_nodes[1]} : std::array<Vec2<hostFloat>, 2>{side_nodes[0], new_node};
                    const std::array<Vec2<hostFloat>, 2> other_element_nodes {nodes[other_element.nodes_[other_element_side_index]], nodes[other_element.nodes_[other_element_next_side_index]]};

                    const Vec2<hostFloat> this_element_centre = (this_element_nodes[0] + this_element_nodes[1])/2;
                    const std::array<Vec2<hostFloat>, 2> elements_centres = (other_element_face_side_index == 0) ? std::array<Vec2<hostFloat>, 2>{other_element_centre, this_element_centre} : std::array<Vec2<hostFloat>, 2>{this_element_centre, other_element_centre};
                    
                    std::array<std::array<Vec2<hostFloat>, 2>, 2> element_geometry_nodes = (other_element_face_side_index == 0) ? std::array<std::array<Vec2<hostFloat>, 2>, 2>{other_element_nodes, this_element_nodes} : std::array<std::array<Vec2<hostFloat>, 2>, 2>{this_element_nodes, other_element_nodes};

                    face.compute_geometry(elements_centres, {nodes[face.nodes_[0]], nodes[face.nodes_[1]]}, element_geometry_nodes);
                }
            }
            else  {
                elements_refining_without_splitting[i] = false;
                element.refine_ = true;
                element.coarsen_ = false;
                element.p_sigma_ = 0; // CHECK this is not relative to the cutoff, but it should stay above this
                element.u_sigma_ = 0;
                element.v_sigma_ = 0;
            }
        }
        else if (element.N_ < N[i]) {
            element.refine_ = true;
            element.coarsen_ = false;
            element.p_sigma_ = hostFloat{1000}; // CHECK this is not relative to the cutoff, but it should stay below this
            element.u_sigma_ = hostFloat{1000};
            element.v_sigma_ = hostFloat{1000};
            element.additional_nodes_[0] = false;
            elements_refining_without_splitting[i] = false;
            elements_creating_node[i] = false;
        }
        else {
            element.refine_ = false;
            element.coarsen_ = false;
            element.additional_nodes_[0] = false;
            elements_refining_without_splitting[i] = false;
            elements_creating_node[i] = false;
        }
    }
}

auto SEM::Host::Meshes::split_boundaries(
        size_t n_boundaries, 
        size_t n_faces, 
        size_t n_nodes, 
        size_t n_splitting_elements, 
        size_t offset, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const size_t* boundaries, 
        size_t* new_boundaries, 
        const Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        size_t* elements_new_indices) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_boundaries; ++boundary_index) {
        const size_t element_index = boundaries[boundary_index];
        Element2D_t& destination_element = elements[element_index];

        size_t new_boundary_index = boundary_index;
        for (size_t j = 0; j < boundary_index; ++j) {
            if (elements[boundaries[j]].faces_[0].size() == 1) { // Should always be the case 
                new_boundary_index += faces[elements[boundaries[j]].faces_[0][0]].refine_;
            }
        }
        const size_t new_element_index = offset + new_boundary_index;

        elements_new_indices[element_index] = new_element_index;

        if ((elements[element_index].faces_[0].size() == 1) && (faces[elements[element_index].faces_[0][0]].refine_)) {
            new_boundaries[new_boundary_index] = new_element_index;
            new_boundaries[new_boundary_index + 1] = new_element_index + 1;

            const size_t face_index = elements[element_index].faces_[0][0];
            const Face2D_t& face = faces[face_index];

            size_t new_node_index = n_nodes + n_splitting_elements;
            size_t new_face_index = n_faces + 4 * n_splitting_elements;
            for (size_t j = 0; j < face_index; ++j) {
                new_node_index += faces[j].refine_;
                new_face_index += faces[j].refine_;
            }

            const Vec2<hostFloat> new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;

            new_elements[new_element_index].N_ = destination_element.N_;
            new_elements[new_element_index].nodes_ = {destination_element.nodes_[0],
                                                      new_node_index,
                                                      new_node_index,
                                                      destination_element.nodes_[0]};
            new_elements[new_element_index].status_ = destination_element.status_;
            new_elements[new_element_index].rotation_ = destination_element.rotation_;
            new_elements[new_element_index].split_level_ = destination_element.split_level_ + 1;
            new_elements[new_element_index].refine_ = false;
            new_elements[new_element_index].coarsen_ = false;
            new_elements[new_element_index].additional_nodes_ = {false, false, false, false};
            new_elements[new_element_index].allocate_boundary_storage();
            new_elements[new_element_index].faces_[0][0] = new_face_index; // This should always be the case

            new_elements[new_element_index + 1].N_ = destination_element.N_;
            new_elements[new_element_index + 1].nodes_ = {new_node_index,
                                                          destination_element.nodes_[1],
                                                          destination_element.nodes_[1],
                                                          new_node_index};
            new_elements[new_element_index + 1].status_ = destination_element.status_;
            new_elements[new_element_index + 1].rotation_ = destination_element.rotation_;
            new_elements[new_element_index + 1].split_level_ = destination_element.split_level_ + 1;
            new_elements[new_element_index + 1].refine_ = false;
            new_elements[new_element_index + 1].coarsen_ = false;
            new_elements[new_element_index + 1].additional_nodes_ = {false, false, false, false};
            new_elements[new_element_index + 1].allocate_boundary_storage();
            new_elements[new_element_index + 1].faces_[0][0] = face_index; // This should always be the case

            const std::array<Vec2<hostFloat>, 4> points {nodes[new_elements[new_element_index].nodes_[0]],
                                                           new_node,
                                                           new_node,
                                                           nodes[new_elements[new_element_index].nodes_[3]]};
            new_elements[new_element_index].compute_boundary_geometry(points, polynomial_nodes[new_elements[new_element_index].N_]);

            const std::array<Vec2<hostFloat>, 4> points_2 {new_node,
                                                             nodes[new_elements[new_element_index + 1].nodes_[1]],
                                                             nodes[new_elements[new_element_index + 1].nodes_[2]],
                                                             new_node};
            new_elements[new_element_index + 1].compute_boundary_geometry(points_2, polynomial_nodes[new_elements[new_element_index + 1].N_]);
        }
        else {
            new_boundaries[new_boundary_index] = new_element_index;

            int N_element = destination_element.N_;
            for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                const Face2D_t& face = faces[destination_element.faces_[0][face_index]];
                const int neighbour_element_index = face.elements_[0] == element_index;
                const Element2D_t& source_element = elements[face.elements_[neighbour_element_index]];

                N_element = std::max(N_element, source_element.N_);
            }

            if (destination_element.N_ != N_element) {
                destination_element.resize_boundary_storage(N_element);
                const std::array<Vec2<hostFloat>, 4> points {nodes[destination_element.nodes_[0]],
                                                               nodes[destination_element.nodes_[1]],
                                                               nodes[destination_element.nodes_[2]],
                                                               nodes[destination_element.nodes_[3]]};
                destination_element.compute_boundary_geometry(points, polynomial_nodes[destination_element.N_]);
            }

            new_elements[new_element_index] = std::move(destination_element);
        }
    }
}

auto SEM::Host::Meshes::split_interfaces(
        size_t n_local_interfaces, 
        size_t n_faces, 
        size_t n_nodes, 
        size_t n_splitting_elements, 
        size_t offset, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const size_t* local_interfaces_origin, 
        const size_t* local_interfaces_origin_side, 
        const size_t* local_interfaces_destination, 
        size_t* new_local_interfaces_origin, 
        size_t* new_local_interfaces_origin_side, 
        size_t* new_local_interfaces_destination, 
        const Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        int max_split_level, 
        int N_max, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        size_t* elements_new_indices) -> void {
    
    for (size_t interface_index = 0; interface_index < n_local_interfaces; ++interface_index) {
        const size_t source_element_index = local_interfaces_origin[interface_index];
        const size_t destination_element_index = local_interfaces_destination[interface_index];
        const Element2D_t& source_element = elements[source_element_index];
        Element2D_t& destination_element = elements[destination_element_index];
        const size_t source_element_side = local_interfaces_origin_side[interface_index];

        size_t source_element_new_index = source_element_index;
        for (size_t j = 0; j < source_element_index; ++j) {
            source_element_new_index += 3 * elements[j].would_h_refine(max_split_level);
        }

        size_t new_interface_index = interface_index;
        for (size_t j = 0; j < interface_index; ++j) {
            new_interface_index += elements[local_interfaces_origin[j]].would_h_refine(max_split_level);
        }
        const size_t new_element_index = offset + new_interface_index;

        elements_new_indices[destination_element_index] = new_element_index;

        if (source_element.would_h_refine(max_split_level)) {
            const size_t source_element_next_side = (source_element_side + 1 < source_element.nodes_.size()) ? source_element_side + 1 : size_t{0};
            const std::array<size_t, 4> child_order = Hilbert::child_order(source_element.status_, source_element.rotation_);

            new_local_interfaces_origin[new_interface_index]     = source_element_new_index + child_order[source_element_side];
            new_local_interfaces_origin[new_interface_index + 1] = source_element_new_index + child_order[source_element_next_side];
            new_local_interfaces_origin_side[new_interface_index]     = source_element_side;
            new_local_interfaces_origin_side[new_interface_index + 1] = source_element_side;
            new_local_interfaces_destination[new_interface_index]     = new_element_index;
            new_local_interfaces_destination[new_interface_index + 1] = new_element_index + 1;

            Vec2<hostFloat> new_node {};
            size_t new_node_index = static_cast<size_t>(-1);

            if (destination_element.additional_nodes_[0]) {
                const size_t face_index = destination_element.faces_[0][0];
                new_node = (nodes[faces[face_index].nodes_[0]] + nodes[faces[face_index].nodes_[1]])/2;

                new_node_index = n_nodes + n_splitting_elements;
                for (size_t j = 0; j < face_index; ++j) {
                    new_node_index += faces[j].refine_;
                }
            }
            else {
                const std::array<Vec2<hostFloat>, 2> side_nodes = {nodes[destination_element.nodes_[0]], nodes[destination_element.nodes_[1]]};
                const Vec2<hostFloat> side_new_node = (side_nodes[0] + side_nodes[1])/2;

                for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                    const Face2D_t& face = faces[destination_element.faces_[0][face_index]];
                    if (nodes[face.nodes_[0]].almost_equal(side_new_node)) {
                        new_node_index = face.nodes_[0];
                        new_node = nodes[face.nodes_[0]];
                        break;
                    } 
                    else if (nodes[face.nodes_[1]].almost_equal(side_new_node)) {
                        new_node_index = face.nodes_[1];
                        new_node = nodes[face.nodes_[1]];
                        break;
                    }
                }
            }

            new_elements[new_element_index].N_     = source_element.N_;
            new_elements[new_element_index].nodes_ = {destination_element.nodes_[0],
                                                      new_node_index,
                                                      new_node_index,
                                                      destination_element.nodes_[0]};
            new_elements[new_element_index].status_ = destination_element.status_;
            new_elements[new_element_index].rotation_ = destination_element.rotation_;
            new_elements[new_element_index].split_level_ = destination_element.split_level_ + 1;
            new_elements[new_element_index].refine_ = false;
            new_elements[new_element_index].coarsen_ = false;
            new_elements[new_element_index].additional_nodes_ = {false, false, false, false};
            new_elements[new_element_index].allocate_boundary_storage();

            new_elements[new_element_index + 1].N_ = source_element.N_;
            new_elements[new_element_index + 1].nodes_ = {new_node_index,
                                                          destination_element.nodes_[1],
                                                          destination_element.nodes_[1],
                                                          new_node_index};
            new_elements[new_element_index + 1].status_ = destination_element.status_;
            new_elements[new_element_index + 1].rotation_ = destination_element.rotation_;
            new_elements[new_element_index + 1].split_level_ = destination_element.split_level_ + 1;
            new_elements[new_element_index + 1].refine_ = false;
            new_elements[new_element_index + 1].coarsen_ = false;
            new_elements[new_element_index + 1].additional_nodes_ = {false, false, false, false};
            new_elements[new_element_index + 1].allocate_boundary_storage();

            const std::array<Vec2<hostFloat>, 4> points {nodes[new_elements[new_element_index].nodes_[0]],
                                                           new_node,
                                                           new_node,
                                                           nodes[new_elements[new_element_index].nodes_[3]]};
            new_elements[new_element_index].compute_boundary_geometry(points, polynomial_nodes[new_elements[new_element_index].N_]);

            const std::array<Vec2<hostFloat>, 4> points_2 {new_node,
                                                             nodes[new_elements[new_element_index + 1].nodes_[1]],
                                                             nodes[new_elements[new_element_index + 1].nodes_[2]],
                                                             new_node};
            new_elements[new_element_index + 1].compute_boundary_geometry(points_2, polynomial_nodes[new_elements[new_element_index].N_]);

            if (destination_element.additional_nodes_[0]) {
                const size_t face_index = destination_element.faces_[0][0];

                size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                for (size_t j = 0; j < face_index; ++j) {
                    splitting_face_index += faces[j].refine_;
                }

                new_elements[new_element_index].faces_[0][0] = splitting_face_index; // Should always be the case
                new_elements[new_element_index + 1].faces_[0][0] = face_index; // Should always be the case
            }
            else {
                std::array<size_t, 2> n_side_faces {0, 0};

                const std::array<Vec2<hostFloat>, 2> AB {
                    new_node - nodes[destination_element.nodes_[0]],
                    nodes[destination_element.nodes_[1]] - new_node
                };

                const std::array<hostFloat, 2> AB_dot_inv {
                    1/AB[0].dot(AB[0]),
                    1/AB[1].dot(AB[1])
                };

                for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = destination_element.faces_[0][side_face_index];
                    const Face2D_t& face = faces[face_index];
                    if (face.refine_) {
                        const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            face_new_node - nodes[destination_element.nodes_[0]],
                            face_new_node - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AE {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> E_proj {
                            AE[0].dot(AB[0]) * AB_dot_inv[0],
                            AE[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The first half of the face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[0];
                        }
                        // The first half of the face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[1];
                        }
                        // The second half of the face is within the first element
                        if (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[0];
                        }
                        // The second half of the face is within the second element
                        if (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[1];
                        }
                    }
                    else {
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[0];
                        }
                        // The face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[1];
                        }
                    }
                }

                new_elements[new_element_index].faces_[0] = std::vector<size_t>(n_side_faces[0]);
                new_elements[new_element_index + 1].faces_[0] = std::vector<size_t>(n_side_faces[1]);

                std::array<size_t, 2> new_element_side_face_index {0, 0};
                for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = destination_element.faces_[0][side_face_index];
                    const Face2D_t& face = faces[face_index];
                    if (face.refine_) {
                        size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                        for (size_t j = 0; j < face_index; ++j) {
                            splitting_face_index += faces[j].refine_;
                        }
                        const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            face_new_node - nodes[destination_element.nodes_[0]],
                            face_new_node - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AE {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> E_proj {
                            AE[0].dot(AB[0]) * AB_dot_inv[0],
                            AE[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The first half of the face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index].faces_[0][new_element_side_face_index[0]] = face_index;
                            ++new_element_side_face_index[0];
                        }
                        // The first half of the face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index + 1].faces_[0][new_element_side_face_index[1]] = face_index;
                            ++new_element_side_face_index[1];
                        }
                        // The second half of the face is within the first element
                        if (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index].faces_[0][new_element_side_face_index[0]] = splitting_face_index;
                            ++new_element_side_face_index[0];
                        }
                        // The second half of the face is within the second element
                        if (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index + 1].faces_[0][new_element_side_face_index[1]] = splitting_face_index;
                            ++new_element_side_face_index[1];
                        }
                    }
                    else {
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index].faces_[0][new_element_side_face_index[0]] = face_index;
                            ++new_element_side_face_index[0];
                        }
                        // The face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index + 1].faces_[0][new_element_side_face_index[1]] = face_index;
                            ++new_element_side_face_index[1];
                        }
                    }
                }
            }
        }
        else {
            new_local_interfaces_origin[new_interface_index] = source_element_new_index;
            new_local_interfaces_origin_side[new_interface_index] = source_element_side;
            new_local_interfaces_destination[new_interface_index] = new_element_index;

            size_t side_n_splitting_faces = 0;
            for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                side_n_splitting_faces += faces[destination_element.faces_[0][face_index]].refine_;
            }

            if (side_n_splitting_faces > 0) {
                std::vector<size_t> side_new_faces(destination_element.faces_[0].size() + side_n_splitting_faces);

                size_t side_new_face_index = 0;
                for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = destination_element.faces_[0][side_face_index];
                    if (faces[face_index].refine_) {
                        side_new_faces[side_new_face_index] = face_index;

                        size_t new_face_index = n_faces + 4 * n_splitting_elements;
                        for (size_t j = 0; j < face_index; ++j) {
                            new_face_index += faces[j].refine_;
                        }

                        side_new_faces[side_new_face_index + 1] = new_face_index;

                        side_new_face_index += 2;
                    }
                    else {
                        side_new_faces[side_new_face_index] = face_index;
                        ++side_new_face_index;
                    }
                }

                destination_element.faces_[0] = std::move(side_new_faces);
            }

            if (source_element.would_p_refine(N_max)) {
                destination_element.resize_boundary_storage(source_element.N_ + 2);
                const std::array<Vec2<hostFloat>, 4> points {nodes[destination_element.nodes_[0]],
                                                               nodes[destination_element.nodes_[1]],
                                                               nodes[destination_element.nodes_[2]],
                                                               nodes[destination_element.nodes_[3]]};
                destination_element.compute_boundary_geometry(points, polynomial_nodes[destination_element.N_]);
            }

            new_elements[new_element_index] = std::move(destination_element);
        }
    }
}

auto SEM::Host::Meshes::split_mpi_outgoing_interfaces(
        size_t n_MPI_interface_elements, 
        size_t n_domain_elements, 
        size_t n_processes, 
        const Element2D_t* elements, 
        const Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        const size_t* mpi_interfaces_origin, 
        const size_t* mpi_interfaces_origin_side, 
        const size_t* MPI_interfaces_destination, 
        size_t* new_mpi_interfaces_origin, 
        size_t* new_mpi_interfaces_origin_side, 
        const int* MPI_process, 
        const size_t* origin_process_size, 
        const size_t* destination_process_size, 
        const size_t* destination_process_offset, 
        const bool* elements_splitting, 
        const bool* elements_refining_without_splitting, 
        int max_split_level) -> void {
    
    for (size_t i = 0; i < n_MPI_interface_elements; ++i) {
        const size_t origin_element_index = mpi_interfaces_origin[i];
        const Element2D_t& origin_element = elements[origin_element_index];

        size_t new_mpi_interface_index = i;
        for (size_t j = 0; j < i; ++j) {
            new_mpi_interface_index += elements_splitting[j] && !elements_refining_without_splitting[j];
        }

        new_mpi_interfaces_origin_side[new_mpi_interface_index] = mpi_interfaces_origin_side[i];

        size_t origin_element_new_index = origin_element_index;
        for (size_t j = 0; j < origin_element_index; ++j) {
            origin_element_new_index += 3 * elements[j].would_h_refine(max_split_level);
        }

        if (elements_refining_without_splitting[i] && elements_splitting[i]) {
            size_t process_offset = 0;
            size_t process_index = static_cast<size_t>(-1);
            int process = -1;
            for (size_t j = 0; j < n_processes; ++j) {
                process_offset += origin_process_size[j];
                if (process_offset > i) {
                    process = MPI_process[j];
                    process_index = j;
                    break;
                }
            }
            if (process == -1) {
                std::cout << "Error: MPI origin element " << i << ", index " << origin_element_index << ", cannot be found with mpi_interfaces_outgoing_size_. Results are undefined." << std::endl;
            }

            const size_t side = mpi_interfaces_origin_side[i];
            const size_t next_side = (side + 1 < origin_element.faces_.size()) ? side + 1 : size_t{0};
            const Vec2<hostFloat> new_node = (nodes[origin_element.nodes_[side]] + nodes[origin_element.nodes_[next_side]])/2;

            std::array<size_t, 2> n_side_faces {0, 0};
            const std::array<Vec2<hostFloat>, 2> AB {
                new_node - nodes[origin_element.nodes_[side]],
                nodes[origin_element.nodes_[next_side]] - new_node
            };

            const std::array<hostFloat, 2> AB_dot_inv {
                1/AB[0].dot(AB[0]),
                1/AB[1].dot(AB[1])
            };

            for (size_t j = 0; j < origin_element.faces_[side].size(); ++j) {
                const Face2D_t& face = faces[origin_element.faces_[side][j]];
                const size_t face_side = face.elements_[0] == origin_element_index;
                const size_t other_element_index = face.elements_[face_side];
                bool found_face = false;
                if (other_element_index >= n_domain_elements) {
                    for (size_t k = destination_process_offset[process_index]; k < destination_process_offset[process_index] + destination_process_size[process_index]; ++k) {
                        if (MPI_interfaces_destination[k] == other_element_index) {
                            const std::array<Vec2<hostFloat>, 2> AC {
                                nodes[face.nodes_[0]] - nodes[origin_element.nodes_[side]],
                                nodes[face.nodes_[0]] - new_node
                            };
                            const std::array<Vec2<hostFloat>, 2> AD {
                                nodes[face.nodes_[1]] - nodes[origin_element.nodes_[side]],
                                nodes[face.nodes_[1]] - new_node
                            };

                            const std::array<hostFloat, 2> C_proj {
                                AC[0].dot(AB[0]) * AB_dot_inv[0],
                                AC[1].dot(AB[1]) * AB_dot_inv[1]
                            };
                            const std::array<hostFloat, 2> D_proj {
                                AD[0].dot(AB[0]) * AB_dot_inv[0],
                                AD[1].dot(AB[1]) * AB_dot_inv[1]
                            };

                            // CHECK this projection is different than the one used for splitting, maybe wrong
                            // The face is within the first element
                            if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                                && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                                || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                                && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                                ++n_side_faces[0];
                                found_face = true;
                            }
                            // The face is within the second element
                            if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                                && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                                || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                                && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                                ++n_side_faces[1];
                                found_face = true;
                            }

                            break;
                        }
                    }
                    if (found_face) {
                        break;
                    }
                }
            }

            const std::array<size_t, 4> child_order = Hilbert::child_order(origin_element.status_, origin_element.rotation_);

            if (n_side_faces[0] > 0) {
                new_mpi_interfaces_origin[new_mpi_interface_index] = origin_element_new_index + child_order[side];
            }
            else {
                new_mpi_interfaces_origin[new_mpi_interface_index] = origin_element_new_index + child_order[next_side];
            }
        }
        else if (elements_splitting[i]) {
            const size_t next_side_index = (mpi_interfaces_origin_side[i] + 1 < origin_element.nodes_.size()) ? mpi_interfaces_origin_side[i] + 1 : size_t{0};
            const std::array<size_t, 4> child_order = Hilbert::child_order(origin_element.status_, origin_element.rotation_);

            new_mpi_interfaces_origin[new_mpi_interface_index]     = origin_element_new_index + child_order[mpi_interfaces_origin_side[i]];
            new_mpi_interfaces_origin[new_mpi_interface_index + 1] = origin_element_new_index + child_order[next_side_index];
            new_mpi_interfaces_origin_side[new_mpi_interface_index + 1] = mpi_interfaces_origin_side[i];
        }
        else {
            new_mpi_interfaces_origin[new_mpi_interface_index] = origin_element_new_index;
        }
    }
}

auto SEM::Host::Meshes::split_mpi_incoming_interfaces(
        size_t n_MPI_interface_elements, 
        size_t n_faces, 
        size_t n_nodes, 
        size_t n_splitting_elements, 
        size_t n_splitting_faces, 
        size_t offset, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const size_t* mpi_interfaces_destination, 
        size_t* new_mpi_interfaces_destination, 
        const Face2D_t* faces,
        Vec2<hostFloat>* nodes, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const int* N, 
        const bool* elements_splitting, 
        const bool* elements_refining_without_splitting, 
        const bool* elements_creating_node, 
        size_t* elements_new_indices) -> void {

    for (size_t i = 0; i < n_MPI_interface_elements; ++i) {
        const size_t destination_element_index = mpi_interfaces_destination[i];
        Element2D_t& destination_element = elements[destination_element_index];

        size_t new_mpi_interface_index = i;
        for (size_t j = 0; j < i; ++j) {
            new_mpi_interface_index += elements_splitting[j] && !elements_refining_without_splitting[j];
        }
        const size_t new_element_index = offset + new_mpi_interface_index;

        elements_new_indices[destination_element_index] = new_element_index;

        if (elements_refining_without_splitting[i] && elements_splitting[i]) {
            new_mpi_interfaces_destination[new_mpi_interface_index] = new_element_index;

            const Vec2<hostFloat> new_node = (nodes[destination_element.nodes_[0]] + nodes[destination_element.nodes_[1]]) /2;
            size_t new_node_index = static_cast<size_t>(-1);

            if (elements_creating_node[i]) {
                new_node_index = n_nodes + n_splitting_elements + n_splitting_faces;
                for (size_t j = 0; j < i; ++j) {
                    new_node_index += elements_creating_node[j];
                }
                nodes[new_node_index] = new_node;
            }
            else if (destination_element.additional_nodes_[0]) {
                size_t face_index = static_cast<size_t>(-1);
                for (size_t j = 0; j < destination_element.faces_[0].size(); ++j) {
                    const Face2D_t& face = faces[destination_element.faces_[0][j]];
                    if (((nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2).almost_equal(new_node)) {
                        face_index = destination_element.faces_[0][j];
                        break;
                    }
                }
                if (face_index == static_cast<size_t>(-1)) {
                    std::cout << "Error: Boundary element " << i << ", index " << mpi_interfaces_destination[i] << ", splits to single element and can't find its node. Results are undefined." << std::endl;
                }

                new_node_index = n_nodes + n_splitting_elements;
                for (size_t j = 0; j < face_index; ++j) {
                    new_node_index += faces[j].refine_;
                }
            }
            else {
                const std::array<Vec2<hostFloat>, 2> side_nodes = {nodes[destination_element.nodes_[0]], nodes[destination_element.nodes_[1]]};
                const Vec2<hostFloat> side_new_node = (side_nodes[0] + side_nodes[1])/2;

                for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                    const Face2D_t& face = faces[destination_element.faces_[0][face_index]];
                    if (nodes[face.nodes_[0]].almost_equal(side_new_node)) {
                        new_node_index = face.nodes_[0];
                        break;
                    } 
                    else if (nodes[face.nodes_[1]].almost_equal(side_new_node)) {
                        new_node_index = face.nodes_[1];
                        break;
                    }
                }
            }

            std::array<Vec2<hostFloat>, 4> points;
            std::array<size_t, 2> n_side_faces {0, 0};

            const std::array<Vec2<hostFloat>, 2> AB {
                new_node - nodes[destination_element.nodes_[0]],
                nodes[destination_element.nodes_[1]] - new_node
            };

            const std::array<hostFloat, 2> AB_dot_inv {
                1/AB[0].dot(AB[0]),
                1/AB[1].dot(AB[1])
            };

            for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                const size_t face_index = destination_element.faces_[0][side_face_index];
                const Face2D_t& face = faces[face_index];
                
                const std::array<Vec2<hostFloat>, 2> AC {
                    nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                    nodes[face.nodes_[0]] - new_node
                };
                const std::array<Vec2<hostFloat>, 2> AD {
                    nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                    nodes[face.nodes_[1]] - new_node
                };

                const std::array<hostFloat, 2> C_proj {
                    AC[0].dot(AB[0]) * AB_dot_inv[0],
                    AC[1].dot(AB[1]) * AB_dot_inv[1]
                };
                const std::array<hostFloat, 2> D_proj {
                    AD[0].dot(AB[0]) * AB_dot_inv[0],
                    AD[1].dot(AB[1]) * AB_dot_inv[1]
                };

                // CHECK this projection is different than the one used for splitting, maybe wrong
                // The face is within the first element
                if ((C_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1} 
                    && D_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                    || (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && C_proj[0] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                    ++n_side_faces[0];
                    break; // We can break here because we know only one side has faces, so this is it
                }
                // The face is within the second element
                if ((C_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && D_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())
                    || (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() <= hostFloat{1}
                    && C_proj[1] >= hostFloat{0} + std::numeric_limits<hostFloat>::epsilon())) {

                    ++n_side_faces[1];
                    break; // We can break here because we know only one side has faces, so this is it
                }
            }

            size_t side_n_splitting_faces = 0;
            for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                side_n_splitting_faces += faces[destination_element.faces_[0][face_index]].refine_;
            }

            if (side_n_splitting_faces > 0) {
                std::vector<size_t> side_new_faces(destination_element.faces_[0].size() + side_n_splitting_faces);

                size_t side_new_face_index = 0;
                for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = destination_element.faces_[0][side_face_index];
                    if (faces[face_index].refine_) {
                        side_new_faces[side_new_face_index] = face_index;

                        size_t new_face_index = n_faces + 4 * n_splitting_elements;
                        for (size_t j = 0; j < face_index; ++j) {
                            new_face_index += faces[j].refine_;
                        }

                        side_new_faces[side_new_face_index + 1] = new_face_index;

                        side_new_face_index += 2;
                    }
                    else {
                        side_new_faces[side_new_face_index] = face_index;
                        ++side_new_face_index;
                    }
                }

                destination_element.faces_[0] = std::move(side_new_faces);
            }

            if (n_side_faces[0] > 0) {
                destination_element.nodes_[1] = new_node_index;
                destination_element.nodes_[2] = new_node_index;

                points = {nodes[destination_element.nodes_[0]],
                          new_node,
                          new_node,
                          nodes[destination_element.nodes_[3]]};
            }
            else {
                destination_element.nodes_[0] = new_node_index;
                destination_element.nodes_[3] = new_node_index;

                points = {new_node,
                          nodes[destination_element.nodes_[1]],
                          nodes[destination_element.nodes_[2]],
                          new_node};
            }
            
            ++destination_element.split_level_;
            
            destination_element.compute_boundary_geometry(points, polynomial_nodes[destination_element.N_]);

            new_elements[new_element_index] = std::move(destination_element);
        }
        else if (elements_splitting[i]) {
            new_mpi_interfaces_destination[new_mpi_interface_index]     = new_element_index;
            new_mpi_interfaces_destination[new_mpi_interface_index + 1] = new_element_index + 1;

            const Vec2<hostFloat> new_node = (nodes[destination_element.nodes_[0]] + nodes[destination_element.nodes_[1]]) /2;
            size_t new_node_index = static_cast<size_t>(-1);

            if (elements_creating_node[i]) {
                new_node_index = n_nodes + n_splitting_elements + n_splitting_faces;
                for (size_t j = 0; j < i; ++j) {
                    new_node_index += elements_creating_node[j];
                }
                nodes[new_node_index] = new_node;
            }
            else if (destination_element.additional_nodes_[0]) {
                size_t face_index = static_cast<size_t>(-1);
                for (size_t j = 0; j < destination_element.faces_[0].size(); ++j) {
                    const Face2D_t& face = faces[destination_element.faces_[0][j]];
                    if (((nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2).almost_equal(new_node)) {
                        face_index = destination_element.faces_[0][j];
                        break;
                    }
                }
                if (face_index == static_cast<size_t>(-1)) {
                    std::cout << "Error: Boundary element " << i << ", index " << mpi_interfaces_destination[i] << ", splits to single element and can't find its node. Results are undefined." << std::endl;
                }

                new_node_index = n_nodes + n_splitting_elements;
                for (size_t j = 0; j < face_index; ++j) {
                    new_node_index += faces[j].refine_;
                }
            }
            else {
                const std::array<Vec2<hostFloat>, 2> side_nodes = {nodes[destination_element.nodes_[0]], nodes[destination_element.nodes_[1]]};
                const Vec2<hostFloat> side_new_node = (side_nodes[0] + side_nodes[1])/2;

                for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                    const Face2D_t& face = faces[destination_element.faces_[0][face_index]];
                    if (nodes[face.nodes_[0]].almost_equal(side_new_node)) {
                        new_node_index = face.nodes_[0];
                        break;
                    } 
                    else if (nodes[face.nodes_[1]].almost_equal(side_new_node)) {
                        new_node_index = face.nodes_[1];
                        break;
                    }
                }
            }

            new_elements[new_element_index].N_     = N[i];
            new_elements[new_element_index].nodes_ = {destination_element.nodes_[0],
                                                      new_node_index,
                                                      new_node_index,
                                                      destination_element.nodes_[0]};
            new_elements[new_element_index].status_ = destination_element.status_;
            new_elements[new_element_index].rotation_ = destination_element.rotation_;
            new_elements[new_element_index].split_level_ = destination_element.split_level_ + 1;
            new_elements[new_element_index].refine_ = false;
            new_elements[new_element_index].coarsen_ = false;
            new_elements[new_element_index].additional_nodes_ = {false, false, false, false};
            new_elements[new_element_index].allocate_boundary_storage();

            new_elements[new_element_index + 1].N_ = N[i];
            new_elements[new_element_index + 1].nodes_ = {new_node_index,
                                                          destination_element.nodes_[1],
                                                          destination_element.nodes_[1],
                                                          new_node_index};
            new_elements[new_element_index + 1].status_ = destination_element.status_;
            new_elements[new_element_index + 1].rotation_ = destination_element.rotation_;
            new_elements[new_element_index + 1].split_level_ = destination_element.split_level_ + 1;
            new_elements[new_element_index + 1].refine_ = false;
            new_elements[new_element_index + 1].coarsen_ = false;
            new_elements[new_element_index + 1].additional_nodes_ = {false, false, false, false};
            new_elements[new_element_index + 1].allocate_boundary_storage();

            const std::array<Vec2<hostFloat>, 4> points {nodes[new_elements[new_element_index].nodes_[0]],
                                                           new_node,
                                                           new_node,
                                                           nodes[new_elements[new_element_index].nodes_[3]]};
            new_elements[new_element_index].compute_boundary_geometry(points, polynomial_nodes[N[i]]);

            const std::array<Vec2<hostFloat>, 4> points_2 {new_node,
                                                             nodes[new_elements[new_element_index + 1].nodes_[1]],
                                                             nodes[new_elements[new_element_index + 1].nodes_[2]],
                                                             new_node};
            new_elements[new_element_index + 1].compute_boundary_geometry(points_2, polynomial_nodes[N[i]]);

            if (destination_element.additional_nodes_[0] && !elements_creating_node[i]) { // Is this always the case, having only one face?
                const size_t face_index = destination_element.faces_[0][0];

                size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                for (size_t j = 0; j < face_index; ++j) {
                    splitting_face_index += faces[j].refine_;
                }

                if (faces[face_index].elements_[0] == destination_element_index) {
                    new_elements[new_element_index].faces_[0][0] = face_index;
                    new_elements[new_element_index + 1].faces_[0][0] = splitting_face_index;
                }
                else {
                    new_elements[new_element_index].faces_[0][0] = splitting_face_index;
                    new_elements[new_element_index + 1].faces_[0][0] = face_index;
                }
            }
            else {
                std::array<size_t, 2> n_side_faces {0, 0};

                const std::array<Vec2<hostFloat>, 2> AB {
                    new_node - nodes[destination_element.nodes_[0]],
                    nodes[destination_element.nodes_[1]] - new_node
                };

                const std::array<hostFloat, 2> AB_dot_inv {
                    1/AB[0].dot(AB[0]),
                    1/AB[1].dot(AB[1])
                };

                for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = destination_element.faces_[0][side_face_index];
                    const Face2D_t& face = faces[face_index];
                    if (face.refine_) {
                        const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            face_new_node - nodes[destination_element.nodes_[0]],
                            face_new_node - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AE {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> E_proj {
                            AE[0].dot(AB[0]) * AB_dot_inv[0],
                            AE[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The first half of the face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[0];
                        }
                        // The first half of the face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[1];
                        }
                        // The second half of the face is within the first element
                        if (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[0];
                        }
                        // The second half of the face is within the second element
                        if (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[1];
                        }
                    }
                    else {
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[0];
                        }
                        // The face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            ++n_side_faces[1];
                        }
                    }
                }

                new_elements[new_element_index].faces_[0] = std::vector<size_t>(n_side_faces[0]);
                new_elements[new_element_index + 1].faces_[0] = std::vector<size_t>(n_side_faces[1]);

                std::array<size_t, 2> new_element_side_face_index {0, 0};
                for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = destination_element.faces_[0][side_face_index];
                    const Face2D_t& face = faces[face_index];
                    if (face.refine_) {

                        size_t splitting_face_index = n_faces + 4 * n_splitting_elements;
                        for (size_t j = 0; j < face_index; ++j) {
                            splitting_face_index += faces[j].refine_;
                        }
                        const Vec2<hostFloat> face_new_node = (nodes[face.nodes_[0]] + nodes[face.nodes_[1]])/2;
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            face_new_node - nodes[destination_element.nodes_[0]],
                            face_new_node - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AE {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> E_proj {
                            AE[0].dot(AB[0]) * AB_dot_inv[0],
                            AE[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The first half of the face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index].faces_[0][new_element_side_face_index[0]] = face_index;
                            ++new_element_side_face_index[0];
                        }
                        // The first half of the face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index + 1].faces_[0][new_element_side_face_index[1]] = face_index;
                            ++new_element_side_face_index[1];
                        }
                        // The second half of the face is within the first element
                        if (D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index].faces_[0][new_element_side_face_index[0]] = splitting_face_index;
                            ++new_element_side_face_index[0];
                        }
                        // The second half of the face is within the second element
                        if (D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && E_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && E_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index + 1].faces_[0][new_element_side_face_index[1]] = splitting_face_index;
                            ++new_element_side_face_index[1];
                        }
                    }
                    else {
                        const std::array<Vec2<hostFloat>, 2> AC {
                            nodes[face.nodes_[0]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[0]] - new_node
                        };
                        const std::array<Vec2<hostFloat>, 2> AD {
                            nodes[face.nodes_[1]] - nodes[destination_element.nodes_[0]],
                            nodes[face.nodes_[1]] - new_node
                        };

                        const std::array<hostFloat, 2> C_proj {
                            AC[0].dot(AB[0]) * AB_dot_inv[0],
                            AC[1].dot(AB[1]) * AB_dot_inv[1]
                        };
                        const std::array<hostFloat, 2> D_proj {
                            AD[0].dot(AB[0]) * AB_dot_inv[0],
                            AD[1].dot(AB[1]) * AB_dot_inv[1]
                        };

                        // The face is within the first element
                        if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index].faces_[0][new_element_side_face_index[0]] = face_index;
                            ++new_element_side_face_index[0];
                        }
                        // The face is within the second element
                        if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                            && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                            && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {
    
                            new_elements[new_element_index + 1].faces_[0][new_element_side_face_index[1]] = face_index;
                            ++new_element_side_face_index[1];
                        }
                    }
                }
            }
        }
        else {
            new_mpi_interfaces_destination[new_mpi_interface_index] = new_element_index;

            size_t side_n_splitting_faces = 0;
            for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
                side_n_splitting_faces += faces[destination_element.faces_[0][face_index]].refine_;
            }

            if (side_n_splitting_faces > 0) {
                std::vector<size_t> side_new_faces(destination_element.faces_[0].size() + side_n_splitting_faces);

                size_t side_new_face_index = 0;
                for (size_t side_face_index = 0; side_face_index < destination_element.faces_[0].size(); ++side_face_index) {
                    const size_t face_index = destination_element.faces_[0][side_face_index];
                    if (faces[face_index].refine_) {
                        side_new_faces[side_new_face_index] = face_index;

                        size_t new_face_index = n_faces + 4 * n_splitting_elements;
                        for (size_t j = 0; j < face_index; ++j) {
                            new_face_index += faces[j].refine_;
                        }

                        side_new_faces[side_new_face_index + 1] = new_face_index;

                        side_new_face_index += 2;
                    }
                    else {
                        side_new_faces[side_new_face_index] = face_index;
                        ++side_new_face_index;
                    }
                }

                destination_element.faces_[0] = std::move(side_new_faces);
            }

            if (destination_element.N_ != N[i]) {
                destination_element.resize_boundary_storage(N[i]);
                const std::array<Vec2<hostFloat>, 4> points {nodes[destination_element.nodes_[0]],
                                                               nodes[destination_element.nodes_[1]],
                                                               nodes[destination_element.nodes_[2]],
                                                               nodes[destination_element.nodes_[3]]};
                destination_element.compute_boundary_geometry(points, polynomial_nodes[destination_element.N_]);
            }

            new_elements[new_element_index] = std::move(destination_element);
        }
    }
}

auto SEM::Host::Meshes::adjust_boundaries(
        size_t n_boundaries, 
        Element2D_t* elements, 
        const size_t* boundaries, 
        const Face2D_t* faces) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_boundaries; ++boundary_index) {
        Element2D_t& destination_element = elements[boundaries[boundary_index]];
        int N_element = destination_element.N_;

        for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
            const Face2D_t& face = faces[destination_element.faces_[0][face_index]];
            const int element_index = face.elements_[0] == boundaries[boundary_index];
            const Element2D_t& source_element = elements[face.elements_[element_index]];

            N_element = std::max(N_element, source_element.N_);
        }

        if (destination_element.N_ != N_element) {
            destination_element.resize_boundary_storage(N_element);
        }
    }
}

auto SEM::Host::Meshes::adjust_interfaces(
        size_t n_local_interfaces, 
        Element2D_t* elements, 
        const size_t* local_interfaces_origin, 
        const size_t* local_interfaces_destination) -> void {
    
    for (size_t interface_index = 0; interface_index < n_local_interfaces; ++interface_index) {
        const Element2D_t& source_element = elements[local_interfaces_origin[interface_index]];
        Element2D_t& destination_element = elements[local_interfaces_destination[interface_index]];

        if (destination_element.N_ != source_element.N_) {
            destination_element.resize_boundary_storage(source_element.N_);
        }
    }
}

auto SEM::Host::Meshes::adjust_faces(
        size_t n_faces, 
        Face2D_t* faces, 
        const Element2D_t* elements) -> void {
    
    for (size_t face_index = 0; face_index < n_faces; ++face_index) {
        Face2D_t& face = faces[face_index];
        const Element2D_t& element_L = elements[face.elements_[0]];
        const Element2D_t& element_R = elements[face.elements_[1]];

        const int N_face = std::max(element_L.N_, element_R.N_);

        if (face.N_ != N_face) {
            face.resize_storage(N_face);
        }
    }
}

auto SEM::Host::Meshes::adjust_faces_neighbours(
        size_t n_faces, 
        Face2D_t* faces, 
        const Element2D_t* elements, 
        const Vec2<hostFloat>* nodes, 
        int max_split_level, 
        int N_max, 
        const size_t* elements_new_indices) -> void {
    
    for (size_t face_index = 0; face_index < n_faces; ++face_index) {
        Face2D_t& face = faces[face_index];
        const size_t element_L_index = face.elements_[0];
        const size_t element_R_index = face.elements_[1];
        const size_t element_L_side_index = face.elements_side_[0];
        const size_t element_R_side_index = face.elements_side_[1];
        const Element2D_t& element_L = elements[element_L_index];
        const Element2D_t& element_R = elements[element_R_index];
        const size_t element_L_new_index = elements_new_indices[element_L_index];
        const size_t element_R_new_index = elements_new_indices[element_R_index];
        std::array<size_t, 2> face_new_element_indices = {element_L_new_index, element_R_new_index};

        const int face_N = std::max(element_L.N_ + 2 * element_L.would_p_refine(N_max), element_R.N_ + 2 * element_R.would_p_refine(N_max));

        if (face.N_ != face_N) {
            face.resize_storage(face_N);
        }

        if (element_L.would_h_refine(max_split_level) || element_R.would_h_refine(max_split_level)) {
            const size_t element_L_next_side_index = (element_L_side_index + 1 < element_L.nodes_.size()) ? element_L_side_index + 1 : size_t{0};
            const size_t element_R_next_side_index = (element_R_side_index + 1 < element_R.nodes_.size()) ? element_R_side_index + 1 : size_t{0};

            std::array<Vec2<hostFloat>, 2> elements_centres {element_L.center_, element_R.center_};
            const std::array<Vec2<hostFloat>, 2> face_nodes {nodes[face.nodes_[0]], nodes[face.nodes_[1]]};
            std::array<std::array<Vec2<hostFloat>, 2>, 2> element_geometry_nodes {
                std::array<Vec2<hostFloat>, 2>{nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[element_L_next_side_index]]},
                std::array<Vec2<hostFloat>, 2>{nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[element_R_next_side_index]]}
            };
        
            if (element_L.would_h_refine(max_split_level)) {
                const std::array<size_t, 4> child_order_L = Hilbert::child_order(element_L.status_, element_L.rotation_);

                const std::array<Vec2<hostFloat>, 2> element_nodes {nodes[element_L.nodes_[element_L_side_index]], nodes[element_L.nodes_[element_L_next_side_index]]};
                const Vec2<hostFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;
                Vec2<hostFloat> element_centre {0};
                for (size_t element_side_index = 0; element_side_index < element_L.nodes_.size(); ++element_side_index) {
                    element_centre += nodes[element_L.nodes_[element_side_index]];
                }
                element_centre /= element_L.nodes_.size();

                const std::array<Vec2<hostFloat>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                const std::array<hostFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                const std::array<Vec2<hostFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                const std::array<Vec2<hostFloat>, 2> AD {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                const std::array<hostFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<hostFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};

                // The face is within the first element
                if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    face_new_element_indices[0] += child_order_L[element_L_side_index];
                    element_geometry_nodes[0] = {element_nodes[0], new_element_node};

                    const size_t previous_side_index = (element_L_side_index > 0) ? element_L_side_index - 1 : element_L.nodes_.size() - 1;
                    elements_centres[0] = (element_nodes[0] + new_element_node + element_centre + nodes[element_L.nodes_[previous_side_index]])/4; // CHECK only works on quadrilaterals

                }
                // The face is within the second element
                if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    face_new_element_indices[0] += child_order_L[element_L_next_side_index];
                    element_geometry_nodes[0] = {new_element_node, element_nodes[1]};

                    const size_t opposite_side_index = (element_L_side_index + 2 < element_L.nodes_.size()) ? element_L_side_index + 2 : element_L_side_index + 2 - element_L.nodes_.size();
                    elements_centres[0] = (new_element_node + element_nodes[1] + element_centre + nodes[element_R.nodes_[opposite_side_index]])/4; // CHECK only works on quadrilaterals

                }

            }
            if (element_R.would_h_refine(max_split_level)) {
                const std::array<size_t, 4> child_order_R = Hilbert::child_order(element_R.status_, element_R.rotation_);

                const std::array<Vec2<hostFloat>, 2> element_nodes {nodes[element_R.nodes_[element_R_side_index]], nodes[element_R.nodes_[element_R_next_side_index]]};
                const Vec2<hostFloat> new_element_node = (element_nodes[0] + element_nodes[1])/2;
                Vec2<hostFloat> element_centre {0};
                for (size_t element_side_index = 0; element_side_index < element_R.nodes_.size(); ++element_side_index) {
                    element_centre += nodes[element_R.nodes_[element_side_index]];
                }
                element_centre /= element_R.nodes_.size();

                const std::array<Vec2<hostFloat>, 2> AB {new_element_node - element_nodes[0], element_nodes[1] - new_element_node};
                const std::array<hostFloat, 2> AB_dot_inv  {1/AB[0].dot(AB[0]), 1/AB[1].dot(AB[1])};

                const std::array<Vec2<hostFloat>, 2> AC {nodes[face.nodes_[0]] - element_nodes[0], nodes[face.nodes_[0]] - new_element_node};
                const std::array<Vec2<hostFloat>, 2> AD {nodes[face.nodes_[1]] - element_nodes[0], nodes[face.nodes_[1]] - new_element_node};

                const std::array<hostFloat, 2> C_proj {AC[0].dot(AB[0]) * AB_dot_inv[0], AC[1].dot(AB[1]) * AB_dot_inv[1]};
                const std::array<hostFloat, 2> D_proj {AD[0].dot(AB[0]) * AB_dot_inv[0], AD[1].dot(AB[1]) * AB_dot_inv[1]};

                // The face is within the first element
                if (C_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[0] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[0] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    face_new_element_indices[1] += child_order_R[element_R_side_index];
                    element_geometry_nodes[1] = {element_nodes[0], new_element_node};
                        
                    const size_t previous_side_index = (element_R_side_index > 0) ? element_R_side_index - 1 : element_R.nodes_.size() - 1;
                    elements_centres[1] = (element_nodes[0] + new_element_node + element_centre + nodes[element_R.nodes_[previous_side_index]])/4; // CHECK only works on quadrilaterals
                }
                // The face is within the second element
                if (C_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && C_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()
                    && D_proj[1] + std::numeric_limits<hostFloat>::epsilon() >= hostFloat{0} 
                    && D_proj[1] <= hostFloat{1} + std::numeric_limits<hostFloat>::epsilon()) {

                    face_new_element_indices[1] += child_order_R[element_R_next_side_index];
                    element_geometry_nodes[1] = {new_element_node, element_nodes[1]};
                        
                    const size_t opposite_side_index = (element_R_side_index + 2 < element_R.nodes_.size()) ? element_R_side_index + 2 : element_R_side_index + 2 - element_R.nodes_.size();
                    elements_centres[1] = (new_element_node + element_nodes[1] + element_centre + nodes[element_R.nodes_[opposite_side_index]])/4; // CHECK only works on quadrilaterals
                }
            }

            face.compute_geometry(elements_centres, face_nodes, element_geometry_nodes);
        }

        face.elements_ = face_new_element_indices;
    }
}

auto SEM::Host::Meshes::move_boundaries(
        size_t n_boundaries, 
        size_t offset, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        size_t* boundaries, 
        const Face2D_t* faces, 
        size_t* elements_new_indices) -> void {
    
    for (size_t boundary_index = 0; boundary_index < n_boundaries; ++boundary_index) {
        const size_t boundary_element_index = boundaries[boundary_index];
        Element2D_t& destination_element = elements[boundary_element_index];
        const size_t new_element_index = offset + boundary_index;
        int N_element = destination_element.N_;
        elements_new_indices[boundary_element_index] = new_element_index;
        boundaries[boundary_index] = new_element_index;

        for (size_t face_index = 0; face_index < destination_element.faces_[0].size(); ++face_index) {
            const Face2D_t& face = faces[destination_element.faces_[0][face_index]];
            const int element_index = face.elements_[0] == boundary_element_index;
            const Element2D_t& source_element = elements[face.elements_[element_index]];

            N_element = std::max(N_element, source_element.N_);
        }

        if (destination_element.N_ != N_element) {
            destination_element.resize_boundary_storage(N_element);
        }

        new_elements[new_element_index] = std::move(destination_element);
    }
}

auto SEM::Host::Meshes::move_interfaces(
        size_t n_local_interfaces, 
        size_t offset,
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const size_t* local_interfaces_origin, 
        const size_t* local_interfaces_destination, 
        size_t* elements_new_indices) -> void {

    for (size_t interface_index = 0; interface_index < n_local_interfaces; ++interface_index) {
        const size_t destination_element_index = local_interfaces_destination[interface_index];
        const Element2D_t& source_element = elements[local_interfaces_origin[interface_index]];
        Element2D_t& destination_element = elements[destination_element_index];
        const size_t new_element_index = offset + interface_index;
        elements_new_indices[destination_element_index] = new_element_index;

        if (destination_element.N_ != source_element.N_) {
            destination_element.resize_boundary_storage(source_element.N_);
        }

        new_elements[new_element_index] = std::move(elements[destination_element_index]);
    }
}

auto SEM::Host::Meshes::get_transfer_solution(
        size_t n_elements, 
        const Element2D_t* elements, 
        int maximum_N, 
        const Vec2<hostFloat>* nodes, 
        hostFloat* solution, 
        size_t* n_neighbours, 
        hostFloat* element_nodes) -> void {
    
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t p_offset =  3 * element_index      * std::pow(maximum_N + 1, 2);
        const size_t u_offset = (3 * element_index + 1) * std::pow(maximum_N + 1, 2);
        const size_t v_offset = (3 * element_index + 2) * std::pow(maximum_N + 1, 2);
        const size_t neighbours_offset = 4 * element_index;
        const size_t nodes_offset = 8 * element_index;

        const Element2D_t& element = elements[element_index];

        n_neighbours[neighbours_offset]     = element.faces_[0].size();
        n_neighbours[neighbours_offset + 1] = element.faces_[1].size();
        n_neighbours[neighbours_offset + 2] = element.faces_[2].size();
        n_neighbours[neighbours_offset + 3] = element.faces_[3].size();
        element_nodes[nodes_offset]     = nodes[element.nodes_[0]].x();
        element_nodes[nodes_offset + 1] = nodes[element.nodes_[0]].y();
        element_nodes[nodes_offset + 2] = nodes[element.nodes_[1]].x();
        element_nodes[nodes_offset + 3] = nodes[element.nodes_[1]].y();
        element_nodes[nodes_offset + 4] = nodes[element.nodes_[2]].x();
        element_nodes[nodes_offset + 5] = nodes[element.nodes_[2]].y();
        element_nodes[nodes_offset + 6] = nodes[element.nodes_[3]].x();
        element_nodes[nodes_offset + 7] = nodes[element.nodes_[3]].y();

        for (int i = 0; i < std::pow(element.N_ + 1, 2); ++i) {
            solution[p_offset + i] = element.p_[i];
            solution[u_offset + i] = element.u_[i];
            solution[v_offset + i] = element.v_[i];
        }
    }
}

auto SEM::Host::Meshes::fill_received_elements(
        size_t n_elements, 
        size_t element_offset, 
        Element2D_t* elements, 
        int maximum_N, 
        const Vec2<hostFloat>* nodes, 
        const hostFloat* solution, 
        const size_t* received_node_indices, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes) -> void {
    
    for (size_t i = 0; i < n_elements; ++i) {
        const size_t p_offset =  3 * i      * std::pow(maximum_N + 1, 2);
        const size_t u_offset = (3 * i + 1) * std::pow(maximum_N + 1, 2);
        const size_t v_offset = (3 * i + 2) * std::pow(maximum_N + 1, 2);
        const size_t neighbours_offset = 4 * i;

        const size_t element_index = element_offset + i;
        Element2D_t& element = elements[element_index];
        element.allocate_storage();

        element.nodes_[0] = received_node_indices[neighbours_offset];
        element.nodes_[1] = received_node_indices[neighbours_offset + 1];
        element.nodes_[2] = received_node_indices[neighbours_offset + 2];
        element.nodes_[3] = received_node_indices[neighbours_offset + 3];

        for (int j = 0; j < std::pow(element.N_ + 1, 2); ++j) {
            element.p_[j] = solution[p_offset + j];
            element.u_[j] = solution[u_offset + j];
            element.v_[j] = solution[v_offset + j];
        }

        const std::array<Vec2<hostFloat>, 4> received_element_nodes {
            nodes[element.nodes_[0]],
            nodes[element.nodes_[1]],
            nodes[element.nodes_[2]],
            nodes[element.nodes_[3]]
        };

        element.compute_geometry(received_element_nodes, polynomial_nodes[element.N_]);
    }
}

auto SEM::Host::Meshes::fill_received_elements_faces(
        size_t n_elements, 
        size_t element_offset, 
        size_t face_offset, 
        size_t n_domain_elements, 
        size_t n_elements_recv_left, 
        size_t n_elements_recv_right, 
        Element2D_t* elements, 
        Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        const size_t* n_neighbours, 
        const size_t* n_neighbours_left, 
        const size_t* n_neighbours_right, 
        const size_t* neighbours_offsets, 
        const size_t* neighbours_offsets_left, 
        const size_t* neighbours_offsets_right, 
        const size_t* neighbours_indices, 
        const size_t* neighbours_sides, 
        const int* neighbours_procs, 
        const size_t* face_offsets, 
        const size_t* face_offsets_left, 
        const size_t* face_offsets_right) -> void {

    for (size_t i = 0; i < n_elements; ++i) {
        const size_t neighbours_offset = 4 * i;

        const size_t element_index = element_offset + i;
        Element2D_t& element = elements[element_index];
        
        size_t neighbours_index = neighbours_offsets[i];
        size_t face_index = face_offset + face_offsets[i];
        for (size_t j = 0; j < element.faces_.size(); ++j) {
            const size_t side_n_neighbours = n_neighbours[neighbours_offset + j];
            element.faces_[j] = std::vector<size_t>(side_n_neighbours);

            for (size_t k = 0; k < side_n_neighbours; ++k) {
                const size_t neighbour_index = neighbours_index + k;
                const size_t neighbour_element_index = neighbours_indices[neighbour_index];
                const size_t neighbour_side = neighbours_sides[neighbour_index];
                const Element2D_t& neighbour_element = elements[neighbour_element_index];

                if (neighbour_element_index >= n_domain_elements 
                        || (neighbour_element_index < n_elements_recv_left && neighbour_element_index >= element_index)
                        || (neighbour_element_index >= n_domain_elements - n_elements_recv_right && neighbour_element_index >= element_index)) { 
                    
                    // We create
                    const size_t next_side = (j + 1 < element.nodes_.size()) ? j + 1 : 0;
                    const size_t neighbour_next_side = (neighbour_side + 1 < neighbour_element.nodes_.size()) ? neighbour_side + 1 : 0;

                    const int face_N = std::max(element.N_, neighbour_element.N_);
                    const std::array<std::array<size_t, 2>, 2> node_indices {
                        std::array<size_t, 2>{element.nodes_[j], element.nodes_[next_side]},
                        std::array<size_t, 2>{neighbour_element.nodes_[neighbour_next_side], neighbour_element.nodes_[neighbour_side]}
                    };
                    const std::array<std::array<Vec2<hostFloat>, 2>, 2> element_nodes {
                        std::array<Vec2<hostFloat>, 2> {nodes[node_indices[0][0]], nodes[node_indices[0][1]]},
                        std::array<Vec2<hostFloat>, 2> {nodes[node_indices[1][1]], nodes[node_indices[1][0]]}
                    };
                    const std::array<hostFloat, 2> side_lengths {
                        (element_nodes[0][1] - element_nodes[0][0]).magnitude(),
                        (element_nodes[1][1] - element_nodes[1][0]).magnitude()
                    };

                    const std::array<size_t, 2> face_node_indices = node_indices[side_lengths[0] > side_lengths[1]];

                    faces[face_index] = Face2D_t(face_N, face_node_indices, {element_index, neighbour_element_index}, {j, neighbour_side});

                    const std::array<Vec2<hostFloat>, 2> face_nodes {nodes[face_node_indices[0]], nodes[face_node_indices[1]]};

                    faces[face_index].compute_geometry({element.center_, neighbour_element.center_}, face_nodes, element_nodes);
                    
                    element.faces_[j][k] = face_index;
                   
                    if (neighbours_procs[neighbour_index] < 0) {
                        elements[neighbour_element_index].faces_[0][0] = face_index;
                    }

                    ++face_index;
                }
                else if (neighbour_element_index < n_elements_recv_left && neighbour_element_index < element_index) {
                    // They create
                    size_t neighbour_face_index = face_offset + face_offsets_left[neighbour_element_index];
                    const size_t neighbours_neighbour_offset = 4 * neighbour_element_index;
                    size_t neighbours_neighbour_index = neighbours_offsets_left[neighbour_element_index];
                    bool found_self = false;

                    // This is insane
                    for (size_t m = 0; m < neighbour_element.faces_.size(); ++m) {
                        const size_t neighbour_side_n_neighbours = n_neighbours_left[neighbours_neighbour_offset + m];
                        for (size_t n = 0; n < neighbour_side_n_neighbours; ++n) {
                            const size_t neighbour_neighbour_index = neighbours_neighbour_index + n;
                            const size_t neighbour_neighbour_element_index = neighbours_indices[neighbour_neighbour_index];
                            const size_t neighbour_neighbour_side = neighbours_sides[neighbour_neighbour_index];

                            if (neighbour_neighbour_element_index >= n_domain_elements 
                                || (neighbour_neighbour_element_index < n_elements_recv_left && neighbour_neighbour_element_index >= neighbour_element_index)
                                || (neighbour_neighbour_element_index >= n_domain_elements - n_elements_recv_right && neighbour_neighbour_element_index >= neighbour_element_index)) { 

                                if (neighbour_neighbour_element_index == element_index && neighbour_neighbour_side == j) {
                                    element.faces_[j][k] = neighbour_face_index;
                                    found_self = true;
                                    break;
                                }
                                ++neighbour_face_index;
                            }

                        }

                        if (found_self) {
                            break;
                        }

                        neighbours_neighbour_index += neighbour_side_n_neighbours;
                    }   
                } 
                else if (neighbour_element_index >= n_domain_elements - n_elements_recv_right && neighbour_element_index < element_index) {
                    // They create
                    const size_t neighbour_recv_index = neighbour_element_index + n_elements_recv_right - n_domain_elements;
                    size_t neighbour_face_index = face_offset + face_offsets_right[neighbour_recv_index];
                    const size_t neighbours_neighbour_offset = 4 * neighbour_recv_index;
                    size_t neighbours_neighbour_index = neighbours_offsets_right[neighbour_recv_index];
                    bool found_self = false;

                    // This is insane
                    for (size_t m = 0; m < neighbour_element.faces_.size(); ++m) {
                        const size_t neighbour_side_n_neighbours = n_neighbours_right[neighbours_neighbour_offset + m];
                        for (size_t n = 0; n < neighbour_side_n_neighbours; ++n) {
                            const size_t neighbour_neighbour_index = neighbours_neighbour_index + n;
                            const size_t neighbour_neighbour_element_index = neighbours_indices[neighbour_neighbour_index];
                            const size_t neighbour_neighbour_side = neighbours_sides[neighbour_neighbour_index];

                            if (neighbour_neighbour_element_index >= n_domain_elements 
                                || (neighbour_neighbour_element_index < n_elements_recv_left && neighbour_neighbour_element_index >= neighbour_element_index)
                                || (neighbour_neighbour_element_index >= n_domain_elements - n_elements_recv_right && neighbour_neighbour_element_index >= neighbour_element_index)) { 

                                if (neighbour_neighbour_element_index == element_index && neighbour_neighbour_side == j) {
                                    element.faces_[j][k] = neighbour_face_index;
                                    found_self = true;
                                    break;
                                }
                                ++neighbour_face_index;
                            }

                        }

                        if (found_self) {
                            break;
                        }

                        neighbours_neighbour_index += neighbour_side_n_neighbours;
                    }  
                }
                else { // In domain, we reuse a face. The face will already have the correct info
                    for (size_t m = 0; m < neighbour_element.faces_[neighbour_side].size(); ++m) {
                        const size_t neighbour_face_index = neighbour_element.faces_[neighbour_side][m]; // CHECK should probably check for -1
                        const Face2D_t& face = faces[neighbour_face_index];
                        const size_t face_side_index = face.elements_[0] == neighbour_element_index;
                        if (face.elements_[face_side_index] == element_index && face.elements_side_[face_side_index] == j) {
                            element.faces_[j][k] = neighbour_face_index;
                            break;
                        }
                    }
                }
            }

            neighbours_index += side_n_neighbours;
        }
    }
}

auto SEM::Host::Meshes::get_neighbours(
        size_t n_elements_send, 
        size_t start_index, 
        size_t n_domain_elements_old, 
        size_t n_wall_boundaries, 
        size_t n_symmetry_boundaries, 
        size_t n_inflow_boundaries, 
        size_t n_outflow_boundaries, 
        size_t n_local_interfaces, 
        size_t n_MPI_interface_elements_receiving, 
        int rank, 
        int n_procs, 
        const Element2D_t* elements, 
        const Face2D_t* faces, 
        const Vec2<hostFloat>* nodes, 
        const size_t* wall_boundaries, 
        const size_t* symmetry_boundaries, 
        const size_t* inflow_boundaries, 
        const size_t* outflow_boundaries, 
        const size_t* interfaces_destination, 
        const size_t* interfaces_origin, 
        const size_t* mpi_interfaces_destination, 
        const int* mpi_interfaces_process, 
        const size_t* mpi_interfaces_local_indices, 
        const size_t* mpi_interfaces_side, 
        const size_t* offsets, 
        const size_t* n_elements_received_left, 
        const size_t* n_elements_sent_left, 
        const size_t* n_elements_received_right, 
        const size_t* n_elements_sent_right, 
        const size_t* global_element_offset, 
        const size_t* global_element_offset_new, 
        const size_t* global_element_offset_end_new, 
        size_t* neighbours, 
        hostFloat* neighbours_nodes, 
        int* neighbours_proc, 
        size_t* neighbours_side, 
        int* neighbours_N) -> void {

    for (size_t i = 0; i < n_elements_send; ++i) {
        const size_t element_index = i + start_index;
        const Element2D_t& element = elements[element_index];
        size_t element_offset = offsets[i];
        for (size_t side_index = 0; side_index < element.faces_.size(); ++side_index) {
            for (size_t side_face_index = 0; side_face_index < element.faces_[side_index].size(); ++side_face_index) {
                const size_t face_index = element.faces_[side_index][side_face_index];
                const Face2D_t& face = faces[face_index];
                const size_t face_side_index = face.elements_[0] == element_index;
                const size_t other_element_index = face.elements_[face_side_index];
                const Element2D_t& other_element = elements[other_element_index];

                const std::array<size_t, 2> neighbour_node_indices {
                    other_element.nodes_[face.elements_side_[face_side_index]],
                    other_element.nodes_[(face.elements_side_[face_side_index] + 1 < other_element.nodes_.size()) ? face.elements_side_[face_side_index] + 1 : 0],
                };
                neighbours_nodes[4 * element_offset]     = nodes[neighbour_node_indices[0]].x();
                neighbours_nodes[4 * element_offset + 1] = nodes[neighbour_node_indices[0]].y();
                neighbours_nodes[4 * element_offset + 2] = nodes[neighbour_node_indices[1]].x();
                neighbours_nodes[4 * element_offset + 3] = nodes[neighbour_node_indices[1]].y();

                neighbours_N[element_offset] = other_element.N_;

                if (other_element_index < n_domain_elements_old) {
                    if (other_element_index < n_elements_sent_left[rank] || other_element_index >= n_domain_elements_old - n_elements_sent_right[rank]) {
                        const size_t element_global_index = other_element_index + global_element_offset[rank];
                        neighbours_proc[element_offset] = -1;

                        for (int process_index = 0; process_index < n_procs; ++process_index) {
                            if (element_global_index >= global_element_offset_new[process_index] && element_global_index < global_element_offset_end_new[process_index]) {
                                neighbours[element_offset] = element_global_index - global_element_offset_new[process_index];
                                neighbours_proc[element_offset] = process_index;

                                break;
                            }
                        }

                        if (neighbours_proc[element_offset] == -1) {
                            std::cerr << "Error: Element " << element_index << ", side " << side_index << ", face " << side_face_index << " with index " << face_index << " has other sent element " << element_global_index << " but it is not found in any process. Exiting." << std::endl;
                            exit(65);
                        }
                    }
                    else {
                        neighbours[element_offset] = other_element_index + n_elements_received_left[rank] - n_elements_sent_left[rank];
                        neighbours_proc[element_offset] = rank;
                    }
                    neighbours_side[element_offset] = face.elements_side_[face_side_index];
                }
                else { // It is either a local or mpi interface
                    // It could also be a boundary condition
                    // This is a nightmare
                    bool missing = true;
                    for (size_t j = 0; j < n_local_interfaces; ++j) {
                        if (interfaces_destination[j] == other_element_index) {
                            if (interfaces_origin[j] < n_elements_sent_left[rank] || interfaces_origin[j] >= n_domain_elements_old - n_elements_sent_right[rank]) {
                                const size_t element_global_index = interfaces_origin[j] + global_element_offset[rank];
                                neighbours_proc[element_offset] = -1;

                                for (int process_index = 0; process_index < n_procs; ++process_index) {
                                    if (element_global_index >= global_element_offset_new[process_index] && element_global_index < global_element_offset_end_new[process_index]) {
                                        neighbours[element_offset] = element_global_index - global_element_offset_new[process_index];
                                        neighbours_proc[element_offset] = process_index;

                                        break;
                                    }
                                }

                                if (neighbours_proc[element_offset] == -1) {
                                    std::cerr << "Error: Element " << element_index << ", side " << side_index << ", face " << side_face_index << " with index " << face_index << " has other local sent element " << element_global_index << " but it is not found in any process. Exiting." << std::endl;
                                    exit(66);
                                }
                            }
                            else {
                                neighbours[element_offset] = interfaces_origin[j] + n_elements_received_left[rank] - n_elements_sent_left[rank];
                                neighbours_proc[element_offset] = rank;
                            }
                            neighbours_side[element_offset] = mpi_interfaces_side[j];

                            missing = false;
                            break;
                        }
                    }

                    if (missing) {
                        for (size_t j = 0; j < n_MPI_interface_elements_receiving; ++j) {
                            if (mpi_interfaces_destination[j] == other_element_index) {
                                neighbours[element_offset] = mpi_interfaces_local_indices[j];
                                neighbours_proc[element_offset] = mpi_interfaces_process[j];
                                neighbours_side[element_offset] = mpi_interfaces_side[j];
    
                                missing = false;
                                break;
                            }
                        }

                        if (missing) {
                            for (size_t j = 0; j < n_wall_boundaries; ++j) {
                                if (wall_boundaries[j] == other_element_index) {
                                    neighbours[element_offset] = static_cast<size_t>(-1);
                                    neighbours_proc[element_offset] = SEM::Host::Meshes::Mesh2D_t::boundary_type::wall;
                                    neighbours_side[element_offset] = static_cast<size_t>(-1);
        
                                    missing = false;
                                    break;
                                }
                            }

                            if (missing) {
                                for (size_t j = 0; j < n_symmetry_boundaries; ++j) {
                                    if (symmetry_boundaries[j] == other_element_index) {
                                        neighbours[element_offset] = static_cast<size_t>(-1);
                                        neighbours_proc[element_offset] = SEM::Host::Meshes::Mesh2D_t::boundary_type::symmetry;
                                        neighbours_side[element_offset] = static_cast<size_t>(-1);
            
                                        missing = false;
                                        break;
                                    }
                                }
    
                                if (missing) {
                                    for (size_t j = 0; j < n_inflow_boundaries; ++j) {
                                        if (inflow_boundaries[j] == other_element_index) {
                                            neighbours[element_offset] = static_cast<size_t>(-1);
                                            neighbours_proc[element_offset] = SEM::Host::Meshes::Mesh2D_t::boundary_type::inflow;
                                            neighbours_side[element_offset] = static_cast<size_t>(-1);
                
                                            missing = false;
                                            break;
                                        }
                                    }
        
                                    if (missing) {
                                        for (size_t j = 0; j < n_outflow_boundaries; ++j) {
                                            if (outflow_boundaries[j] == other_element_index) {
                                                neighbours[element_offset] = static_cast<size_t>(-1);
                                                neighbours_proc[element_offset] = SEM::Host::Meshes::Mesh2D_t::boundary_type::outflow;
                                                neighbours_side[element_offset] = static_cast<size_t>(-1);
                    
                                                missing = false;
                                                break;
                                            }
                                        }
            
                                        if (missing) {
                                            std::cout << "Error: Element " << other_element_index << " is not part of local or mpi interfaces, or boundaries. Results are undefined." << std::endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                ++element_offset;
            }
        }
    }
}

auto SEM::Host::Meshes::move_elements(
        size_t n_elements_move, 
        size_t n_elements_send_left, 
        size_t n_elements_recv_left, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const std::vector<bool>& faces_to_delete) -> void {
    
    for (size_t element_index = 0; element_index < n_elements_move; ++element_index) {
        const size_t source_element_index = element_index + n_elements_send_left;
        const size_t destination_element_index = element_index + n_elements_recv_left;

        new_elements[destination_element_index] = std::move(elements[source_element_index]);

        // We update indices
        for (size_t i = 0; i < new_elements[destination_element_index].faces_.size(); ++i) {
            for (size_t j = 0; j < new_elements[destination_element_index].faces_[i].size(); ++j) {
                const size_t face_index = new_elements[destination_element_index].faces_[i][j];

                if (faces_to_delete[face_index]) {
                    new_elements[destination_element_index].faces_[i][j] = static_cast<size_t>(-1); // This should not happen. Cue the thing happening and me searching for the bug for 8 hours.
                }
                else {
                    size_t new_face_index = face_index;
                    for (size_t k = 0; k < face_index; ++k) {
                        new_face_index -= faces_to_delete[k];
                    }
                    new_elements[destination_element_index].faces_[i][j] = new_face_index;
                }
            }
        }
    }
}

auto SEM::Host::Meshes::get_interface_n_processes(
        size_t n_mpi_interfaces, 
        size_t n_mpi_interfaces_incoming, 
        const Element2D_t* elements, 
        const Face2D_t* faces, 
        const size_t* mpi_interfaces_origin, 
        const size_t* mpi_interfaces_origin_side, 
        const size_t* mpi_interfaces_destination, 
        const int* mpi_interfaces_new_process_incoming, 
        size_t* n_processes) -> void {
    
    for (size_t interface_index = 0; interface_index < n_mpi_interfaces; ++interface_index) {
        const size_t origin_element_index = mpi_interfaces_origin[interface_index];
        const Element2D_t& origin_element = elements[origin_element_index];
        const size_t side_index = mpi_interfaces_origin_side[interface_index];
        const size_t n_faces = origin_element.faces_[side_index].size();

        n_processes[interface_index] = 0;

        bool duplicate_origin = false; // We must check if the same element already needs to be sent to two processes
        for (size_t i = 0; i < interface_index; ++i) {
            if (mpi_interfaces_origin[i] == origin_element_index) {
                duplicate_origin = true;
                break;
            }
        }
        if (duplicate_origin) {
            break;
        }

        for (size_t i = 0; i < n_faces; ++i) {
            const size_t face_index = origin_element.faces_[side_index][i];
            const Face2D_t& face = faces[face_index];
            const size_t face_side_index = face.elements_[0] == origin_element_index;
            const size_t other_element_index = face.elements_[face_side_index];

            size_t other_interface_index = static_cast<size_t>(-1);
            bool missing = true;
            for (size_t incoming_interface_index = 0; incoming_interface_index < n_mpi_interfaces_incoming; ++incoming_interface_index) {
                if (mpi_interfaces_destination[incoming_interface_index] == other_element_index) {
                    other_interface_index = incoming_interface_index;
                    missing = false;
                    break;
                }
            }
            if (missing) {
                printf("Error: Element %llu is not part of an mpi incoming interface. Results are undefined.\n", static_cast<unsigned long long>(other_element_index));
            }

            const int other_interface_process = mpi_interfaces_new_process_incoming[other_interface_index];

            bool missing_process = true;
            for (size_t j = 0; j < i; ++j) {
                const size_t previous_face_index = origin_element.faces_[side_index][j];
                const Face2D_t& previous_face = faces[previous_face_index];
                const size_t previous_face_side_index = previous_face.elements_[0] == origin_element_index;
                const size_t previous_other_element_index = previous_face.elements_[previous_face_side_index];

                size_t previous_other_interface_index = static_cast<size_t>(-1);
                bool previous_missing = true;
                for (size_t incoming_interface_index = 0; incoming_interface_index < n_mpi_interfaces_incoming; ++incoming_interface_index) {
                    if (mpi_interfaces_destination[incoming_interface_index] == previous_other_element_index) {
                        previous_other_interface_index = incoming_interface_index;
                        previous_missing = false;
                        break;
                    }
                }
                if (previous_missing) {
                    printf("Error: Previous element %llu is not part of an mpi incoming interface. Results are undefined.\n", static_cast<unsigned long long>(other_element_index));
                }

                const int previous_other_interface_process = mpi_interfaces_new_process_incoming[previous_other_interface_index];
                if (previous_other_interface_process == other_interface_process) {
                    missing_process = false;
                    break;
                }
            }

            if (missing_process) {
                ++n_processes[interface_index];
            }
        }
    }
}

auto SEM::Host::Meshes::get_interface_processes(
        size_t n_mpi_interfaces, 
        size_t n_mpi_interfaces_incoming, 
        const Element2D_t* elements, 
        const Face2D_t* faces, 
        const size_t* mpi_interfaces_origin, 
        const size_t* mpi_interfaces_origin_side, 
        const size_t* mpi_interfaces_destination, 
        const int* mpi_interfaces_new_process_incoming, 
        const size_t* process_offsets, 
        int* processes) -> void {
    
    for (size_t interface_index = 0; interface_index < n_mpi_interfaces; ++interface_index) {
        const size_t origin_element_index = mpi_interfaces_origin[interface_index];
        const Element2D_t& origin_element = elements[origin_element_index];
        const size_t side_index = mpi_interfaces_origin_side[interface_index];
        const size_t n_faces = origin_element.faces_[side_index].size();
        const size_t process_offset = process_offsets[interface_index];

        bool duplicate_origin = false; // We must check if the same element already needs to be sent to two processes
        for (size_t i = 0; i < interface_index; ++i) {
            if (mpi_interfaces_origin[i] == origin_element_index) {
                duplicate_origin = true;
                break;
            }
        }
        if (duplicate_origin) {
            break;
        }

        size_t process_index = 0;

        for (size_t i = 0; i < n_faces; ++i) {
            const size_t face_index = origin_element.faces_[side_index][i];
            const Face2D_t& face = faces[face_index];
            const size_t face_side_index = face.elements_[0] == origin_element_index;
            const size_t other_element_index = face.elements_[face_side_index];

            size_t other_interface_index = static_cast<size_t>(-1);
            bool missing = true;
            for (size_t incoming_interface_index = 0; incoming_interface_index < n_mpi_interfaces_incoming; ++incoming_interface_index) {
                if (mpi_interfaces_destination[incoming_interface_index] == other_element_index) {
                    other_interface_index = incoming_interface_index;
                    missing = false;
                    break;
                }
            }
            if (missing) {
                printf("Error: Element %llu is not part of an mpi incoming interface, its process is unknown. Results are undefined.\n", static_cast<unsigned long long>(other_element_index));
            }

            const int other_interface_process = mpi_interfaces_new_process_incoming[other_interface_index];

            bool missing_process = true;
            for (size_t j = 0; j < i; ++j) {
                const size_t previous_face_index = origin_element.faces_[side_index][j];
                const Face2D_t& previous_face = faces[previous_face_index];
                const size_t previous_face_side_index = previous_face.elements_[0] == origin_element_index;
                const size_t previous_other_element_index = previous_face.elements_[previous_face_side_index];

                size_t previous_other_interface_index = static_cast<size_t>(-1);
                bool previous_missing = true;
                for (size_t incoming_interface_index = 0; incoming_interface_index < n_mpi_interfaces_incoming; ++incoming_interface_index) {
                    if (mpi_interfaces_destination[incoming_interface_index] == previous_other_element_index) {
                        previous_other_interface_index = incoming_interface_index;
                        previous_missing = false;
                        break;
                    }
                }
                if (previous_missing) {
                    printf("Error: Previous element %llu is not part of an mpi incoming interface, its process is unknown. Results are undefined.\n", static_cast<unsigned long long>(other_element_index));
                }

                const int previous_other_interface_process = mpi_interfaces_new_process_incoming[previous_other_interface_index];
                if (previous_other_interface_process == other_interface_process) {
                    missing_process = false;
                    break;
                }
            }

            if (missing_process) {
                processes[process_offset + process_index] = other_interface_process;
                ++process_index;
            }
        }
    }
}

auto SEM::Host::Meshes::find_received_nodes(
        size_t n_nodes, 
        const Vec2<hostFloat>* nodes, 
        const hostFloat* received_nodes, 
        std::vector<bool>& missing_nodes, 
        std::vector<bool>& missing_received_nodes, 
        size_t* received_nodes_indices, 
        size_t* received_node_received_indices) -> void {
    
    for (size_t i = 0; i < missing_received_nodes.size(); ++i) {
        const Vec2<hostFloat> received_node(received_nodes[2 * i], received_nodes[2 * i + 1]);
        missing_nodes[i] = true;
        missing_received_nodes[i] = false;

        for (size_t j = 0; j < n_nodes; ++j) {
            if (received_node.almost_equal(nodes[j])) {
                missing_nodes[i] = false;
                received_nodes_indices[i] = j;
                break;
            }
        }

        if (missing_nodes[i]) {
            for (size_t j = 0; j < i; ++j) {
                if (received_node.almost_equal(Vec2<hostFloat>(received_nodes[2 * j], received_nodes[2 * j + 1]))) {
                    missing_nodes[i] = false;
                    missing_received_nodes[i] = true;
                    received_node_received_indices[i] = j;
                    break;
                }
            }
        }
    }
}

auto SEM::Host::Meshes::add_new_received_nodes(
        size_t n_nodes, 
        Vec2<hostFloat>* nodes, 
        const hostFloat* received_nodes, 
        const std::vector<bool>& missing_nodes, 
        const std::vector<bool>& missing_received_nodes, 
        size_t* received_nodes_indices, 
        const size_t* received_node_received_indices) -> void {
    
    for (size_t i = 0; i < missing_nodes.size(); ++i) {
        if (missing_nodes[i]) {
            size_t new_received_index = n_nodes;
            for (size_t j = 0; j < i; ++j) {
                new_received_index += missing_nodes[j];
            }

            nodes[new_received_index] = Vec2<hostFloat>(received_nodes[2 * i], received_nodes[2 * i + 1]);
            received_nodes_indices[i] = new_received_index;
        }
        else if (missing_received_nodes[i]) {
            size_t new_received_index = n_nodes;
            for (size_t j = 0; j < received_node_received_indices[i]; ++j) {
                new_received_index += missing_nodes[j];
            }

            received_nodes_indices[i] = new_received_index;
        }
    }
}

auto SEM::Host::Meshes::find_received_neighbour_nodes(
        size_t n_received_nodes, 
        size_t n_nodes, 
        const Vec2<hostFloat>* nodes, 
        const hostFloat* received_neighbour_nodes, 
        const hostFloat* received_nodes, 
        std::vector<bool>& missing_neighbour_nodes,
        std::vector<bool>& missing_received_neighbour_nodes, 
        size_t* received_neighbour_nodes_indices, 
        size_t* received_neighbour_node_received_indices) -> void {

    for (size_t i = 0; i < missing_neighbour_nodes.size(); ++i) {
        const Vec2<hostFloat> received_neighbour_node(received_neighbour_nodes[2 * i], received_neighbour_nodes[2 * i + 1]);
        missing_neighbour_nodes[i] = true;
        missing_received_neighbour_nodes[i] = false;

        for (size_t j = 0; j < n_nodes; ++j) {
            if (received_neighbour_node.almost_equal(nodes[j])) {
                missing_neighbour_nodes[i] = false;
                received_neighbour_nodes_indices[i] = j;
                break;
            }
        }

        if (missing_neighbour_nodes[i]) {
            for (size_t j = 0; j < n_received_nodes; ++j) {
                if (received_neighbour_node.almost_equal(Vec2<hostFloat>(received_nodes[2 * j], received_nodes[2 * j + 1]))) {
                    missing_neighbour_nodes[i] = false;
                    missing_received_neighbour_nodes[i] = true;
                    received_neighbour_node_received_indices[i] = j;
                    break;
                }
            }
        }

        if (missing_neighbour_nodes[i]) {
            for (size_t j = 0; j < i; ++j) {
                if (received_neighbour_node.almost_equal(Vec2<hostFloat>(received_neighbour_nodes[2 * j], received_neighbour_nodes[2 * j + 1]))) {
                    missing_neighbour_nodes[i] = false;
                    missing_received_neighbour_nodes[i] = true;
                    received_neighbour_node_received_indices[i] = n_received_nodes + j;
                    break;
                }
            }
        }
    }
}

auto SEM::Host::Meshes::add_new_received_neighbour_nodes(
        size_t n_nodes, 
        size_t n_received_created_nodes,
        Vec2<hostFloat>* nodes, 
        const hostFloat* received_neighbour_nodes, 
        const std::vector<bool>& missing_neighbour_nodes, 
        const std::vector<bool>& missing_nodes, 
        const std::vector<bool>& missing_received_neighbour_nodes, 
        size_t* received_neighbour_nodes_indices, 
        const size_t* received_neighbour_node_received_indices) -> void {

    for (size_t i = 0; i < missing_neighbour_nodes.size(); ++i) {
        if (missing_neighbour_nodes[i]) {
            size_t new_received_index = n_nodes + n_received_created_nodes;
            for (size_t j = 0; j < i; ++j) {
                new_received_index += missing_neighbour_nodes[j];
            }

            nodes[new_received_index] = Vec2<hostFloat>(received_neighbour_nodes[2 * i], received_neighbour_nodes[2 * i + 1]);
            received_neighbour_nodes_indices[i] = new_received_index;
        }
        else if (missing_received_neighbour_nodes[i]) {
            if (received_neighbour_node_received_indices[i] < missing_nodes.size()) {
                size_t new_received_index = n_nodes;
                for (size_t j = 0; j < received_neighbour_node_received_indices[i]; ++j) {
                    new_received_index += missing_nodes[j];
                }

                received_neighbour_nodes_indices[i] = new_received_index;
            }
            else {
                size_t new_received_index = n_nodes + n_received_created_nodes;
                for (size_t j = 0; j < received_neighbour_node_received_indices[i] - missing_nodes.size(); ++j) {
                    new_received_index += missing_neighbour_nodes[j];
                }

                received_neighbour_nodes_indices[i] = new_received_index;
            }
        }
    }
}

auto SEM::Host::Meshes::find_faces_to_delete(
        size_t n_domain_elements, 
        size_t n_elements_send_left, 
        size_t n_elements_send_right, 
        const Face2D_t* faces, 
        std::vector<bool>& faces_to_delete) -> void {
    
    for (size_t i = 0; i < faces_to_delete.size(); ++i) {
        const Face2D_t& face = faces[i];
        const std::array<bool, 2> elements_leaving {
            face.elements_[0] < n_elements_send_left || face.elements_[0] >= n_domain_elements - n_elements_send_right && face.elements_[0] < n_domain_elements,
            face.elements_[1] < n_elements_send_left || face.elements_[1] >= n_domain_elements - n_elements_send_right && face.elements_[1] < n_domain_elements
        };
        const std::array<bool, 2> boundary_elements {
            face.elements_[0] >= n_domain_elements,
            face.elements_[1] >= n_domain_elements
        };

        faces_to_delete[i] = elements_leaving[0] && elements_leaving[1] || elements_leaving[0] && boundary_elements[1] || elements_leaving[1] && boundary_elements[0];
    }
}

auto SEM::Host::Meshes::no_faces_to_delete(std::vector<bool>& faces_to_delete) -> void {
    for (size_t i = 0; i < faces_to_delete.size(); ++i) {
        faces_to_delete[i] = false;
    }
}

auto SEM::Host::Meshes::move_faces(
        size_t n_faces, 
        size_t n_domain_elements, 
        size_t new_n_domain_elements, 
        size_t n_elements_recv_left, 
        size_t n_elements_recv_right, 
        size_t n_mpi_destinations, 
        int rank, 
        Face2D_t* faces, 
        Face2D_t* new_faces, 
        const std::vector<bool>& boundary_elements_to_delete, 
        const size_t* mpi_interfaces_destination, 
        const int* mpi_interfaces_new_process_incoming, 
        const size_t* mpi_interfaces_new_local_index_incoming, 
        const size_t* mpi_interfaces_new_side_incoming) -> void {
    
    for (size_t i = 0; i < n_faces; ++i) {
        new_faces[i] = std::move(faces[i]);

        const size_t element_index_L = new_faces[i].elements_[0];
        if (element_index_L < n_domain_elements) {
            new_faces[i].elements_[0] += n_elements_recv_left;
        }
        else {
            const size_t boundary_element_index = element_index_L - n_domain_elements;
            if (!boundary_elements_to_delete[boundary_element_index]) { // Should always be the case
                size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
                for (size_t j = 0; j < boundary_element_index; ++j) {
                    new_boundary_element_index -= boundary_elements_to_delete[j];
                }

                new_faces[i].elements_[0] = new_boundary_element_index;
            }
            else {
                bool found_mpi_destination = false;
                size_t mpi_destination_index = static_cast<size_t>(-1);
                for (size_t j = 0; j < n_mpi_destinations; ++j) {
                    if (mpi_interfaces_destination[j] == element_index_L) {
                        mpi_destination_index = j;
                        found_mpi_destination = true;
                        break;
                    }
                }

                // CHECK why check for index? All elements should be received now if their new process is here
                if (found_mpi_destination && mpi_interfaces_new_process_incoming[mpi_destination_index] == rank && (mpi_interfaces_new_local_index_incoming[mpi_destination_index] < n_elements_recv_left || mpi_interfaces_new_local_index_incoming[mpi_destination_index] >= new_n_domain_elements - n_elements_recv_right)) {
                    new_faces[i].elements_[0] = mpi_interfaces_new_local_index_incoming[mpi_destination_index];
                    new_faces[i].elements_side_[0] = mpi_interfaces_new_side_incoming[mpi_destination_index];
                }
            }
        }

        const size_t element_index_R = new_faces[i].elements_[1];
        if (element_index_R < n_domain_elements) {
            new_faces[i].elements_[1] += n_elements_recv_left;
        }
        else {
            const size_t boundary_element_index = element_index_R - n_domain_elements;
            if (!boundary_elements_to_delete[boundary_element_index]) { // Should always be the case
                size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
                for (size_t j = 0; j < boundary_element_index; ++j) {
                    new_boundary_element_index -= boundary_elements_to_delete[j];
                }

                new_faces[i].elements_[1] = new_boundary_element_index;
            }
            else {
                bool found_mpi_destination = false;
                size_t mpi_destination_index = static_cast<size_t>(-1);
                for (size_t j = 0; j < n_mpi_destinations; ++j) {
                    if (mpi_interfaces_destination[j] == element_index_R) {
                        mpi_destination_index = j;
                        found_mpi_destination = true;
                        break;
                    }
                }

                // CHECK why check for index? All elements should be received now if their new process is here
                if (found_mpi_destination && mpi_interfaces_new_process_incoming[mpi_destination_index] == rank && (mpi_interfaces_new_local_index_incoming[mpi_destination_index] < n_elements_recv_left || mpi_interfaces_new_local_index_incoming[mpi_destination_index] >= new_n_domain_elements - n_elements_recv_right)) {
                    new_faces[i].elements_[1] = mpi_interfaces_new_local_index_incoming[mpi_destination_index];
                    new_faces[i].elements_side_[1] = mpi_interfaces_new_side_incoming[mpi_destination_index];
                }
            }
        }
    }
}

auto SEM::Host::Meshes::move_required_faces(
        size_t n_domain_elements, 
        size_t new_n_domain_elements, 
        size_t n_elements_send_left, 
        size_t n_elements_send_right, 
        size_t n_elements_recv_left, 
        size_t n_elements_recv_right, 
        size_t new_elements_offset_left, 
        size_t new_elements_offset_right, 
        size_t n_mpi_destinations, 
        int rank, 
        Face2D_t* faces, 
        Face2D_t* new_faces, 
        const std::vector<bool>& faces_to_delete, 
        const std::vector<bool>& boundary_elements_to_delete, 
        const size_t* elements_send_destinations_offset_left, 
        const size_t* elements_send_destinations_offset_right, 
        const int* elements_send_destinations_keep_left, 
        const int* elements_send_destinations_keep_right, 
        const size_t* mpi_interfaces_destination, 
        const int* mpi_interfaces_new_process_incoming, 
        const size_t* mpi_interfaces_new_local_index_incoming, 
        const size_t* mpi_interfaces_new_side_incoming) -> void {
    
    for (size_t i = 0; i < faces_to_delete.size(); ++i) {
        if (!faces_to_delete[i]) {
            size_t new_face_index = i;
            for (size_t j = 0; j < i; ++j) {
                new_face_index -= faces_to_delete[j];
            }

            new_faces[new_face_index] = std::move(faces[i]);

            const size_t element_index_L = new_faces[new_face_index].elements_[0];
            
            if (element_index_L < n_elements_send_left) {
                const size_t sent_element_index = element_index_L;
                size_t n_kept_before = 0;
                for (size_t j = 0; j < new_faces[new_face_index].elements_side_[0]; ++j) { // CHECK only works with quadrilaterals
                    n_kept_before += elements_send_destinations_keep_left[4 * sent_element_index + j];
                }
                const size_t new_element_index = new_elements_offset_left + elements_send_destinations_offset_left[sent_element_index] + n_kept_before;
                new_faces[new_face_index].elements_[0] = new_element_index;
                new_faces[new_face_index].elements_side_[0] = 0;
            }
            else if (element_index_L < n_domain_elements && element_index_L >= n_domain_elements - n_elements_send_right) {
                const size_t sent_element_index = element_index_L + n_elements_send_right - n_domain_elements;
                size_t n_kept_before = 0;
                for (size_t j = 0; j < new_faces[new_face_index].elements_side_[0]; ++j) { // CHECK only works with quadrilaterals
                    n_kept_before += elements_send_destinations_keep_right[4 * sent_element_index + j];
                }
                const size_t new_element_index = new_elements_offset_right + elements_send_destinations_offset_right[sent_element_index] + n_kept_before;
                new_faces[new_face_index].elements_[0] = new_element_index;
                new_faces[new_face_index].elements_side_[0] = 0;
            }
            else if (element_index_L < n_domain_elements) {
                new_faces[new_face_index].elements_[0] = new_faces[new_face_index].elements_[0] + n_elements_recv_left - n_elements_send_left;
            }
            else {
                const size_t boundary_element_index = element_index_L - n_domain_elements;
                if (!boundary_elements_to_delete[boundary_element_index]) { // Should always be the case
                    size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
                    for (size_t j = 0; j < boundary_element_index; ++j) {
                        new_boundary_element_index -= boundary_elements_to_delete[j];
                    }

                    new_faces[new_face_index].elements_[0] = new_boundary_element_index;
                }
                else {
                    bool found_mpi_destination = false;
                    size_t mpi_destination_index = static_cast<size_t>(-1);
                    for (size_t j = 0; j < n_mpi_destinations; ++j) {
                        if (mpi_interfaces_destination[j] == element_index_L) {
                            mpi_destination_index = j;
                            found_mpi_destination = true;
                            break;
                        }
                    }

                    // CHECK why check for index? All elements should be received now if their new process is here
                    if (found_mpi_destination && mpi_interfaces_new_process_incoming[mpi_destination_index] == rank && (mpi_interfaces_new_local_index_incoming[mpi_destination_index] < n_elements_recv_left || mpi_interfaces_new_local_index_incoming[mpi_destination_index] >= new_n_domain_elements - n_elements_recv_right)) {
                        new_faces[new_face_index].elements_[0] = mpi_interfaces_new_local_index_incoming[mpi_destination_index];
                        new_faces[new_face_index].elements_side_[0] = mpi_interfaces_new_side_incoming[mpi_destination_index];
                    }
                }
            }

            const size_t element_index_R = new_faces[new_face_index].elements_[1];
            if (element_index_R < n_elements_send_left) {
                const size_t sent_element_index = element_index_R;
                size_t n_kept_before = 0;
                for (size_t j = 0; j < new_faces[new_face_index].elements_side_[1]; ++j) { // CHECK only works with quadrilaterals
                    n_kept_before += elements_send_destinations_keep_left[4 * sent_element_index + j];
                }
                const size_t new_element_index = new_elements_offset_left + elements_send_destinations_offset_left[sent_element_index] + n_kept_before;
                new_faces[new_face_index].elements_[1] = new_element_index;
                new_faces[new_face_index].elements_side_[1] = 0;
            }
            else if (element_index_R < n_domain_elements && element_index_R >= n_domain_elements - n_elements_send_right) {
                const size_t sent_element_index = element_index_R + n_elements_send_right - n_domain_elements;
                size_t n_kept_before = 0;
                for (size_t j = 0; j < new_faces[new_face_index].elements_side_[1]; ++j) { // CHECK only works with quadrilaterals
                    n_kept_before += elements_send_destinations_keep_right[4 * sent_element_index + j];
                }
                const size_t new_element_index = new_elements_offset_right + elements_send_destinations_offset_right[sent_element_index] + n_kept_before;
                new_faces[new_face_index].elements_[1] = new_element_index;
                new_faces[new_face_index].elements_side_[1] = 0;
            }
            else if (element_index_R < n_domain_elements) {
                new_faces[new_face_index].elements_[1] = new_faces[new_face_index].elements_[1] + n_elements_recv_left - n_elements_send_left;
            }
            else {
                const size_t boundary_element_index = element_index_R - n_domain_elements;
                if (!boundary_elements_to_delete[boundary_element_index]) { // Should always be the case
                    size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
                    for (size_t j = 0; j < boundary_element_index; ++j) {
                        new_boundary_element_index -= boundary_elements_to_delete[j];
                    }

                    new_faces[new_face_index].elements_[1] = new_boundary_element_index;
                }
                else {
                    bool found_mpi_destination = false;
                    size_t mpi_destination_index = static_cast<size_t>(-1);
                    for (size_t j = 0; j < n_mpi_destinations; ++j) {
                        if (mpi_interfaces_destination[j] == element_index_R) {
                            mpi_destination_index = j;
                            found_mpi_destination = true;
                            break;
                        }
                    }

                    // CHECK why check for index? All elements should be received now if their new process is here
                    if (found_mpi_destination && mpi_interfaces_new_process_incoming[mpi_destination_index] == rank && (mpi_interfaces_new_local_index_incoming[mpi_destination_index] < n_elements_recv_left || mpi_interfaces_new_local_index_incoming[mpi_destination_index] >= new_n_domain_elements - n_elements_recv_right)) {
                        new_faces[new_face_index].elements_[1] = mpi_interfaces_new_local_index_incoming[mpi_destination_index];
                        new_faces[new_face_index].elements_side_[1] = mpi_interfaces_new_side_incoming[mpi_destination_index];
                    }
                }
            }
        }
    }
}

auto SEM::Host::Meshes::find_boundary_elements_to_delete(
        size_t n_boundary_elements, 
        size_t n_domain_elements, 
        size_t n_elements_send_left, 
        size_t n_elements_send_right, 
        const Element2D_t* elements, 
        const Face2D_t* faces, 
        std::vector<bool>& boundary_elements_to_delete, 
        const std::vector<bool>& faces_to_delete) -> void {
    
    for (size_t i = 0; i < boundary_elements_to_delete.size(); ++i) {
        const size_t element_index = i + n_domain_elements;
        const Element2D_t& element = elements[element_index];

        boundary_elements_to_delete[i] = true;

        for (size_t j = 0; j < element.faces_[0].size(); ++j) {
            const size_t face_index = element.faces_[0][j];
            if (!faces_to_delete[face_index]) {
                const Face2D_t& face = faces[face_index];
                const size_t side_index = face.elements_[0] == element_index;
                const size_t other_element_index = face.elements_[side_index];
                
                if (other_element_index >= n_elements_send_left && other_element_index < n_domain_elements - n_elements_send_right) {
                    boundary_elements_to_delete[i] = false;
                    break;
                }
            }
        }
    }
}

auto SEM::Host::Meshes::find_mpi_interface_elements_to_delete(
        size_t n_mpi_interface_elements, 
        size_t n_domain_elements, 
        int rank, 
        const size_t* mpi_interfaces_destination, 
        const int* mpi_interfaces_new_process_incoming, 
        std::vector<bool>& boundary_elements_to_delete) -> void {
    
    for (size_t i = 0; i < n_mpi_interface_elements; ++i) {
        if (mpi_interfaces_new_process_incoming[i] == rank) {
            boundary_elements_to_delete[mpi_interfaces_destination[i] - n_domain_elements] = true;
        }
    }
}

auto SEM::Host::Meshes::find_mpi_interface_elements_to_keep(
        size_t n_mpi_destinations, 
        size_t n_neighbours, 
        int rank, 
        size_t n_domain_elements, 
        const int* neighbour_procs, 
        const size_t* neighbour_indices, 
        const size_t* neighbour_sides, 
        const int* mpi_destination_procs, 
        const size_t* mpi_destination_local_indices, 
        const size_t* mpi_destination_sides, 
        const size_t* mpi_interfaces_destination, 
        std::vector<bool>& boundary_elements_to_delete) -> void {
    
    for (size_t i = 0; i < n_mpi_destinations; ++i) {
        const int mpi_proc = mpi_destination_procs[i];

        // The element is marked for deletion because either all linking elements are sent
        // Maybe we received an element that links to it, then we must keep it
        if (mpi_proc != rank && boundary_elements_to_delete[mpi_interfaces_destination[i] - n_domain_elements]) {
            const size_t mpi_index = mpi_destination_local_indices[i];
            const size_t mpi_side = mpi_destination_sides[i];
            for (size_t j = 0; j < n_neighbours; ++j) {
                const int neighbour_proc = neighbour_procs[j];
                const size_t neighbour_index = neighbour_indices[j];
                const size_t neighbour_side = neighbour_sides[j];

                if (mpi_proc == neighbour_proc && mpi_index == neighbour_index && mpi_side == neighbour_side) {
                    boundary_elements_to_delete[mpi_interfaces_destination[i] - n_domain_elements] = false; 
                    break;
                }
            }
        }
    }
}

auto SEM::Host::Meshes::move_boundary_elements(
        size_t n_domain_elements, 
        size_t new_n_domain_elements, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const std::vector<bool>& boundary_elements_to_delete, 
        const std::vector<bool>& faces_to_delete) -> void {
    
    for (size_t i = 0; i < boundary_elements_to_delete.size(); ++i) {
        if (!boundary_elements_to_delete[i]) {
            size_t new_boundary_element_index = new_n_domain_elements + i;
            for (size_t j = 0; j < i; ++j) {
                new_boundary_element_index -= boundary_elements_to_delete[j];
            }

            new_elements[new_boundary_element_index] = std::move(elements[i + n_domain_elements]);

            // We check for bad faces and update their indices
            size_t n_good_faces = 0;
            for (size_t j = 0; j < new_elements[new_boundary_element_index].faces_[0].size(); ++j) {
                const size_t face_index = new_elements[new_boundary_element_index].faces_[0][j];
                if (!faces_to_delete[face_index]) {
                    ++n_good_faces;

                    size_t new_face_index = face_index;
                    for (size_t k = 0; k < face_index; ++k) {
                        new_face_index -= faces_to_delete[k];
                    }
                    new_elements[new_boundary_element_index].faces_[0][j] = new_face_index;
                }
                else {
                    new_elements[new_boundary_element_index].faces_[0][j] = static_cast<size_t>(-1);
                }
            }
            if (n_good_faces != new_elements[new_boundary_element_index].faces_[0].size()) {
                std::vector<size_t> new_faces(n_good_faces);
                size_t faces_pos = 0;
                for (size_t j = 0; j < new_elements[new_boundary_element_index].faces_[0].size(); ++j) {
                    const size_t face_index = new_elements[new_boundary_element_index].faces_[0][j];
                    if (face_index != static_cast<size_t>(-1)) {
                        new_faces[faces_pos] = face_index;
                        ++faces_pos;
                    }
                }

                new_elements[new_boundary_element_index].faces_[0] = std::move(new_faces);
            }
        }
    }
}

auto SEM::Host::Meshes::find_boundaries_to_delete(
        const std::vector<size_t>& boundary, 
        size_t n_domain_elements, 
        const std::vector<bool>& boundary_elements_to_delete, 
        std::vector<bool>& boundaries_to_delete) -> void {
    
    for (size_t i = 0; i < boundary.size(); ++i) {
        const size_t boundary_element_index = boundary[i] - n_domain_elements;

        if (boundary_elements_to_delete[boundary_element_index]) {
            boundaries_to_delete[i] = true;
        }
        else {
            boundaries_to_delete[i] = false;
        }
    }
}

auto SEM::Host::Meshes::move_all_boundaries(
        size_t n_domain_elements, 
        size_t new_n_domain_elements, 
        const std::vector<size_t>& boundary, 
        std::vector<size_t>& new_boundary, 
        const std::vector<bool>& boundary_elements_to_delete) -> void {
    
    for (size_t i = 0; i < boundary.size(); ++i) {
        const size_t boundary_element_index = boundary[i] - n_domain_elements;

        size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
        for (size_t j = 0; j < boundary_element_index; ++j) {
            new_boundary_element_index -= boundary_elements_to_delete[j];
        }

        new_boundary[i] = new_boundary_element_index;
    }
}

auto SEM::Host::Meshes::move_required_boundaries(
        size_t n_domain_elements, 
        size_t new_n_domain_elements, 
        const std::vector<size_t>& boundary, 
        std::vector<size_t>& new_boundary, 
        const std::vector<bool>& boundaries_to_delete, 
        const std::vector<bool>& boundary_elements_to_delete) -> void {
    
    for (size_t i = 0; i < boundary.size(); ++i) {
        if (!boundaries_to_delete[i]) {
            size_t new_boundary_index = i;
            for (size_t j = 0; j < i; ++j) {
                new_boundary_index -= boundaries_to_delete[j];
            }

            const size_t boundary_element_index = boundary[i] - n_domain_elements;

            size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
            for (size_t j = 0; j < boundary_element_index; ++j) {
                new_boundary_element_index -= boundary_elements_to_delete[j];
            }

            new_boundary[new_boundary_index] = new_boundary_element_index;
        }
    }
}

auto SEM::Host::Meshes::move_required_boundaries(
        size_t n_domain_elements, 
        size_t new_n_domain_elements, 
        const std::vector<size_t>& boundary, 
        std::vector<size_t>& new_boundary, 
        std::vector<int>& new_boundary_process, 
        std::vector<size_t>& new_local_index, 
        std::vector<size_t>& new_side, 
        const std::vector<int>& mpi_interfaces_process, 
        const std::vector<size_t>& mpi_interfaces_local_index, 
        const std::vector<size_t>& mpi_interfaces_side, 
        const std::vector<bool>& boundaries_to_delete, 
        const std::vector<bool>& boundary_elements_to_delete) -> void {

    for (size_t i = 0; i < boundary.size(); ++i) {
        if (!boundaries_to_delete[i]) {
            size_t new_boundary_index = i;
            for (size_t j = 0; j < i; ++j) {
                new_boundary_index -= boundaries_to_delete[j];
            }

            const size_t boundary_element_index = boundary[i] - n_domain_elements;

            size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
            for (size_t j = 0; j < boundary_element_index; ++j) {
                new_boundary_element_index -= boundary_elements_to_delete[j];
            }

            new_boundary[new_boundary_index] = new_boundary_element_index;
            new_boundary_process[new_boundary_index] = mpi_interfaces_process[i];
            new_local_index[new_boundary_index] = mpi_interfaces_local_index[i];
            new_side[new_boundary_index] = mpi_interfaces_side[i];
        }
    }
}

auto SEM::Host::Meshes::create_sent_mpi_boundaries_destinations(
        size_t n_sent_elements, 
        size_t sent_elements_offset, 
        size_t new_elements_offset, 
        size_t mpi_interfaces_destination_offset, 
        Element2D_t* elements, 
        Element2D_t* new_elements, 
        const Vec2<hostFloat>* nodes, 
        const size_t* elements_send_destinations_offset, 
        const int* elements_send_destinations_keep, 
        const int* destination_process, 
        const size_t* destination_local_index, 
        size_t* mpi_interfaces_destination, 
        int* mpi_interfaces_destination_process, 
        size_t* mpi_interfaces_destination_local_index, 
        size_t* mpi_interfaces_destination_side, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<bool>& faces_to_delete) -> void {

    for (size_t i = 0; i < n_sent_elements; ++i) {
        const size_t element_index = sent_elements_offset + i;
        const Element2D_t& element = elements[element_index];

        size_t new_element_offset = elements_send_destinations_offset[i];

        for (size_t j = 0; j < 4; ++j) { // CHECK only works with quadrilaterals
            if (elements_send_destinations_keep[4 * i + j]) {
                const size_t new_element_index = new_elements_offset + new_element_offset;
                Element2D_t& new_element = new_elements[new_element_index];
                const size_t next_side_index = (j + 1 >= element.nodes_.size()) ? 0 : j + 1;

                mpi_interfaces_destination[mpi_interfaces_destination_offset + new_element_offset] = new_element_index;
                mpi_interfaces_destination_process[mpi_interfaces_destination_offset + new_element_offset] = destination_process[i];
                mpi_interfaces_destination_local_index[mpi_interfaces_destination_offset + new_element_offset] = destination_local_index[i];
                mpi_interfaces_destination_side[mpi_interfaces_destination_offset + new_element_offset] = j;

                new_element.N_ = element.N_;
                new_element.status_ = Hilbert::Status::A;
                new_element.rotation_ = 0;
                new_element.nodes_ = {element.nodes_[j], element.nodes_[next_side_index], element.nodes_[next_side_index], element.nodes_[j]};
                new_element.split_level_ = element.split_level_; // CHECK should set other variables?

                new_element.allocate_boundary_storage();
                new_element.faces_[0] = std::move(element.faces_[j]); // CHECK some faces could have been deleted, maybe check for -1?

                // We check for deleted faces and update their indices
                size_t n_good_faces = 0;
                for (size_t k = 0; k < new_element.faces_[0].size(); ++k) {
                    const size_t face_index = new_element.faces_[0][k];
                    if (!faces_to_delete[face_index]) {
                        ++n_good_faces;

                        size_t new_face_index = face_index;
                        for (size_t l = 0; l < face_index; ++l) {
                            new_face_index -= faces_to_delete[l];
                        }
                        new_element.faces_[0][k] = new_face_index;
                    }
                    else {
                        new_element.faces_[0][k] = static_cast<size_t>(-1);
                    }
                }
                if (n_good_faces != new_element.faces_[0].size()) {
                    std::vector<size_t> new_faces(n_good_faces);
                    size_t faces_pos = 0;
                    for (size_t k = 0; k < new_element.faces_[0].size(); ++k) {
                        const size_t face_index = new_element.faces_[0][k];
                        if (face_index != static_cast<size_t>(-1)) {
                            new_faces[faces_pos] = face_index;
                            ++faces_pos;
                        }
                    }

                    new_element.faces_[0] = std::move(new_faces);
                }

                // Geometry
                std::array<Vec2<hostFloat>, 4> points {nodes[new_element.nodes_[0]], 
                                                         nodes[new_element.nodes_[1]], 
                                                         nodes[new_element.nodes_[2]], 
                                                         nodes[new_element.nodes_[3]]};
                new_element.compute_boundary_geometry(points, polynomial_nodes[new_element.N_]);

                ++new_element_offset;
            }
        } 
    }
}

auto SEM::Host::Meshes::create_received_neighbours(
        size_t n_neighbours, 
        int rank, 
        size_t n_mpi_destinations, 
        size_t elements_offset, 
        size_t wall_offset, 
        size_t symmetry_offset, 
        size_t inflow_offset, 
        size_t outflow_offset, 
        size_t mpi_destinations_offset, 
        size_t n_new_wall, 
        size_t n_new_symmetry, 
        size_t n_new_inflow, 
        size_t n_new_outflow, 
        size_t n_new_mpi_destinations, 
        size_t n_domain_elements, 
        size_t new_n_domain_elements, 
        const size_t* neighbour_indices, 
        const int* neighbour_procs, 
        const size_t* neighbour_sides, 
        const int* neighbour_N, 
        const size_t* neighbour_node_indices, 
        const size_t* mpi_destinations_indices, 
        const int* mpi_destinations_procs, 
        const size_t* mpi_destinations_sides, 
        Element2D_t* elements, 
        const Vec2<hostFloat>* nodes, 
        const size_t* old_mpi_destinations, 
        size_t* neighbour_given_indices,
        size_t* neighbour_given_sides, 
        size_t* wall_boundaries, 
        size_t* symmetry_boundaries, 
        size_t* inflow_boundaries, 
        size_t* outflow_boundaries, 
        size_t* mpi_destinations, 
        int* mpi_destinations_process, 
        size_t* mpi_destinations_local_index, 
        size_t* mpi_destinations_side, 
        const std::vector<std::vector<hostFloat>>& polynomial_nodes, 
        const std::vector<bool>& boundary_elements_to_delete) -> void {
    
    for (size_t i = 0; i < n_neighbours; ++i) {
        const int neighbour_proc = neighbour_procs[i];

        switch (neighbour_proc) {
            case SEM::Host::Meshes::Mesh2D_t::boundary_type::wall :
                {
                    size_t n_additional_before = 0;
                    for (size_t j = 0; j < i; ++j) {
                        n_additional_before += (neighbour_procs[j] == SEM::Host::Meshes::Mesh2D_t::boundary_type::wall);
                    }

                    const size_t new_element_index = elements_offset + n_new_mpi_destinations + n_additional_before;
                    const size_t new_boundary_index = wall_offset + n_additional_before;

                    wall_boundaries[new_boundary_index] = new_element_index;
                    neighbour_given_indices[i] = new_element_index;
                    neighbour_given_sides[i] = 0;

                    Element2D_t& element = elements[new_element_index];
                    element.N_ = neighbour_N[i];
                    element.status_ = Hilbert::Status::A;
                    element.rotation_ = 0;
                    element.nodes_ = {neighbour_node_indices[2 * i], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i]};
                    element.split_level_ = 0; 
                    element.additional_nodes_[0] = 0; // CHECK should set other variables?

                    element.allocate_boundary_storage();

                    std::array<Vec2<hostFloat>, 4> points {nodes[element.nodes_[0]], 
                                                            nodes[element.nodes_[1]], 
                                                            nodes[element.nodes_[2]], 
                                                            nodes[element.nodes_[3]]};
                    element.compute_boundary_geometry(points, polynomial_nodes[element.N_]);
                }
                break;

            case SEM::Host::Meshes::Mesh2D_t::boundary_type::symmetry :
                {
                    size_t n_additional_before = 0;
                    for (size_t j = 0; j < i; ++j) {
                        n_additional_before += (neighbour_procs[j] == SEM::Host::Meshes::Mesh2D_t::boundary_type::symmetry);
                    }

                    const size_t new_element_index = elements_offset + n_new_mpi_destinations + n_new_wall + n_additional_before;
                    const size_t new_boundary_index = symmetry_offset + n_additional_before;
                    
                    symmetry_boundaries[new_boundary_index] = new_element_index;
                    neighbour_given_indices[i] = new_element_index;
                    neighbour_given_sides[i] = 0;

                    Element2D_t& element = elements[new_element_index];
                    element.N_ = neighbour_N[i];
                    element.status_ = Hilbert::Status::A;
                    element.rotation_ = 0;
                    element.nodes_ = {neighbour_node_indices[2 * i], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i]};
                    element.split_level_ = 0; 
                    element.additional_nodes_[0] = 0; // CHECK should set other variables?

                    element.allocate_boundary_storage();

                    std::array<Vec2<hostFloat>, 4> points {nodes[element.nodes_[0]], 
                                                            nodes[element.nodes_[1]], 
                                                            nodes[element.nodes_[2]], 
                                                            nodes[element.nodes_[3]]};
                    element.compute_boundary_geometry(points, polynomial_nodes[element.N_]);
                }
                break;

            case SEM::Host::Meshes::Mesh2D_t::boundary_type::inflow :
                {
                    size_t n_additional_before = 0;
                    for (size_t j = 0; j < i; ++j) {
                        n_additional_before += (neighbour_procs[j] == SEM::Host::Meshes::Mesh2D_t::boundary_type::inflow);
                    }

                    const size_t new_element_index = elements_offset + n_new_mpi_destinations + n_new_wall + n_new_symmetry + n_additional_before;
                    const size_t new_boundary_index = inflow_offset + n_additional_before;
                    
                    inflow_boundaries[new_boundary_index] = new_element_index;
                    neighbour_given_indices[i] = new_element_index;
                    neighbour_given_sides[i] = 0;

                    Element2D_t& element = elements[new_element_index];
                    element.N_ = neighbour_N[i];
                    element.status_ = Hilbert::Status::A;
                    element.rotation_ = 0;
                    element.nodes_ = {neighbour_node_indices[2 * i], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i]};
                    element.split_level_ = 0; 
                    element.additional_nodes_[0] = 0; // CHECK should set other variables?

                    element.allocate_boundary_storage();

                    std::array<Vec2<hostFloat>, 4> points {nodes[element.nodes_[0]], 
                                                            nodes[element.nodes_[1]], 
                                                            nodes[element.nodes_[2]], 
                                                            nodes[element.nodes_[3]]};
                    element.compute_boundary_geometry(points, polynomial_nodes[element.N_]);
                }
                break;

            case SEM::Host::Meshes::Mesh2D_t::boundary_type::outflow :
                {
                    size_t n_additional_before = 0;
                    for (size_t j = 0; j < i; ++j) {
                        n_additional_before += (neighbour_procs[j] == SEM::Host::Meshes::Mesh2D_t::boundary_type::outflow);
                    }
                    
                    const size_t new_element_index = elements_offset + n_new_mpi_destinations + n_new_wall + n_new_symmetry + n_new_inflow + n_additional_before;
                    const size_t new_boundary_index = outflow_offset + n_additional_before;
                
                    outflow_boundaries[new_boundary_index] = new_element_index;
                    neighbour_given_indices[i] = new_element_index;
                    neighbour_given_sides[i] = 0;

                    Element2D_t& element = elements[new_element_index];
                    element.N_ = neighbour_N[i];
                    element.status_ = Hilbert::Status::A;
                    element.rotation_ = 0;
                    element.nodes_ = {neighbour_node_indices[2 * i], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i]};
                    element.split_level_ = 0; 
                    element.additional_nodes_[0] = 0; // CHECK should set other variables?

                    element.allocate_boundary_storage();

                    std::array<Vec2<hostFloat>, 4> points {nodes[element.nodes_[0]], 
                                                            nodes[element.nodes_[1]], 
                                                            nodes[element.nodes_[2]], 
                                                            nodes[element.nodes_[3]]};
                    element.compute_boundary_geometry(points, polynomial_nodes[element.N_]);
                }
                break;

            default: // Not so simple!
                if (neighbour_proc >= 0) {
                    if (neighbour_proc != rank) {
                        // Check if creating or reusing
                        neighbour_given_sides[i] = 0;

                        const size_t local_element_index = neighbour_indices[i];
                        const size_t element_side_index = neighbour_sides[i];

                        bool first_time = true;
                        for (size_t j = 0; j < n_mpi_destinations; ++j) {
                            if (mpi_destinations_procs[j] == neighbour_proc && mpi_destinations_indices[j] == local_element_index && mpi_destinations_sides[j] == element_side_index) {
                                const size_t boundary_element_index = old_mpi_destinations[j] - n_domain_elements;
                                if (!boundary_elements_to_delete[boundary_element_index]) { // Should always be the case
                                   size_t new_boundary_element_index = new_n_domain_elements + boundary_element_index;
                                    for (size_t j = 0; j < boundary_element_index; ++j) {
                                        new_boundary_element_index -= boundary_elements_to_delete[j];
                                    }

                                    neighbour_given_indices[i] = new_boundary_element_index;
                                }
                                
                                first_time = false;
                                break;
                            }
                        }
                        if (first_time) {
                            for (size_t j = 0; j < i; ++j) {
                                if (neighbour_procs[j] == neighbour_proc && neighbour_indices[j] == local_element_index && neighbour_sides[j] == element_side_index) {
                                    size_t n_additional_before = 0;
                                    for (size_t k = 0; k < j; ++k) {
                                        if (neighbour_procs[k] >= 0 && neighbour_procs[k] != rank) {
                                            bool other_first_time = true;
                                            for (size_t l = 0; l < n_mpi_destinations; ++l) {
                                                if (mpi_destinations_procs[l] == neighbour_procs[k] && mpi_destinations_indices[l] == neighbour_indices[k] && mpi_destinations_sides[l] == neighbour_sides[k]) {
                                                    other_first_time = false;
                                                    break;
                                                }
                                            }
                                            if (other_first_time) {
                                                for (size_t l = 0; l < k; ++l) {
                                                    if (neighbour_procs[l] == neighbour_procs[k] && neighbour_indices[l] == neighbour_indices[k] && neighbour_sides[l] == neighbour_sides[k]) {
                                                        other_first_time = false;
                                                        break;
                                                    }
                                                }
                                            }
                                            if (other_first_time) {
                                                ++n_additional_before;
                                            }
                                        }
                                    }

                                    neighbour_given_indices[i] = elements_offset + n_additional_before;
                                    first_time = false;
                                    break;
                                }
                            }
                        }
                        if (first_time) {
                            size_t n_additional_before = 0;
                            for (size_t k = 0; k < i; ++k) {
                                if (neighbour_procs[k] >= 0 && neighbour_procs[k] != rank) {
                                    bool other_first_time = true;
                                    for (size_t l = 0; l < n_mpi_destinations; ++l) {
                                        if (mpi_destinations_procs[l] == neighbour_procs[k] && mpi_destinations_indices[l] == neighbour_indices[k] && mpi_destinations_sides[l] == neighbour_sides[k]) {
                                            other_first_time = false;
                                            break;
                                        }
                                    }
                                    if (other_first_time) {
                                        for (size_t l = 0; l < k; ++l) {
                                            if (neighbour_procs[l] == neighbour_procs[k] && neighbour_indices[l] == neighbour_indices[k] && neighbour_sides[l] == neighbour_sides[k]) {
                                                other_first_time = false;
                                                break;
                                            }
                                        }
                                    }
                                    if (other_first_time) {
                                        ++n_additional_before;
                                    }
                                }
                            }

                            const size_t new_element_index = elements_offset + n_additional_before;
                            const size_t new_mpi_destination_index = mpi_destinations_offset + n_additional_before;

                            mpi_destinations[new_mpi_destination_index] = new_element_index;
                            mpi_destinations_process[new_mpi_destination_index] = neighbour_proc;
                            mpi_destinations_local_index[new_mpi_destination_index] = local_element_index;
                            mpi_destinations_side[new_mpi_destination_index] = element_side_index;
                            neighbour_given_indices[i] = new_element_index;

                            Element2D_t& element = elements[new_element_index];
                            element.N_ = neighbour_N[i];
                            element.status_ = Hilbert::Status::A;
                            element.rotation_ = 0;
                            element.nodes_ = {neighbour_node_indices[2 * i], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i + 1], neighbour_node_indices[2 * i]};
                            element.split_level_ = 0; 
                            element.additional_nodes_[0] = 0; // CHECK should set other variables?

                            element.allocate_boundary_storage();
                            element.faces_[0] = std::vector<size_t>();

                            std::array<Vec2<hostFloat>, 4> points {nodes[element.nodes_[0]], 
                                                                     nodes[element.nodes_[1]], 
                                                                     nodes[element.nodes_[2]], 
                                                                     nodes[element.nodes_[3]]};
                            element.compute_boundary_geometry(points, polynomial_nodes[element.N_]);
                        }
                    }
                    else {
                        neighbour_given_indices[i] = neighbour_indices[i];
                        neighbour_given_sides[i] = neighbour_sides[i];
                    }
                }
        }
    }
}

auto SEM::Host::Meshes::add_new_faces_to_mpi_destination_elements(
        size_t n_mpi_destinations, 
        size_t face_offset, 
        size_t n_new_faces, 
        const size_t* mpi_destinations, 
        Element2D_t* elements, 
        const Face2D_t* faces) -> void {

    for (size_t i = 0; i < n_mpi_destinations; ++i) {
        const size_t element_index = mpi_destinations[i];
        Element2D_t& element = elements[element_index];

        size_t n_aditionnal_faces = 0;
        for (size_t j = 0; j < n_new_faces; ++j) {
            const size_t face_index = face_offset + j;
            const Face2D_t& face = faces[face_index];

            if (face.elements_[0] == element_index || face.elements_[1] == element_index) {
                ++n_aditionnal_faces;
            }
        }

        if (n_aditionnal_faces > 0) {
            std::vector<size_t> new_faces(element.faces_[0].size() + n_aditionnal_faces);
            for (size_t j = 0; j < element.faces_[0].size(); ++j) {
                new_faces[j] = element.faces_[0][j];
            }

            size_t added_face_index = 0;
            for (size_t j = 0; j < n_new_faces; ++j) {
                const size_t face_index = face_offset + j;
                const Face2D_t& face = faces[face_index];

                if (face.elements_[0] == element_index || face.elements_[1] == element_index) {
                    new_faces[element.faces_[0].size() + added_face_index] = face_index;
                    ++added_face_index;
                }
            }

            element.faces_[0] = std::move(new_faces);
        }
    }
}
