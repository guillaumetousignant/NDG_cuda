#include "helpers/InputParser_t.h"
#include "cgnslib.h"
#include <string>
#include <iostream>
#include <filesystem>
#include <array>
#include <vector>

// I think this was the most annoying thing to write yet.

namespace fs = std::filesystem;

constexpr int CGIO_MAX_NAME_LENGTH = 33; // Includes the null terminator

auto get_input_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string input_path = input_parser.getCmdOption("--in_path");
    if (!input_path.empty()) {
        return input_path;
    }
    else {
        const std::string filename_default("mesh.cgns");
        const std::string save_filename = input_parser.getCmdOptionOr("--in_filename", filename_default);
        const fs::path save_dir = input_parser.getCmdOptionOr("--in_directory", fs::current_path() / "meshes");

        return save_dir / save_filename;
    }
}

auto get_output_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string output_save_path = input_parser.getCmdOption("--out_path");
    if (!output_save_path.empty()) {
        const fs::path out_file = output_save_path;
        fs::create_directory(out_file.parent_path());
        return out_file;
    }
    else {
        const std::string filename_default("mesh_partitioned.cgns");
        const std::string save_filename = input_parser.getCmdOptionOr("--out_filename", filename_default);
        const fs::path save_dir = input_parser.getCmdOptionOr("--out_directory", fs::current_path() / "meshes"); 

        fs::create_directory(save_dir);
        return save_dir / save_filename;
    }
}

auto build_node_to_element(cgsize_t n_nodes, const std::vector<cgsize_t>& elements, cgsize_t n_elements_domain, cgsize_t n_elements_ghost) -> std::vector<std::vector<cgsize_t>> {
    std::vector<std::vector<cgsize_t>> node_to_element(n_nodes);

    for (cgsize_t j = 0; j < n_elements_domain; ++j) {
        for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
            const size_t node_index = elements[4 * j + side_index] - 1;
            node_to_element[node_index].push_back(j); // Doesn't work if elements have the same node multiple times. Shouldn't happen with correct meshes I think, otherwise do as in Mesh2D_t.
        }
    }
    for (cgsize_t j = 0; j < n_elements_ghost; ++j) {
        for (cgsize_t side_index = 0; side_index < 2; ++side_index) {
            const size_t node_index = elements[4 * n_elements_domain + 2 * j + side_index] - 1;
            node_to_element[node_index].push_back(n_elements_domain + j); // Doesn't work if elements have the same node multiple times. Shouldn't happen with correct meshes I think, otherwise do as in Mesh2D_t.
        }
    }

    return node_to_element;
}

auto build_element_to_element(const std::vector<cgsize_t>& elements, const std::vector<std::vector<cgsize_t>>& node_to_element, cgsize_t n_elements_domain, cgsize_t n_elements_ghost) -> std::vector<cgsize_t> {
    std::vector<cgsize_t> element_to_element(4 * n_elements_domain + n_elements_ghost);

    for (cgsize_t i = 0; i < n_elements_domain; ++i) {
        for (cgsize_t j = 0; j < 4; ++j) {
            const cgsize_t node_index = elements[4 * i + j];
            const cgsize_t node_index_next = (j < 3) ? elements[4 * i + j + 1] : elements[4 * i];

            for (auto element_index : node_to_element[node_index - 1]) {
                if (element_index != i) {
                    const auto iterator_begin = (element_index < n_elements_domain) ? elements.begin() + 4 * element_index : elements.begin() + 4 * n_elements_domain + 2 * (element_index - n_elements_domain);
                    const auto iterator_end = (element_index < n_elements_domain) ? elements.begin() + 4 * element_index + 4 : elements.begin() + 4 * n_elements_domain + 2 * (element_index - n_elements_domain) + 2;

                    auto it = std::find(iterator_begin, iterator_end, node_index_next);
                    if (it != iterator_end) {
                        element_to_element[4 * i + j] = element_index;
                        break;
                    }
                }
            }
        }
    }

    for (cgsize_t i = 0; i < n_elements_ghost; ++i) {
        const cgsize_t node_index = elements[4 * n_elements_domain + 2 * i];
        const cgsize_t node_index_next = elements[4 * n_elements_domain + 2 * i + 1];

        for (auto element_index : node_to_element[node_index - 1]) {
            if (element_index != i + n_elements_domain) {
                const auto iterator_begin = (element_index < n_elements_domain) ? elements.begin() + 4 * element_index : elements.begin() + 4 * n_elements_domain + 2 * (element_index - n_elements_domain);
                const auto iterator_end = (element_index < n_elements_domain) ? elements.begin() + 4 * element_index + 4 : elements.begin() + 4 * n_elements_domain + 2 * (element_index - n_elements_domain) + 2;

                auto it = std::find(iterator_begin, iterator_end, node_index_next);
                if (it != iterator_end) {
                    element_to_element[4 * n_elements_domain + i] = element_index;
                    break;
                }
            }
        }
    }
    
 
    return element_to_element;
}

auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);
    if (input_parser.cmdOptionExists("--help") || input_parser.cmdOptionExists("-h")) {
        std::cout << "Square unstructured mesh partitioner" << std::endl;
        std::cout << '\t' << "Generates multi-block 2D unstructured meshes from single-block meshes, both using the CGNS HDF5 format." << std::endl << std::endl;
        std::cout << "Available options:" << std::endl;
        std::cout << '\t' <<  "--in_path"       <<  '\t' <<  "Full path of the input mesh file. Overrides filename and directory if set." << std::endl;
        std::cout << '\t' <<  "--in_filename"   <<  '\t' <<  "File name of the input mesh file. Defaults to [mesh.cgns]" << std::endl;
        std::cout << '\t' <<  "--in_directory"  <<  '\t' <<  "Directory of the input mesh file. Defaults to [./meshes/]" << std::endl;
        std::cout << '\t' <<  "--out_path"      <<  '\t' <<  "Full path of the output mesh file. Overrides filename and directory if set." << std::endl;
        std::cout << '\t' <<  "--out_filename"  <<  '\t' <<  "File name of the output mesh file. Defaults to [mesh_partitioned.cgns]" << std::endl;
        std::cout << '\t' <<  "--out_directory" <<  '\t' <<  "Directory of the output mesh file. Defaults to [./meshes/]" << std::endl;
        std::cout << '\t' <<  "--n"             <<  '\t' <<  "Number of blocks in the output mesh. Defaults to [4]" << std::endl;
        exit(0);
    }

    // Argument parsing
    const fs::path in_file = get_input_file(input_parser);
    const fs::path out_file = get_output_file(input_parser);
    const cgsize_t n_proc = input_parser.getCmdOptionOr("--n", static_cast<cgsize_t>(4));

    // CGNS input
    int index_in_file = 0;
    const int open_error = cg_open(in_file.string().c_str(), CG_MODE_READ, &index_in_file);
    if (open_error != CG_OK) {
        std::cerr << "Error: input file '" << in_file << "' could not be opened with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(16);
    }

    // Getting base information
    int n_bases = 0;
    cg_nbases(index_in_file, &n_bases);
    if (n_bases != 1) {
        std::cerr << "Error: CGNS mesh has " << n_bases << " base(s), but for now only a single base is supported. Exiting." << std::endl;
        exit(17);
    }
    constexpr int index_in_base = 1;

    std::array<char, CGIO_MAX_NAME_LENGTH> base_name; // Oh yeah cause it's the 80s still
    int dim = 0;
    int physDim = 0;
    cg_base_read(index_in_file, index_in_base, base_name.data(), &dim, &physDim);
    if (dim != 2) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << " has " << dim << " dimensions, but the program only supports 2 dimensions. Exiting." << std::endl;
        exit(18);
    }
    if (physDim != 2) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << " has " << physDim << " physical dimensions, but the program only supports 2 physical dimensions. Exiting." << std::endl;
        exit(19);
    }

    // Getting zone information
    int n_zones = 0;
    cg_nzones(index_in_file, index_in_base, &n_zones);
    if (n_bases != 1) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << " has " << n_zones << " zone(s), but for now only a single zone is supported. Exiting." << std::endl;
        exit(20);
    }
    constexpr int index_in_zone = 1;

    ZoneType_t zone_type = ZoneType_t::ZoneTypeNull;
    cg_zone_type(index_in_file, index_in_base, index_in_zone, &zone_type);
    if (zone_type != ZoneType_t::Unstructured) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << " is not an unstructured zone. For now only unstructured zones are supported. Exiting." << std::endl;
        exit(21);
    }

    std::array<char, CGIO_MAX_NAME_LENGTH> zone_name; // Oh yeah cause it's the 80s still
    std::array<cgsize_t, 3> isize{0, 0, 0};
    cg_zone_read(index_in_file, index_in_base, index_in_zone, zone_name.data(), isize.data());
    if (isize[2] != 0) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << " has " << isize[2] << " boundary vertices, but to be honest I'm not sure how to deal with them. Exiting." << std::endl;
        exit(22);
    }
    const cgsize_t n_nodes = isize[0];
    const cgsize_t n_elements_tot = isize[1];

    // Getting nodes
    int n_coords = 0;
    cg_ncoords(index_in_file, index_in_base, index_in_zone, &n_coords);
    if (n_coords != 2) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << " has " << n_coords << " sets of coordinates, but for now only two are supported. Exiting." << std::endl;
        exit(23);
    }

    std::array<std::array<char, CGIO_MAX_NAME_LENGTH>, 2> coord_names; // Oh yeah cause it's the 80s still
    std::array<DataType_t, 2> coord_data_types {DataType_t::DataTypeNull, DataType_t::DataTypeNull};
    for (int index_coord = 1; index_coord <= n_coords; ++index_coord) {
        cg_coord_info(index_in_file, index_in_base, index_in_zone, index_coord, &coord_data_types[index_coord - 1], coord_names[index_coord - 1].data());
    }

    std::array<std::vector<double>, 2> xy{std::vector<double>(n_nodes), std::vector<double>(n_nodes)};

    for (int index_coord = 1; index_coord <= n_coords; ++index_coord) {
        const cgsize_t index_coord_start = 1;
        cg_coord_read(index_in_file, index_in_base, index_in_zone, coord_names[index_coord - 1].data(), DataType_t::RealDouble, &index_coord_start, &n_nodes, xy[index_coord - 1].data());
    }
    
    // Getting connectivity
    int n_sections = 0;
    cg_nsections(index_in_file, index_in_base, index_in_zone, &n_sections);

    std::vector<cgsize_t> section_data_size(n_sections);
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> section_names(n_sections); // Oh yeah cause it's the 80s still
    std::vector<ElementType_t> section_type(n_sections);
    std::vector<std::array<cgsize_t, 2>> section_ranges(n_sections);
    std::vector<int> section_n_boundaries(n_sections);
    std::vector<int> section_parent_flags(n_sections);
    for (int index_section = 1; index_section <= n_sections; ++index_section) {
        cg_ElementDataSize(index_in_file, index_in_base, index_in_zone, index_section, &section_data_size[index_section - 1]);
        cg_section_read(index_in_file, index_in_base, index_in_zone, index_section, section_names[index_section - 1].data(), &section_type[index_section - 1], &section_ranges[index_section - 1][0], &section_ranges[index_section - 1][1], &section_n_boundaries[index_section - 1], &section_parent_flags[index_section - 1]);
        if (section_n_boundaries[index_section - 1] != 0) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", section " << index_section << " has " << section_n_boundaries[index_section - 1] << " boundary elements, but to be honest I'm not sure how to deal with them. Exiting." << std::endl;
            exit(24);
        }
    }

    std::vector<std::vector<cgsize_t>> connectivity(n_sections);
    std::vector<std::vector<cgsize_t>> parent_data(n_sections);
    for (int index_section = 1; index_section <= n_sections; ++index_section) {
        connectivity[index_section - 1] = std::vector<cgsize_t>(section_data_size[index_section - 1]);
        parent_data[index_section - 1] = std::vector<cgsize_t>(section_ranges[index_section - 1][1] - section_ranges[index_section - 1][0] + 1);

        cg_elements_read(index_in_file, index_in_base, index_in_zone, index_section, connectivity[index_section - 1].data(), parent_data[index_section - 1].data());
    }

    // Interfaces
    int n_connectivity = 0;
    cg_nconns(index_in_file, index_in_base, index_in_zone, &n_connectivity);

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
        cg_conn_info(index_in_file, index_in_base, index_in_zone, index_connectivity, connectivity_names[index_connectivity - 1].data(),
            &connectivity_grid_locations[index_connectivity - 1], &connectivity_types[index_connectivity - 1],
            &connectivity_point_set_types[index_connectivity - 1], &connectivity_sizes[index_connectivity - 1], connectivity_donor_names[index_connectivity - 1].data(),
            &connectivity_donor_zone_types[index_connectivity - 1], &connectivity_donor_point_set_types[index_connectivity - 1],
            &connectivity_donor_data_types[index_connectivity - 1], &connectivity_donor_sizes[index_connectivity - 1]);

        if (connectivity_donor_zone_types[index_connectivity - 1] != ZoneType_t::Unstructured) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " has a donor zone type that isn't unstructured. For now only unstructured zones are supported. Exiting." << std::endl;
            exit(25);
        }
        if (connectivity_point_set_types[index_connectivity - 1] != PointSetType_t::PointList) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " has a point set type that isn't PointList. For now only PointList point set types are supported. Exiting." << std::endl;
            exit(26);
        }
        if (connectivity_donor_point_set_types[index_connectivity - 1] != PointSetType_t::PointListDonor) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " has a donor point set type that isn't PointListDonor. For now only PointListDonor point set types are supported. Exiting." << std::endl;
            exit(27);
        }

        if (connectivity_grid_locations[index_connectivity - 1] != GridLocation_t::FaceCenter) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " has a grid location that isn't FaceCenter. For now only FaceCenter grid locations are supported. Exiting." << std::endl;
            exit(28);
        }

        if (connectivity_types[index_connectivity - 1] != GridConnectivityType_t::Abutting1to1) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " has a grid connectivity type that isn't Abutting1to1. For now only Abutting1to1 grid connectivity types are supported. Exiting." << std::endl;
            exit(29);
        }

        if (connectivity_sizes[index_connectivity - 1] != connectivity_donor_sizes[index_connectivity - 1]) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " has a different number of elements in the origin and destination zones. Exiting." << std::endl;
            exit(30);
        }
    }

    std::vector<std::vector<cgsize_t>> interface_elements(n_connectivity);
    std::vector<std::vector<cgsize_t>> interface_donor_elements(n_connectivity);
    for (int index_connectivity = 1; index_connectivity <= n_connectivity; ++index_connectivity) {
        interface_elements[index_connectivity - 1] = std::vector<cgsize_t>(connectivity_sizes[index_connectivity - 1]);
        interface_donor_elements[index_connectivity - 1] = std::vector<cgsize_t>(connectivity_donor_sizes[index_connectivity - 1]);
        cg_conn_read(index_in_file, index_in_base, index_in_zone, index_connectivity, interface_elements[index_connectivity - 1].data(),
            DataType_t::Integer, interface_donor_elements[index_connectivity - 1].data());
    }

    // Boundary conditions
    int n_boundaries = 0;
    cg_nbocos(index_in_file, index_in_base, index_in_zone, &n_boundaries);

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
        cg_boco_info(index_in_file, index_in_base, index_in_zone, index_boundary, boundary_names[index_boundary - 1].data(),
            &boundary_types[index_boundary - 1], &boundary_point_set_types[index_boundary - 1], &boundary_sizes[index_boundary - 1],
            &boundary_normal_indices[index_boundary - 1], &boundary_normal_list_sizes[index_boundary - 1],
            &boundary_normal_data_types[index_boundary - 1], &boundary_n_datasets[index_boundary - 1]);

        if (boundary_point_set_types[index_boundary - 1] != PointSetType_t::PointList) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", boundary " << index_boundary << " has a point set type that isn't PointList. For now only PointList point set types are supported. Exiting." << std::endl;
            exit(31);
        }

        cg_boco_gridlocation_read(index_in_file, index_in_base, index_in_zone, index_boundary, &boundary_grid_locations[index_boundary - 1]);

        if (boundary_grid_locations[index_boundary - 1] != GridLocation_t::EdgeCenter) {
            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", boundary " << index_boundary << " has a grid location that isn't EdgeCenter. For now only EdgeCenter grid locations are supported. Exiting." << std::endl;
            exit(32);
        }
    }

    std::vector<std::vector<cgsize_t>> boundary_elements(n_boundaries);
    std::vector<std::vector<cgsize_t>> boundary_normals(n_boundaries);
    for (int index_boundary = 1; index_boundary <= n_boundaries; ++index_boundary) {
        boundary_elements[index_boundary - 1] = std::vector<cgsize_t>(boundary_sizes[index_boundary - 1]);
        boundary_normals[index_boundary - 1] = std::vector<cgsize_t>(boundary_normal_list_sizes[index_boundary - 1]);
        cg_boco_read(index_in_file, index_in_base, index_in_zone, index_boundary, boundary_elements[index_boundary - 1].data(), boundary_normals[index_boundary - 1].data());
    }

    const int close_error = cg_close(index_in_file);
    if (close_error != CG_OK) {
        std::cerr << "Error: input file '" << in_file << "' could not be closed with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(41);
    }

    // Partitioning
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
                std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", section " << i << " has an unknown element type. For now only BAR_2 and QUAD_4 are implemented, for boundaries and domain respectively. Exiting." << std::endl;
                exit(33);
        }
    }

    if (n_elements_domain + n_elements_ghost != n_elements_tot) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << " has " << n_elements_tot << " elements but the sum of its sections is " << n_elements_domain + n_elements_ghost << " elements. Exiting." << std::endl;
        exit(34);  
    }

    // Splitting elements
    const auto [n_elements_div, n_elements_mod] = std::div(n_elements_domain, n_proc);
    std::vector<cgsize_t> n_elements(n_proc);
    std::vector<cgsize_t> starting_elements(n_proc);
    for (cgsize_t i = 0; i < n_proc; ++i) {
        starting_elements[i] = i * n_elements_div + std::min(i, n_elements_mod);
        const cgsize_t ending_element = (i + 1) * n_elements_div + std::min(i + 1, n_elements_mod);
        n_elements[i] = ending_element - starting_elements[i];
    }

    // Putting connectivity data together
    std::vector<cgsize_t> elements(4 * n_elements_domain + 2 * n_elements_ghost);
    std::vector<cgsize_t> section_start_indices(n_sections);
    cgsize_t element_domain_index = 0;
    cgsize_t element_ghost_index = 4 * n_elements_domain;
    for (int i = 0; i < n_sections; ++i) {
        if (section_is_domain[i]) {
            section_start_indices[i] = element_domain_index;
            for (cgsize_t j = 0; j < section_ranges[i][1] - section_ranges[i][0] + 1; ++j) {
                elements[section_start_indices[i] + 4 * j] = connectivity[i][4 * j];
                elements[section_start_indices[i] + 4 * j + 1] = connectivity[i][4 * j + 1];
                elements[section_start_indices[i] + 4 * j + 2] = connectivity[i][4 * j + 2];
                elements[section_start_indices[i] + 4 * j + 3] = connectivity[i][4 * j + 3];
            }
            element_domain_index += 4 * (section_ranges[i][1] - section_ranges[i][0] + 1);
        }
        else {
            section_start_indices[i] = element_ghost_index;
            for (cgsize_t j = 0; j < section_ranges[i][1] - section_ranges[i][0] + 1; ++j) {
                elements[section_start_indices[i] + 2 * j] = connectivity[i][2 * j];
                elements[section_start_indices[i] + 2 * j + 1] = connectivity[i][2 * j + 1];
            }
            element_ghost_index += 2 * (section_ranges[i][1] - section_ranges[i][0] + 1);
        }
    }

    // Computing nodes to elements
    const std::vector<std::vector<cgsize_t>> node_to_element = build_node_to_element(n_nodes, elements, n_elements_domain, n_elements_ghost);

    // Computing element to elements
    const std::vector<cgsize_t> element_to_element = build_element_to_element(elements, node_to_element, n_elements_domain, n_elements_ghost);

    // Creating output file
    int index_out_file = 0;
    const int open_out_error = cg_open(out_file.string().c_str(), CG_MODE_WRITE, &index_out_file);
    if (open_out_error != CG_OK) {
        std::cerr << "Error: output file '" << out_file << "' could not be opened with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(40);
    }

    int index_out_base = 0;
    cg_base_write(index_out_file, base_name.data(), dim, physDim, &index_out_base);

    // Building the different sections
    std::vector<int> index_out_zone(n_proc);
    std::vector<std::vector<std::vector<std::array<cgsize_t, 2>>>> cotton_eyed_joe(n_proc); // [Where did he come from; where did he go]
    std::vector<std::vector<std::vector<std::array<cgsize_t, 2>>>> origin_and_destination_ghosts(n_proc); // [origin; destination]
    const size_t n_digits = std::to_string(n_proc).length();

    for (cgsize_t i = 0; i < n_proc; ++i) {
        cgsize_t n_elements_in_proc = n_elements[i];

        // Getting section elements
        std::vector<cgsize_t> elements_in_proc(4 * n_elements[i]);
        for (cgsize_t element_index = 0; element_index < n_elements[i]; ++element_index) {
            for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                elements_in_proc[4 * element_index + side_index] = elements[4 * (element_index + starting_elements[i]) + side_index];
            }
        }

        // Getting relevant zones
        std::vector<std::vector<cgsize_t>> new_boundary_indices(n_sections);
        for (int k = 0; k < n_sections; ++k) {
            if (!section_is_domain[k]) {
                new_boundary_indices[k] = std::vector<cgsize_t>(section_ranges[k][1] - section_ranges[k][0] + 1);
                for (cgsize_t j = 0; j < section_ranges[k][1] - section_ranges[k][0] + 1; ++j) {
                    cgsize_t ghost_index = j;
                    for (cgsize_t zone_loop = 0; zone_loop < k; ++zone_loop) {
                        if (!section_is_domain[zone_loop]) {
                            ghost_index += section_ranges[k][1] - section_ranges[k][0] + 1;
                        }
                    }

                    const cgsize_t domain_element_index = element_to_element[4 * n_elements_domain + ghost_index];
                    if (domain_element_index >= starting_elements[i] && domain_element_index < starting_elements[i] + n_elements[i]) {
                        new_boundary_indices[k][j] = 1;
                    }
                }
            }
        }
        std::vector<cgsize_t> n_boundaries_in_proc(n_sections);
        std::vector<cgsize_t> boundaries_start_index(n_sections);
        cgsize_t n_total_boundaries_in_proc = 0;
        for (int k = 0; k < n_sections; ++k) {
            if (!section_is_domain[k]) {
                boundaries_start_index[k] = n_elements[i] + n_total_boundaries_in_proc;
                for (cgsize_t j = 0; j < section_ranges[k][1] - section_ranges[k][0] + 1; ++j) {
                    if (new_boundary_indices[k][j]) {
                        new_boundary_indices[k][j] = n_elements[i] + n_total_boundaries_in_proc;
                        ++n_boundaries_in_proc[k];
                        ++n_total_boundaries_in_proc;
                    }
                }
            }
        }
        std::vector<std::vector<cgsize_t>> boundaries_in_proc(n_sections);
        for (int k = 0; k < n_sections; ++k) {
            if (!section_is_domain[k]) {
                if (n_boundaries_in_proc[k] > 0) {
                    boundaries_in_proc[k].reserve(n_boundaries_in_proc[k] * 2);
                    n_elements_in_proc += n_boundaries_in_proc[k];

                    for (cgsize_t j = 0; j < section_ranges[k][1] - section_ranges[k][0] + 1; ++j) {
                        if (new_boundary_indices[k][j]) {
                            const cgsize_t ghost_index = 2 * j + section_start_indices[k];
                            boundaries_in_proc[k].push_back(elements[ghost_index]);
                            boundaries_in_proc[k].push_back(elements[ghost_index + 1]);
                        }
                    }
                }
            }
        }

        // Adding elements for connectivity
        cotton_eyed_joe[i] = std::vector<std::vector<std::array<cgsize_t, 2>>>(n_proc); // [Where did he come from; where did he go]
        origin_and_destination_ghosts[i] = std::vector<std::vector<std::array<cgsize_t, 2>>>(n_proc); // [origin; destination]
        std::vector<cgsize_t> connectivity_elements;
        cgsize_t n_connectivity_elements = 0;
        for (cgsize_t j = 0; j < n_elements[i]; ++j) {
            const cgsize_t element_index = j + starting_elements[i];
            for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                const cgsize_t neighbour_element_index = element_to_element[4 * element_index + side_index];
                if (neighbour_element_index < n_elements_domain && !(neighbour_element_index >= starting_elements[i] && neighbour_element_index < starting_elements[i] + n_elements[i])) {
                    cgsize_t destination_proc = static_cast<cgsize_t>(-1);
                    cgsize_t element_index_in_destination = static_cast<cgsize_t>(-1);
                    for (cgsize_t process_index = 0; process_index < n_proc; ++process_index) {
                        if (neighbour_element_index >= starting_elements[process_index] && neighbour_element_index < starting_elements[process_index] + n_elements[process_index]) {
                            destination_proc = process_index;
                            element_index_in_destination = neighbour_element_index - starting_elements[process_index];
                            break;
                        }
                    }

                    if (destination_proc == static_cast<cgsize_t>(-1) ) {
                        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << " contains neighbour element " << neighbour_element_index << " but it is not found in any process. Exiting." << std::endl;
                        exit(60);
                    }

                    cotton_eyed_joe[i][destination_proc].push_back({j, element_index_in_destination});
                    origin_and_destination_ghosts[i][destination_proc].push_back({n_elements_in_proc + n_connectivity_elements, static_cast<cgsize_t>(-1)});

                    connectivity_elements.push_back((side_index < 3) ? elements_in_proc[4 * j + side_index + 1] : elements_in_proc[4 * j]);
                    connectivity_elements.push_back(elements_in_proc[4 * j + side_index]);
                    ++n_connectivity_elements;
                }
            }
        }
        n_elements_in_proc += n_connectivity_elements;

        // Getting relevant points
        std::vector<bool> is_point_needed(n_nodes);
        for (cgsize_t element_index = 0; element_index < n_elements[i]; ++element_index) {
            for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                is_point_needed[elements_in_proc[4 * element_index + side_index] - 1] = true;
            }
        }
        for (int k = 0; k < n_sections; ++k) {
            for (cgsize_t j = 0; j < n_boundaries_in_proc[k]; ++j) {
                is_point_needed[boundaries_in_proc[k][2 * j] - 1] = true;
                is_point_needed[boundaries_in_proc[k][2 * j + 1] - 1] = true;
            }
        }
        for (cgsize_t connectivity_index = 0; connectivity_index < n_connectivity_elements; ++connectivity_index) {
            is_point_needed[connectivity_elements[2 * connectivity_index] - 1] = true;
            is_point_needed[connectivity_elements[2 * connectivity_index + 1] - 1] = true;
        }

        cgsize_t n_nodes_in_proc = 0;
        for (auto needed : is_point_needed) {
            n_nodes_in_proc += needed;
        }

        std::array<std::vector<double>, 2> xy_in_proc{std::vector<double>(), std::vector<double>()};
        xy_in_proc[0].reserve(n_nodes_in_proc);
        xy_in_proc[1].reserve(n_nodes_in_proc);

        for (cgsize_t node_index = 0; node_index < n_nodes; ++node_index) {
            if (is_point_needed[node_index]) {
                xy_in_proc[0].push_back(xy[0][node_index]);
                xy_in_proc[1].push_back(xy[1][node_index]);

                // Replacing point indices CHECK do with node_to_elem
                for (auto element_index : node_to_element[node_index]) {
                    if (element_index < n_elements_domain ) {
                        if (element_index >= starting_elements[i] && element_index < starting_elements[i] + n_elements[i]) {
                            for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                                if (elements_in_proc[4 * (element_index - starting_elements[i]) + side_index] == node_index + 1) {
                                    elements_in_proc[4 * (element_index - starting_elements[i]) + side_index] = xy_in_proc[0].size();
                                }
                            }
                        }
                    }
                    else {
                        int section_index = -1;
                        for (int k = 0; k < n_sections; ++k) {
                            if (element_index + 1 >= section_ranges[k][0] && element_index + 1 <= section_ranges[k][1]) {
                                section_index = k;
                                break;
                            }
                        }
                        if (section_index == -1) {
                            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", node to elements of point " << node_index << " contains element " << element_index << " but it is not found in any mesh section. Exiting." << std::endl;
                            exit(52);
                        }

                        const cgsize_t element_section_index = element_index + 1 - section_ranges[section_index][0];
                        const cgsize_t new_element_index = new_boundary_indices[section_index][element_section_index];
                        if (new_element_index != 0) {
                            const cgsize_t boundary_element_index = new_element_index - boundaries_start_index[section_index];

                            if (boundaries_in_proc[section_index][2 * boundary_element_index] == node_index + 1) {
                                boundaries_in_proc[section_index][2 * boundary_element_index] = xy_in_proc[0].size();
                            }
                            if (boundaries_in_proc[section_index][2 * boundary_element_index + 1] == node_index + 1) {
                                boundaries_in_proc[section_index][2 * boundary_element_index + 1] = xy_in_proc[0].size();
                            }
                        }
                    }
                }

                // Nothing to do fot this one I'm afraid.
                for (cgsize_t connectivity_index = 0; connectivity_index < 2 * n_connectivity_elements; ++connectivity_index) {
                    if (connectivity_elements[connectivity_index] == node_index + 1) {
                        connectivity_elements[connectivity_index] = xy_in_proc[0].size();
                    }
                }
            }
        }
        
        // Writing zone information to file
        /* vertex size, cell size, boundary vertex size (always zero for structured grids) */
        std::array<cgsize_t, 3> isize {n_nodes_in_proc,
                                       n_elements_in_proc,
                                       0};

        std::stringstream ss;
        ss << "Zone " << std::setfill('0') << std::setw(n_digits) << i + 1;
        cg_zone_write(index_out_file, index_out_base, ss.str().c_str(), isize.data(), zone_type, &index_out_zone[i]);

        /* write grid coordinates (user must use SIDS-standard names here) */
        int index_out_coord = 0;
        cg_coord_write(index_out_file, index_out_base, index_out_zone[i], DataType_t::RealDouble, coord_names[0].data(), xy_in_proc[0].data(), &index_out_coord);
        cg_coord_write(index_out_file, index_out_base, index_out_zone[i], DataType_t::RealDouble, coord_names[1].data(), xy_in_proc[1].data(), &index_out_coord);

        // Writing domain sections to file
        cgsize_t section_start = 0;
        cgsize_t section_end = 0;
        cgsize_t element_index = starting_elements[i];
        cgsize_t remaining_elements = n_elements[i];
        cgsize_t domain_index_start = 0;
        cgsize_t domain_index_end = 0;
        for (int k = 0; k < n_sections; ++k) {
            if (section_is_domain[k]) {
                section_end += section_ranges[k][1] - section_ranges[k][0] + 1;

                if (element_index >= section_start && element_index < section_end) {
                    const cgsize_t section_offset = element_index - section_start;
                    const cgsize_t n_elements_in_this_section = section_ranges[k][1] - section_ranges[k][0] + 1;
                    const cgsize_t section_end_offset = (n_elements_in_this_section - section_offset > remaining_elements) ? n_elements_in_this_section - section_offset - remaining_elements: 0;
                    const cgsize_t n_elements_from_this_section = n_elements_in_this_section - section_offset - section_end_offset;

                    int index_out_section = 0;
                    domain_index_end += n_elements_from_this_section;
                    const cgsize_t n_boundary_elem = 0; // No boundary elements yet
                    cg_section_write(index_out_file, index_out_base, index_out_zone[i], section_names[k].data(), ElementType_t::QUAD_4, domain_index_start + 1, domain_index_end, n_boundary_elem, elements_in_proc.data() + element_index - starting_elements[i], &index_out_section);
                    domain_index_start = domain_index_end;

                    element_index += n_elements_from_this_section;
                    remaining_elements -= n_elements_from_this_section;
                    if (remaining_elements <= 0) {
                        break;
                    }
                }

                section_start = section_end;
            }
        }

        // Writing ghost sections to file
        cgsize_t boundary_index_start = n_elements[i];
        cgsize_t boundary_index_end = n_elements[i];
        for (int k = 0; k < n_sections; ++k) {
            if (!section_is_domain[k]) {
                if (n_boundaries_in_proc[k] > 0) {
                    boundary_index_end += n_boundaries_in_proc[k];
                    int index_out_section = 0;
                    const cgsize_t n_boundary_elem = 0; // No boundary elements yet
                    cg_section_write(index_out_file, index_out_base, index_out_zone[i], section_names[k].data(), ElementType_t::BAR_2, boundary_index_start + 1, boundary_index_end, n_boundary_elem, boundaries_in_proc[k].data(), &index_out_section);
                    boundary_index_start = boundary_index_end;
                }
            }
        }

        // Writing connectivity elements to file
        if (n_connectivity_elements > 0) {
            const std::string connectivity_elements_name("ConnectivityElements");
            const cgsize_t connectivity_index_start = boundary_index_end;
            const cgsize_t connectivity_index_end = connectivity_index_start + n_connectivity_elements;
            int connectivity_out_section = 0;
            const cgsize_t n_boundary_elem = 0; // No boundary elements yet
            cg_section_write(index_out_file, index_out_base, index_out_zone[i], connectivity_elements_name.c_str(), ElementType_t::BAR_2, connectivity_index_start + 1, connectivity_index_end, n_boundary_elem, connectivity_elements.data(), &connectivity_out_section);
        }

        // Finding which elements are boundary condition elements
        for (int index_boundary = 0; index_boundary < n_boundaries; ++index_boundary) {
            cgsize_t n_boundary_elements_in_proc = 0;
            for (cgsize_t j = 0; j < boundary_sizes[index_boundary]; ++j) {
                int section_index = -1;
                for (int k = 0; k < n_sections; ++k) {
                    if ((boundary_elements[index_boundary][j] >= section_ranges[k][0]) && (boundary_elements[index_boundary][j] <= section_ranges[k][1])) {
                        section_index = k;
                        break;
                    }
                }
                if (section_index == -1) {
                    std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", boundary " << index_boundary << " contains element " << boundary_elements[index_boundary][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(45);
                }

                const cgsize_t new_element_index = new_boundary_indices[section_index][boundary_elements[index_boundary][j] - section_ranges[section_index][0]];
                if (new_element_index != 0) {
                    ++n_boundary_elements_in_proc;
                }
            }

            if (n_boundary_elements_in_proc > 0) {
                std::vector<cgsize_t> boundary_conditions;
                boundary_conditions.reserve(n_boundary_elements_in_proc);

                for (cgsize_t j = 0; j < boundary_sizes[index_boundary]; ++j) {
                    int section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((boundary_elements[index_boundary][j] >= section_ranges[k][0]) && (boundary_elements[index_boundary][j] <= section_ranges[k][1])) {
                            section_index = k;
                            break;
                        }
                    }
                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", boundary " << index_boundary << " contains element " << boundary_elements[index_boundary][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(45);
                    }

                    const cgsize_t new_element_index = new_boundary_indices[section_index][boundary_elements[index_boundary][j] - section_ranges[section_index][0]];
                    if (new_element_index != 0) {
                        boundary_conditions.push_back(new_element_index + 1);
                    }
                }

                int index_out_boundary = 0;
                cg_boco_write(index_out_file, index_out_base, index_out_zone[i], boundary_names[index_boundary].data(), boundary_types[index_boundary], boundary_point_set_types[index_boundary], n_boundary_elements_in_proc, boundary_conditions.data(), &index_out_boundary);
                cg_boco_gridlocation_write(index_out_file, index_out_base, index_out_zone[i], index_out_boundary, boundary_grid_locations[index_boundary]);
            }
        }

        // Self interfaces
        for (int index_connectivity = 0; index_connectivity < n_connectivity; ++index_connectivity) {
            cgsize_t n_connectivity_elements_in_proc = 0;
            for (cgsize_t j = 0; j < connectivity_sizes[index_connectivity]; ++j) {
                const cgsize_t element_index = interface_elements[index_connectivity][j];
                const cgsize_t element_donor_index = interface_donor_elements[index_connectivity][j];
                
                int section_index = -1;
                for (int k = 0; k < n_sections; ++k) {
                    if ((element_index >= section_ranges[k][0]) && (element_index <= section_ranges[k][1])) {
                        section_index = k;
                        break;
                    }
                }
                if (section_index == -1) {
                    std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " contains element " << element_index << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(46);
                }

                const cgsize_t new_element_index = new_boundary_indices[section_index][element_index - section_ranges[section_index][0]];

                section_index = -1;
                for (int k = 0; k < n_sections; ++k) {
                    if ((element_donor_index >= section_ranges[k][0]) && (element_donor_index <= section_ranges[k][1])) {
                        section_index = k;
                        break;
                    }
                }
                if (section_index == -1) {
                    std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " contains donor element " << element_donor_index << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(47);
                }

                const cgsize_t new_element_donor_index = new_boundary_indices[section_index][element_donor_index - section_ranges[section_index][0]];

                if (new_element_index != 0) {
                    if (new_element_donor_index != 0) {
                        ++n_connectivity_elements_in_proc;
                    }
                    else {
                        const cgsize_t domain_element = element_to_element[4 * n_elements_domain + element_index - n_elements_domain - 1];
                        const cgsize_t donor_domain_element = element_to_element[4 * n_elements_domain + element_donor_index - n_elements_domain - 1];

                        cgsize_t destination_proc = static_cast<cgsize_t>(-1);
                        cgsize_t element_index_in_destination = static_cast<cgsize_t>(-1);
                        for (cgsize_t process_index = 0; process_index < n_proc; ++process_index) {
                            if (donor_domain_element >= starting_elements[process_index] && donor_domain_element < starting_elements[process_index] + n_elements[process_index]) {
                                destination_proc = process_index;
                                element_index_in_destination = donor_domain_element - starting_elements[process_index];
                                break;
                            }
                        }

                        if (destination_proc == static_cast<cgsize_t>(-1)) {
                            std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << " contains neighbour element " << donor_domain_element << " but it is not found in any process. Exiting." << std::endl;
                            exit(61);
                        }

                        const cgsize_t element_index_in_proc = domain_element - starting_elements[i];

                        cotton_eyed_joe[i][destination_proc].push_back({element_index_in_proc, element_index_in_destination});
                        origin_and_destination_ghosts[i][destination_proc].push_back({new_element_index, static_cast<cgsize_t>(-1)});
                    }
                }
            }

            if (n_connectivity_elements_in_proc > 0) {
                std::vector<cgsize_t> local_connectivity_elements;
                local_connectivity_elements.reserve(n_connectivity_elements_in_proc);
                std::vector<cgsize_t> local_connectivity_donor_elements;
                local_connectivity_donor_elements.reserve(n_connectivity_elements_in_proc);

                for (cgsize_t j = 0; j < connectivity_sizes[index_connectivity]; ++j) {
                    const cgsize_t element_index = interface_elements[index_connectivity][j];
                    const cgsize_t element_donor_index = interface_donor_elements[index_connectivity][j];
                    
                    int section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((element_index >= section_ranges[k][0]) && (element_index <= section_ranges[k][1])) {
                            section_index = k;
                            break;
                        }
                    }
                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " contains element " << element_index << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(46);
                    }

                    const cgsize_t new_element_index = new_boundary_indices[section_index][element_index - section_ranges[section_index][0]];

                    section_index = -1;
                    for (int k = 0; k < n_sections; ++k) {
                        if ((element_donor_index >= section_ranges[k][0]) && (element_donor_index <= section_ranges[k][1])) {
                            section_index = k;
                            break;
                        }
                    }
                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << ", connectivity " << index_connectivity << " contains donor element " << element_donor_index << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(47);
                    }

                    const cgsize_t new_element_donor_index = new_boundary_indices[section_index][element_donor_index - section_ranges[section_index][0]];

                    if (new_element_index != 0 && new_element_donor_index != 0) {
                        local_connectivity_elements.push_back(new_element_index + 1);
                        local_connectivity_donor_elements.push_back(new_element_donor_index + 1);
                    }
                }

                int index_out_connectivity = 0;
                cg_conn_write(index_out_file, index_out_base, index_out_zone[i], connectivity_names[index_connectivity].data(), connectivity_grid_locations[index_connectivity], connectivity_types[index_connectivity], connectivity_point_set_types[index_connectivity], n_connectivity_elements_in_proc, local_connectivity_elements.data(), ss.str().c_str(), connectivity_donor_zone_types[index_connectivity], connectivity_donor_point_set_types[index_connectivity], connectivity_donor_data_types[index_connectivity], n_connectivity_elements_in_proc, local_connectivity_donor_elements.data(), &index_out_connectivity);
            }
        }
    }

    // Adding cross-zone interfaces
    for (cgsize_t i = 0; i < n_proc; ++i) {
        for (cgsize_t j = 0; j < n_proc; ++j) {
            for (cgsize_t k = 0; k < cotton_eyed_joe[j][i].size(); ++k) {
                const cgsize_t domain_element_in_this_proc = cotton_eyed_joe[j][i][k][1];
                const cgsize_t domain_element_in_other_proc = cotton_eyed_joe[j][i][k][0];

                for (cgsize_t m = 0; m < cotton_eyed_joe[i][j].size(); ++m) {
                    if (cotton_eyed_joe[i][j][m][0] == domain_element_in_this_proc && cotton_eyed_joe[i][j][m][1] == domain_element_in_other_proc) {
                        origin_and_destination_ghosts[j][i][k][1] = origin_and_destination_ghosts[i][j][m][0];
                    }
                }
            } 
        }
    }

    // Writing cross-zone interfaces to file
    for (cgsize_t i = 0; i < n_proc; ++i) {
        for (cgsize_t j = 0; j < n_proc; ++j) {
            if (origin_and_destination_ghosts[i][j].size() != 0) {
                std::vector<cgsize_t> connectivity_elements(origin_and_destination_ghosts[i][j].size());
                std::vector<cgsize_t> connectivity_donor_elements(origin_and_destination_ghosts[i][j].size());
                for (cgsize_t k = 0; k < origin_and_destination_ghosts[i][j].size(); ++k) {
                    connectivity_elements[k] = origin_and_destination_ghosts[i][j][k][0] + 1;
                    connectivity_donor_elements[k] = origin_and_destination_ghosts[i][j][k][1] + 1;
                }

                int index_out_connectivity = 0;
                std::stringstream ss2;
                ss2 << "Connectivity" << i + 1 << "to" << j + 1;
                std::stringstream ss3;
                ss3 << "Zone " << j + 1;
                cg_conn_write(index_out_file, index_out_base, index_out_zone[i], ss2.str().c_str(), GridLocation_t::FaceCenter, GridConnectivityType_t::Abutting1to1, PointSetType_t::PointList, connectivity_elements.size(), connectivity_elements.data(), ss3.str().c_str(), ZoneType_t::Unstructured, PointSetType_t::PointListDonor, DataType_t::Integer, connectivity_donor_elements.size(), connectivity_donor_elements.data(), &index_out_connectivity);
            }
        }
    }

    const int close_out_error = cg_close(index_out_file);
    if (close_out_error != CG_OK) {
        std::cerr << "Error: output file '" << out_file << "' could not be closed with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(42);
    }

    return 0;
}