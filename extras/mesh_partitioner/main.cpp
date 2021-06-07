#include "helpers/InputParser_t.h"
#include "cgnslib.h"
#include <string>
#include <iostream>
#include <filesystem>
#include <array>
#include <vector>

namespace fs = std::filesystem;

constexpr int CGIO_MAX_NAME_LENGTH = 33; // Includes the null terminator

auto get_input_file(const SEM::Helpers::InputParser_t& input_parser) -> fs::path {
    const std::string input_path = input_parser.getCmdOption("--in_path");
    if (!input_path.empty()) {
        return input_path;
    }
    else {
        const std::string input_filename = input_parser.getCmdOption("--in_filename");
        const std::string save_filename = (input_filename.empty()) ? "mesh.cgns" : input_filename;

        const std::string input_save_dir = input_parser.getCmdOption("--in_directory");
        const fs::path save_dir = (input_save_dir.empty()) ? fs::current_path() / "meshes" : input_save_dir;

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
        const std::string output_filename = input_parser.getCmdOption("--out_filename");
        const std::string save_filename = (output_filename.empty()) ? "mesh_partitioned.cgns" : output_filename;

        const std::string output_save_dir = input_parser.getCmdOption("--out_directory");
        const fs::path save_dir = (output_save_dir.empty()) ? fs::current_path() / "meshes" : output_save_dir;

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
            node_to_element[node_index].push_back(j); // Doesn't work if elements have the same node multiple times. Shouldn't happen with correct meshes I think, otherwise do as in Mesh2D_t.
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

                    auto it = std::find(iterator_begin, iterator_end, node_index);
                    if (it != iterator_end) {
                        const cgsize_t node_element_index = it - elements.begin();
                        const cgsize_t node_element_index_start = (element_index < n_elements_domain) ? 4 * element_index : 4 * n_elements_domain + 2 * (element_index - n_elements_domain);
                        const cgsize_t node_element_index_end = (element_index < n_elements_domain) ? 4 * element_index + 4 : 4 * n_elements_domain + 2 * (element_index - n_elements_domain) + 2;
                        
                        for (cgsize_t node_element_index_next = node_element_index_start; node_element_index_next < node_element_index; ++node_element_index_next) {
                            if (elements[node_element_index_next] == node_index_next) {
                                element_to_element[4 * i + j] = element_index;
                                goto endloop; // I hate this too don't worry
                            }
                        }

                        for (size_t node_element_index_next = node_element_index + 1; node_element_index_next < node_element_index_end; ++node_element_index_next) {
                            if (elements[node_element_index_next] == node_index_next) {
                                element_to_element[4 * i + j] = element_index;
                                goto endloop; // I hate this too don't worry
                            }
                        }
                    }
                }
            }
            endloop: ;
        }
    }

    for (cgsize_t i = 0; i < n_elements_domain; ++i) {
        const cgsize_t node_index = elements[4 * n_elements_domain + 2 * i];
        const cgsize_t node_index_next = elements[4 * n_elements_domain + 2 * i + 1];

        for (auto element_index : node_to_element[node_index - 1]) {
            if (element_index != i) {
                const auto iterator_begin = (element_index < n_elements_domain) ? elements.begin() + 4 * element_index : elements.begin() + 4 * n_elements_domain + 2 * (element_index - n_elements_domain);
                const auto iterator_end = (element_index < n_elements_domain) ? elements.begin() + 4 * element_index + 4 : elements.begin() + 4 * n_elements_domain + 2 * (element_index - n_elements_domain) + 2;

                auto it = std::find(iterator_begin, iterator_end, node_index);
                if (it != iterator_end) {
                    const cgsize_t node_element_index = it - elements.begin();
                    const cgsize_t node_element_index_start = (element_index < n_elements_domain) ? 4 * element_index : 4 * n_elements_domain + 2 * (element_index - n_elements_domain);
                    const cgsize_t node_element_index_end = (element_index < n_elements_domain) ? 4 * element_index + 4 : 4 * n_elements_domain + 2 * (element_index - n_elements_domain) + 2;
                    
                    for (cgsize_t node_element_index_next = node_element_index_start; node_element_index_next < node_element_index; ++node_element_index_next) {
                        if (elements[node_element_index_next] == node_index_next) {
                            element_to_element[4 * n_elements_domain + i] = element_index;
                            goto endloop2; // I hate this too don't worry
                        }
                    }

                    for (size_t node_element_index_next = node_element_index + 1; node_element_index_next < node_element_index_end; ++node_element_index_next) {
                        if (elements[node_element_index_next] == node_index_next) {
                            element_to_element[4 * n_elements_domain + i] = element_index;
                            goto endloop2; // I hate this too don't worry
                        }
                    }
                }
            }
        }
        endloop2: ;
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

    const fs::path in_file = get_input_file(input_parser);
    const fs::path out_file = get_output_file(input_parser);

    const std::string input_n_proc = input_parser.getCmdOption("--n");
    const cgsize_t n_proc = (input_n_proc.empty()) ? 4 : std::stoi(input_n_proc);

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
    const cgsize_t n_elements = isize[1];

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
        parent_data[index_section - 1] = std::vector<cgsize_t>(section_ranges[index_section - 1][1] - section_ranges[index_section - 1][0]);

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
        interface_elements[index_connectivity - 1] = std::vector<int>(connectivity_sizes[index_connectivity - 1]);
        interface_donor_elements[index_connectivity - 1] = std::vector<int>(connectivity_donor_sizes[index_connectivity - 1]);
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

    if (n_elements_domain + n_elements_ghost != n_elements) {
        std::cerr << "Error: CGNS mesh, base " << index_in_base << ", zone " << index_in_zone << " has " << n_elements << " elements but the sum of its sections is " << n_elements_domain + n_elements_ghost << " elements. Exiting." << std::endl;
        exit(34);  
    }

    // Splitting elements
    const cgsize_t N_elements_per_process = (n_elements_domain + n_proc - 1)/n_proc;
    std::vector<cgsize_t> N_elements(n_proc);
    std::vector<cgsize_t> starting_elements(n_proc);
    cgsize_t starting_element = 0;
    for (cgsize_t i = 0; i < n_proc; ++i) {
        N_elements[i] = (i == n_proc - 1) ? N_elements_per_process + n_elements_domain - N_elements_per_process * n_proc : N_elements_per_process;
        starting_elements[i] = starting_element;
        starting_element += N_elements[i];
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
    for (cgsize_t i = 0; i < n_proc; ++i) {
        // Getting section elements
        std::vector<cgsize_t> elements_in_proc(4 * N_elements[i]);
        for (cgsize_t element_index = 0; element_index < N_elements[i]; ++element_index) {
            for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                elements_in_proc[4 * element_index + side_index] = elements[4 * (element_index + starting_elements[i]) + side_index];
            }
        }

        // Getting relevant points
        std::vector<bool> is_point_needed(n_nodes);
        for (cgsize_t element_index = 0; element_index < N_elements[i]; ++element_index) {
            for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                is_point_needed[elements_in_proc[4 * element_index + side_index] - 1] = true;
            }
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

                // Replacing point indices
                for (cgsize_t element_index = 0; element_index < N_elements[i]; ++element_index) {
                    for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                        if (elements_in_proc[4 * element_index + side_index] == node_index + 1) {
                            elements_in_proc[4 * element_index + side_index] = xy_in_proc[0].size();
                        }
                    }
                }
            }
        }

        // Getting relevant boundaries
        for (int j = 0; j < n_boundaries; ++j) {

        }
        

        // Writing zone information to file
        /* vertex size, cell size, boundary vertex size (always zero for structured grids) */
        std::array<cgsize_t, 3> isize {n_nodes_in_proc,
                                       N_elements[i], //n_elements_total_in_proc, // CHECK this is wrong! add n boundary elements
                                       0};

        std::stringstream ss;
        ss << "Zone " << i + 1;
        int index_out_zone = 0;
        cg_zone_write(index_out_file, index_out_base, ss.str().c_str(), isize.data(), ZoneType_t::Unstructured, &index_out_zone);

        /* write grid coordinates (user must use SIDS-standard names here) */
        int index_out_coord = 0;
        cg_coord_write(index_out_file, index_out_base, index_out_zone, DataType_t::RealDouble, coord_names[0].data(), xy_in_proc[0].data(), &index_out_coord);
        cg_coord_write(index_out_file, index_out_base, index_out_zone, DataType_t::RealDouble, coord_names[1].data(), xy_in_proc[1].data(), &index_out_coord);

        /* write QUAD_4 element connectivity (user can give any name) */
        const std::string elements_name("Elements");
        int index_out_section = 0;
        const int nelem_start = 1;
        const int nelem_end = N_elements[i];
        const int n_boundary_elem = 0; // No boundaries yet
        cg_section_write(index_out_file, index_out_base, index_out_zone, elements_name.c_str(), ElementType_t::QUAD_4, nelem_start, nelem_end, n_boundary_elem, elements_in_proc.data(), &index_out_section);
    }

    const int close_out_error = cg_close(index_out_file);
    if (close_out_error != CG_OK) {
        std::cerr << "Error: output file '" << out_file << "' could not be closed with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(42);
    }

    return 0;
}