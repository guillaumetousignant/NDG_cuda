#include "helpers/InputParser_t.h"
#include "cgnslib.h"
#include <string>
#include <iostream>
#include <filesystem>
#include <array>
#include <vector>
#include <fstream>

// I think this was the most annoying thing to write yet.

namespace fs = std::filesystem;

constexpr size_t CGIO_MAX_NAME_LENGTH_WITHOUT_TERMINATOR = 32; // Does not include the null terminator
constexpr size_t CGIO_MAX_NAME_LENGTH = CGIO_MAX_NAME_LENGTH_WITHOUT_TERMINATOR + 1; // Includes the null terminator

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
        if (out_file.has_parent_path()) {
            fs::create_directory(out_file.parent_path());
        }
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
            node_to_element[node_index].push_back(j); // Doesn't work if elements have the same node multiple times. Shouldn't happen with correct meshes I think, otherwise do as in MeshPart_t.
        }
    }
    for (cgsize_t j = 0; j < n_elements_ghost; ++j) {
        for (cgsize_t side_index = 0; side_index < 2; ++side_index) {
            const size_t node_index = elements[4 * n_elements_domain + 2 * j + side_index] - 1;
            node_to_element[node_index].push_back(n_elements_domain + j); // Doesn't work if elements have the same node multiple times. Shouldn't happen with correct meshes I think, otherwise do as in MeshPart_t.
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

class MeshPart_t {
    public:
        MeshPart_t(cgsize_t n_elements_domain, 
                   cgsize_t n_elements_ghost, 
                   int dim, 
                   int physDim, 
                   int index_in_base, 
                   int index_in_zone, 
                   ZoneType_t zone_type, 
                   std::array<char, CGIO_MAX_NAME_LENGTH> base_name, 
                   std::array<std::array<char, CGIO_MAX_NAME_LENGTH>, 2> coord_names, 
                   std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> section_names, 
                   std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> boundary_names, 
                   std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> connectivity_names,
                   std::vector<bool> section_is_domain, 
                   std::vector<std::array<cgsize_t, 2>> section_ranges, 
                   std::vector<std::vector<cgsize_t>> connectivity, 
                   std::array<std::vector<double>, 2> xy, 
                   std::vector<cgsize_t> boundary_sizes, 
                   std::vector<std::vector<cgsize_t>> boundary_elements, 
                   std::vector<BCType_t> boundary_types, 
                   std::vector<PointSetType_t> boundary_point_set_types, 
                   std::vector<PointSetType_t> connectivity_point_set_types,
                   std::vector<PointSetType_t> connectivity_donor_point_set_types,
                   std::vector<GridLocation_t> boundary_grid_locations, 
                   std::vector<GridLocation_t> connectivity_grid_locations,
                   std::vector<cgsize_t> connectivity_sizes,
                   std::vector<std::vector<cgsize_t>> interface_elements,
                   std::vector<std::vector<cgsize_t>> interface_donor_elements,
                   std::vector<GridConnectivityType_t> connectivity_types,
                   std::vector<ZoneType_t> connectivity_donor_zone_types,
                   std::vector<DataType_t> connectivity_donor_data_types) : 
            n_elements_domain_(n_elements_domain),
            n_elements_ghost_(n_elements_ghost),
            dim_(dim),
            physDim_(physDim),
            index_in_base_(index_in_base),
            zone_type_(zone_type),
            base_name_(base_name),
            coord_names_(coord_names),
            section_names_(section_names),
            boundary_names_(boundary_names),
            connectivity_names_(connectivity_names),
            section_is_domain_(section_is_domain),
            section_ranges_(section_ranges),
            connectivity_(connectivity),
            xy_(xy),
            boundary_sizes_(boundary_sizes),
            boundary_elements_(boundary_elements),
            boundary_types_(boundary_types),
            boundary_point_set_types_(boundary_point_set_types),
            connectivity_point_set_types_(connectivity_point_set_types),
            connectivity_donor_point_set_types_(connectivity_donor_point_set_types),
            boundary_grid_locations_(boundary_grid_locations),
            connectivity_grid_locations_(connectivity_grid_locations),
            connectivity_sizes_(connectivity_sizes),
            interface_elements_(interface_elements),
            interface_donor_elements_(interface_donor_elements),
            connectivity_types_(connectivity_types),
            connectivity_donor_zone_types_(connectivity_donor_zone_types),
            connectivity_donor_data_types_(connectivity_donor_data_types) {};

        cgsize_t n_elements_domain_;
        cgsize_t n_elements_ghost_;
        int dim_;
        int physDim_;
        int index_in_base_;
        int index_in_zone_;
        ZoneType_t zone_type_;
        std::array<char, CGIO_MAX_NAME_LENGTH> base_name_;
        std::array<std::array<char, CGIO_MAX_NAME_LENGTH>, 2> coord_names_;
        std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> section_names_;
        std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> boundary_names_;
        std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> connectivity_names_;
        std::vector<bool> section_is_domain_;
        std::vector<std::array<cgsize_t, 2>> section_ranges_;
        std::vector<std::vector<cgsize_t>> connectivity_;
        std::array<std::vector<double>, 2> xy_;
        std::vector<cgsize_t> boundary_sizes_;
        std::vector<std::vector<cgsize_t>> boundary_elements_;
        std::vector<BCType_t> boundary_types_;
        std::vector<PointSetType_t> boundary_point_set_types_;
        std::vector<PointSetType_t> connectivity_point_set_types_;
        std::vector<PointSetType_t> connectivity_donor_point_set_types_;
        std::vector<GridLocation_t> boundary_grid_locations_;
        std::vector<GridLocation_t> connectivity_grid_locations_;
        std::vector<cgsize_t> connectivity_sizes_;
        std::vector<std::vector<cgsize_t>> interface_elements_;
        std::vector<std::vector<cgsize_t>> interface_donor_elements_;
        std::vector<GridConnectivityType_t> connectivity_types_;
        std::vector<ZoneType_t> connectivity_donor_zone_types_;
        std::vector<DataType_t> connectivity_donor_data_types_;
};

auto read_cgns_mesh(const fs::path in_file) -> MeshPart_t {
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

    return MeshPart_t(n_elements_domain, 
                      n_elements_ghost, 
                      dim, 
                      physDim, 
                      index_in_base, 
                      index_in_zone, 
                      zone_type, 
                      base_name, 
                      coord_names, 
                      std::move(section_names), 
                      std::move(boundary_names), 
                      std::move(connectivity_names),
                      std::move(section_is_domain), 
                      std::move(section_ranges), 
                      std::move(connectivity), 
                      std::move(xy), 
                      std::move(boundary_sizes), 
                      std::move(boundary_elements), 
                      std::move(boundary_types), 
                      std::move(boundary_point_set_types), 
                      std::move(connectivity_point_set_types),
                      std::move(connectivity_donor_point_set_types),
                      std::move(boundary_grid_locations), 
                      std::move(connectivity_grid_locations),
                      std::move(connectivity_sizes),
                      std::move(interface_elements),
                      std::move(interface_donor_elements),
                      std::move(connectivity_types),
                      std::move(connectivity_donor_zone_types),
                      std::move(connectivity_donor_data_types));
}

auto read_su2_mesh(const fs::path in_file) -> MeshPart_t {
    std::string line;
    std::string token;
    size_t value;

    std::ifstream meshfile(in_file);
    if (!meshfile.is_open()) {
        std::cerr << "Error: file '" << in_file << "' could not be opened. Exiting." << std::endl;
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

    std::array<std::vector<double>, 2> xy;
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
            xy[0] = std::vector<double>(value);
            xy[1] = std::vector<double>(value);

            for (size_t i = 0; i < xy[0].size(); ++i) {
                std::getline(meshfile, line);
                std::istringstream liness2(line);
                liness2 >> xy[0][i] >> xy[0][i];
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
                else if (type == "wall") {
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

    // Change to a format like CGNS
    std::array<std::array<char, CGIO_MAX_NAME_LENGTH>, 2> coord_names {"CoordinateX", "CoordinateY"};
    const int n_boundaries = !wall.empty() + !symmetry.empty() + !inflow.empty() + !outflow.empty();
    const int n_sections = 1 + n_boundaries;
    const int wall_boundary_index = 0;
    const int symmetry_boundary_index = !wall.empty();
    const int inflow_boundary_index = !wall.empty() + !symmetry.empty();
    const int outflow_boundary_index = !wall.empty() + !symmetry.empty() + !inflow.empty();
    const int wall_section_index = 1;
    const int symmetry_section_index = 1 + symmetry_boundary_index;
    const int inflow_section_index = 1 + + inflow_boundary_index;
    const int outflow_section_index = 1 + outflow_boundary_index;

    std::vector<bool> section_is_domain(n_sections);
    std::vector<std::array<cgsize_t, 2>> section_ranges(n_sections);
    std::vector<std::vector<cgsize_t>> connectivity(n_sections);
    std::vector<cgsize_t> boundary_sizes(n_boundaries);
    std::vector<std::vector<cgsize_t>> boundary_elements(n_boundaries);
    std::vector<BCType_t> boundary_types(n_boundaries);
    std::vector<PointSetType_t> boundary_point_set_types(n_boundaries, PointSetType_t::PointList);
    std::vector<PointSetType_t> connectivity_point_set_types; // No connectivity in su2
    std::vector<PointSetType_t> connectivity_donor_point_set_types; // No connectivity in su2
    std::vector<GridLocation_t> boundary_grid_locations(n_boundaries, GridLocation_t::EdgeCenter);
    std::vector<GridLocation_t> connectivity_grid_locations; // No connectivity in su2
    std::vector<cgsize_t> connectivity_sizes; // No connectivity in su2
    std::vector<std::vector<cgsize_t>> interface_elements; // No connectivity in su2
    std::vector<std::vector<cgsize_t>> interface_donor_elements; // No connectivity in su2
    std::vector<GridConnectivityType_t> connectivity_types; // No connectivity in su2
    std::vector<ZoneType_t> connectivity_donor_zone_types; // No connectivity in su2
    std::vector<DataType_t> connectivity_donor_data_types; // No connectivity in su2


    std::array<char, CGIO_MAX_NAME_LENGTH> base_name;
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> section_names(n_sections);
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> boundary_names(n_boundaries);
    std::vector<std::array<char, CGIO_MAX_NAME_LENGTH>> connectivity_names; // No connectivity in su2

    if (in_file.has_stem()) {
        const std::string stem = in_file.stem().string();
        for (size_t i = 0; i < std::min(stem.size(), CGIO_MAX_NAME_LENGTH_WITHOUT_TERMINATOR); ++i) {
            base_name[i] = stem[i];
        }
        base_name[std::min(stem.size(), CGIO_MAX_NAME_LENGTH_WITHOUT_TERMINATOR)] = '\0';
    }
    else {
        base_name[0] = 'B';
        base_name[1] = 'a';
        base_name[2] = 's';
        base_name[3] = 'e';
        base_name[4] = '\0';
    }
    
    section_is_domain[0] = true;
    section_ranges[0][0] = 1;
    section_ranges[0][1] = elements.size();
    section_names[0][0] = 'E'; // straight madness
    section_names[0][1] = 'l';
    section_names[0][2] = 'e';
    section_names[0][3] = 'm';
    section_names[0][4] = 'e';
    section_names[0][5] = 'n';
    section_names[0][6] = 't';
    section_names[0][7] = 's';
    section_names[0][8] = '\0';
    connectivity[0] = std::vector<cgsize_t>(4 * elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        connectivity[0][4 * i] = elements[i][0] + 1;
        connectivity[0][4 * i + 1] = elements[i][1] + 1;
        connectivity[0][4 * i + 2] = elements[i][2] + 1;
        connectivity[0][4 * i + 3] = elements[i][3] + 1;
    }

    if (!wall.empty()) {
        section_is_domain[wall_section_index] = false;
        section_ranges[wall_section_index][0] = elements.size() + 1;
        section_ranges[wall_section_index][1] = elements.size() + wall.size();
        boundary_sizes[wall_boundary_index] = wall.size();
        section_names[wall_section_index][0] = 'W'; // straight madness
        section_names[wall_section_index][1] = 'a';
        section_names[wall_section_index][2] = 'l';
        section_names[wall_section_index][3] = 'l';
        section_names[wall_section_index][4] = 'E';
        section_names[wall_section_index][5] = 'l';
        section_names[wall_section_index][6] = 'e';
        section_names[wall_section_index][7] = 'm';
        section_names[wall_section_index][8] = 'e';
        section_names[wall_section_index][9] = 'n';
        section_names[wall_section_index][10] = 't';
        section_names[wall_section_index][11] = 's';
        section_names[wall_section_index][12] = '\0';
        boundary_names[wall_boundary_index][0] = 'W';
        boundary_names[wall_boundary_index][1] = 'a';
        boundary_names[wall_boundary_index][2] = 'l';
        boundary_names[wall_boundary_index][3] = 'l';
        boundary_names[wall_boundary_index][4] = 'B';
        boundary_names[wall_boundary_index][5] = 'o';
        boundary_names[wall_boundary_index][6] = 'u';
        boundary_names[wall_boundary_index][7] = 'n';
        boundary_names[wall_boundary_index][8] = 'd';
        boundary_names[wall_boundary_index][9] = 'a';
        boundary_names[wall_boundary_index][10] = 'r';
        boundary_names[wall_boundary_index][11] = 'y';
        boundary_names[wall_boundary_index][12] = '\0';
        boundary_types[wall_boundary_index] = BCType_t::BCWall;
        connectivity[wall_section_index] = std::vector<cgsize_t>(2 * wall.size());
        boundary_elements[wall_boundary_index] = std::vector<cgsize_t>(wall.size());
        for (size_t i = 0; i < wall.size(); ++i) {
            connectivity[wall_section_index][2 * i] = wall[i][0] + 1;
            connectivity[wall_section_index][2 * i + 1] = wall[i][1] + 1;
            boundary_elements[wall_boundary_index][i] = section_ranges[wall_section_index][0] + i;
        }
    }

    if (!symmetry.empty()) {
        section_is_domain[symmetry_section_index] = false;
        section_ranges[symmetry_section_index][0] = elements.size() + wall.size() + 1;
        section_ranges[symmetry_section_index][1] = elements.size() + wall.size() + symmetry.size();
        boundary_sizes[symmetry_boundary_index] = symmetry.size();
        section_names[symmetry_section_index][0] = 'S'; // straight madness
        section_names[symmetry_section_index][1] = 'y';
        section_names[symmetry_section_index][2] = 'm';
        section_names[symmetry_section_index][3] = 'm';
        section_names[symmetry_section_index][4] = 'e';
        section_names[symmetry_section_index][5] = 't';
        section_names[symmetry_section_index][6] = 'r';
        section_names[symmetry_section_index][7] = 'y';
        section_names[symmetry_section_index][8] = 'E';
        section_names[symmetry_section_index][9] = 'l';
        section_names[symmetry_section_index][10] = 'e';
        section_names[symmetry_section_index][11] = 'm';
        section_names[symmetry_section_index][12] = 'e';
        section_names[symmetry_section_index][13] = 'n';
        section_names[symmetry_section_index][14] = 't';
        section_names[symmetry_section_index][15] = 's';
        section_names[symmetry_section_index][16] = '\0';
        boundary_names[symmetry_boundary_index][0] = 'S';
        boundary_names[symmetry_boundary_index][1] = 'y';
        boundary_names[symmetry_boundary_index][2] = 'm';
        boundary_names[symmetry_boundary_index][3] = 'm';
        boundary_names[symmetry_boundary_index][4] = 'e';
        boundary_names[symmetry_boundary_index][5] = 't';
        boundary_names[symmetry_boundary_index][6] = 'r';
        boundary_names[symmetry_boundary_index][7] = 'y';
        boundary_names[symmetry_boundary_index][8] = 'B';
        boundary_names[symmetry_boundary_index][9] = 'o';
        boundary_names[symmetry_boundary_index][10] = 'u';
        boundary_names[symmetry_boundary_index][11] = 'n';
        boundary_names[symmetry_boundary_index][12] = 'd';
        boundary_names[symmetry_boundary_index][13] = 'a';
        boundary_names[symmetry_boundary_index][14] = 'r';
        boundary_names[symmetry_boundary_index][15] = 'y';
        boundary_names[symmetry_boundary_index][16] = '\0';
        boundary_types[symmetry_boundary_index] = BCType_t::BCSymmetryPlane;
        connectivity[symmetry_section_index] = std::vector<cgsize_t>(2 * symmetry.size());
        boundary_elements[symmetry_boundary_index] = std::vector<cgsize_t>(symmetry.size());
        for (size_t i = 0; i < symmetry.size(); ++i) {
            connectivity[symmetry_section_index][2 * i] = symmetry[i][0] + 1;
            connectivity[symmetry_section_index][2 * i + 1] = symmetry[i][1] + 1;
            boundary_elements[symmetry_boundary_index][i] = section_ranges[symmetry_section_index][0] + i;
        }
    }

    if (!inflow.empty()) {
        section_is_domain[inflow_section_index] = false;
        section_ranges[inflow_section_index][0] = elements.size() + wall.size() + symmetry.size() + 1;
        section_ranges[inflow_section_index][1] = elements.size() + wall.size() + symmetry.size() + inflow.size();
        boundary_sizes[inflow_boundary_index] = inflow.size();
        section_names[inflow_section_index][0] = 'I'; // straight madness
        section_names[inflow_section_index][1] = 'n';
        section_names[inflow_section_index][2] = 'f';
        section_names[inflow_section_index][3] = 'l';
        section_names[inflow_section_index][4] = 'o';
        section_names[inflow_section_index][5] = 'w';
        section_names[inflow_section_index][6] = 'E';
        section_names[inflow_section_index][7] = 'l';
        section_names[inflow_section_index][8] = 'e';
        section_names[inflow_section_index][9] = 'm';
        section_names[inflow_section_index][10] = 'e';
        section_names[inflow_section_index][11] = 'n';
        section_names[inflow_section_index][12] = 't';
        section_names[inflow_section_index][13] = 's';
        section_names[inflow_section_index][14] = '\0';
        boundary_names[inflow_boundary_index][0] = 'I';
        boundary_names[inflow_boundary_index][1] = 'n';
        boundary_names[inflow_boundary_index][2] = 'f';
        boundary_names[inflow_boundary_index][3] = 'l';
        boundary_names[inflow_boundary_index][4] = 'o';
        boundary_names[inflow_boundary_index][5] = 'w';
        boundary_names[inflow_boundary_index][6] = 'B';
        boundary_names[inflow_boundary_index][7] = 'o';
        boundary_names[inflow_boundary_index][8] = 'u';
        boundary_names[inflow_boundary_index][9] = 'n';
        boundary_names[inflow_boundary_index][10] = 'd';
        boundary_names[inflow_boundary_index][11] = 'a';
        boundary_names[inflow_boundary_index][12] = 'r';
        boundary_names[inflow_boundary_index][13] = 'y';
        boundary_names[inflow_boundary_index][14] = '\0';
        boundary_types[inflow_boundary_index] = BCType_t::BCInflow;
        connectivity[inflow_section_index] = std::vector<cgsize_t>(2 * inflow.size());
        boundary_elements[inflow_boundary_index] = std::vector<cgsize_t>(inflow.size());
        for (size_t i = 0; i < inflow.size(); ++i) {
            connectivity[inflow_section_index][2 * i] = inflow[i][0] + 1;
            connectivity[inflow_section_index][2 * i + 1] = inflow[i][1] + 1;
            boundary_elements[inflow_boundary_index][i] = section_ranges[inflow_section_index][0] + i;
        }
    }

    if (!outflow.empty()) {
        section_is_domain[outflow_section_index] = false;
        section_ranges[outflow_section_index][0] = elements.size() + wall.size() + symmetry.size() + inflow.size() + 1;
        section_ranges[outflow_section_index][1] = elements.size() + wall.size() + symmetry.size() + inflow.size() + outflow.size();
        boundary_sizes[outflow_boundary_index] = outflow.size();
        section_names[outflow_section_index][0] = 'O'; // straight madness
        section_names[outflow_section_index][1] = 'u';
        section_names[outflow_section_index][2] = 't';
        section_names[outflow_section_index][3] = 'f';
        section_names[outflow_section_index][4] = 'l';
        section_names[outflow_section_index][5] = 'o';
        section_names[outflow_section_index][6] = 'w';
        section_names[outflow_section_index][7] = 'E';
        section_names[outflow_section_index][8] = 'l';
        section_names[outflow_section_index][9] = 'e';
        section_names[outflow_section_index][10] = 'm';
        section_names[outflow_section_index][11] = 'e';
        section_names[outflow_section_index][12] = 'n';
        section_names[outflow_section_index][13] = 't';
        section_names[outflow_section_index][14] = 's';
        section_names[outflow_section_index][15] = '\0';
        boundary_names[outflow_boundary_index][0] = 'O';
        boundary_names[outflow_boundary_index][1] = 'u';
        boundary_names[outflow_boundary_index][2] = 't';
        boundary_names[outflow_boundary_index][3] = 'f';
        boundary_names[outflow_boundary_index][4] = 'l';
        boundary_names[outflow_boundary_index][5] = 'o';
        boundary_names[outflow_boundary_index][6] = 'w';
        boundary_names[outflow_boundary_index][7] = 'B';
        boundary_names[outflow_boundary_index][8] = 'o';
        boundary_names[outflow_boundary_index][9] = 'u';
        boundary_names[outflow_boundary_index][10] = 'n';
        boundary_names[outflow_boundary_index][11] = 'd';
        boundary_names[outflow_boundary_index][12] = 'a';
        boundary_names[outflow_boundary_index][13] = 'r';
        boundary_names[outflow_boundary_index][14] = 'y';
        boundary_names[outflow_boundary_index][15] = '\0';
        boundary_types[outflow_boundary_index] = BCType_t::BCOutflow;
        connectivity[outflow_section_index] = std::vector<cgsize_t>(2 * outflow.size());
        boundary_elements[outflow_boundary_index] = std::vector<cgsize_t>(outflow.size());
        for (size_t i = 0; i < outflow.size(); ++i) {
            connectivity[outflow_section_index][2 * i] = outflow[i][0] + 1;
            connectivity[outflow_section_index][2 * i + 1] = outflow[i][1] + 1;
            boundary_elements[outflow_boundary_index][i] = section_ranges[outflow_section_index][0] + i;
        }
    }

    return MeshPart_t{static_cast<cgsize_t>(elements.size()), 
                      static_cast<cgsize_t>(wall.size() + symmetry.size() + inflow.size() + outflow.size()), 
                      2, 
                      2, 
                      1, 
                      1, 
                      ZoneType_t::Unstructured, 
                      base_name, 
                      coord_names, 
                      std::move(section_names), 
                      std::move(boundary_names), 
                      std::move(connectivity_names),
                      std::move(section_is_domain), 
                      std::move(section_ranges), 
                      std::move(connectivity), 
                      std::move(xy), 
                      std::move(boundary_sizes), 
                      std::move(boundary_elements), 
                      std::move(boundary_types), 
                      std::move(boundary_point_set_types), 
                      std::move(connectivity_point_set_types),
                      std::move(connectivity_donor_point_set_types),
                      std::move(boundary_grid_locations), 
                      std::move(connectivity_grid_locations),
                      std::move(connectivity_sizes),
                      std::move(interface_elements),
                      std::move(interface_donor_elements),
                      std::move(connectivity_types),
                      std::move(connectivity_donor_zone_types),
                      std::move(connectivity_donor_data_types)};
}

auto write_cgns_mesh(cgsize_t n_proc, const MeshPart_t mesh, const fs::path out_file) -> void {
    // Splitting elements
    const auto [n_elements_div, n_elements_mod] = std::div(mesh.n_elements_domain_, n_proc);
    std::vector<cgsize_t> n_elements(n_proc);
    std::vector<cgsize_t> starting_elements(n_proc);
    for (cgsize_t i = 0; i < n_proc; ++i) {
        starting_elements[i] = i * n_elements_div + std::min(i, n_elements_mod);
        const cgsize_t ending_element = (i + 1) * n_elements_div + std::min(i + 1, n_elements_mod);
        n_elements[i] = ending_element - starting_elements[i];
    }

    // Putting connectivity data together
    std::vector<cgsize_t> elements(4 * mesh.n_elements_domain_ + 2 * mesh.n_elements_ghost_);
    std::vector<cgsize_t> section_start_indices(mesh.section_is_domain_.size());
    cgsize_t element_domain_index = 0;
    cgsize_t element_ghost_index = 4 * mesh.n_elements_domain_;
    for (int i = 0; i < mesh.section_is_domain_.size(); ++i) {
        if (mesh.section_is_domain_[i]) {
            section_start_indices[i] = element_domain_index;
            for (cgsize_t j = 0; j < mesh.section_ranges_[i][1] - mesh.section_ranges_[i][0] + 1; ++j) {
                elements[section_start_indices[i] + 4 * j] = mesh.connectivity_[i][4 * j];
                elements[section_start_indices[i] + 4 * j + 1] = mesh.connectivity_[i][4 * j + 1];
                elements[section_start_indices[i] + 4 * j + 2] = mesh.connectivity_[i][4 * j + 2];
                elements[section_start_indices[i] + 4 * j + 3] = mesh.connectivity_[i][4 * j + 3];
            }
            element_domain_index += 4 * (mesh.section_ranges_[i][1] - mesh.section_ranges_[i][0] + 1);
        }
        else {
            section_start_indices[i] = element_ghost_index;
            for (cgsize_t j = 0; j < mesh.section_ranges_[i][1] - mesh.section_ranges_[i][0] + 1; ++j) {
                elements[section_start_indices[i] + 2 * j] = mesh.connectivity_[i][2 * j];
                elements[section_start_indices[i] + 2 * j + 1] = mesh.connectivity_[i][2 * j + 1];
            }
            element_ghost_index += 2 * (mesh.section_ranges_[i][1] - mesh.section_ranges_[i][0] + 1);
        }
    }

    // Computing nodes to elements
    const std::vector<std::vector<cgsize_t>> node_to_element = build_node_to_element(mesh.xy_[0].size(), elements, mesh.n_elements_domain_, mesh.n_elements_ghost_);

    // Computing element to elements
    const std::vector<cgsize_t> element_to_element = build_element_to_element(elements, node_to_element, mesh.n_elements_domain_, mesh.n_elements_ghost_);

    // Creating output file
    int index_out_file = 0;
    const int open_out_error = cg_open(out_file.string().c_str(), CG_MODE_WRITE, &index_out_file);
    if (open_out_error != CG_OK) {
        std::cerr << "Error: output file '" << out_file << "' could not be opened with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(40);
    }

    int index_out_base = 0;
    cg_base_write(index_out_file, mesh.base_name_.data(), mesh.dim_, mesh.physDim_, &index_out_base);

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
        std::vector<std::vector<cgsize_t>> new_boundary_indices(mesh.section_is_domain_.size());
        for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
            if (!mesh.section_is_domain_[k]) {
                new_boundary_indices[k] = std::vector<cgsize_t>(mesh.section_ranges_[k][1] - mesh.section_ranges_[k][0] + 1);
                for (cgsize_t j = 0; j < mesh.section_ranges_[k][1] - mesh.section_ranges_[k][0] + 1; ++j) {
                    cgsize_t ghost_index = j;
                    for (cgsize_t zone_loop = 0; zone_loop < k; ++zone_loop) {
                        if (!mesh.section_is_domain_[zone_loop]) {
                            ghost_index += mesh.section_ranges_[zone_loop][1] - mesh.section_ranges_[zone_loop][0] + 1;
                        }
                    }

                    const cgsize_t domain_element_index = element_to_element[4 * mesh.n_elements_domain_ + ghost_index];
                    if (domain_element_index >= starting_elements[i] && domain_element_index < starting_elements[i] + n_elements[i]) {
                        new_boundary_indices[k][j] = 1;
                    }
                }
            }
        }
        std::vector<cgsize_t> n_boundaries_in_proc(mesh.section_is_domain_.size());
        std::vector<cgsize_t> boundaries_start_index(mesh.section_is_domain_.size());
        cgsize_t n_total_boundaries_in_proc = 0;
        for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
            if (!mesh.section_is_domain_[k]) {
                boundaries_start_index[k] = n_elements[i] + n_total_boundaries_in_proc;
                for (cgsize_t j = 0; j < mesh.section_ranges_[k][1] - mesh.section_ranges_[k][0] + 1; ++j) {
                    if (new_boundary_indices[k][j]) {
                        new_boundary_indices[k][j] = n_elements[i] + n_total_boundaries_in_proc;
                        ++n_boundaries_in_proc[k];
                        ++n_total_boundaries_in_proc;
                    }
                }
            }
        }
        std::vector<std::vector<cgsize_t>> boundaries_in_proc(mesh.section_is_domain_.size());
        for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
            if (!mesh.section_is_domain_[k]) {
                if (n_boundaries_in_proc[k] > 0) {
                    boundaries_in_proc[k].reserve(n_boundaries_in_proc[k] * 2);
                    n_elements_in_proc += n_boundaries_in_proc[k];

                    for (cgsize_t j = 0; j < mesh.section_ranges_[k][1] - mesh.section_ranges_[k][0] + 1; ++j) {
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
                if (neighbour_element_index < mesh.n_elements_domain_ && !(neighbour_element_index >= starting_elements[i] && neighbour_element_index < starting_elements[i] + n_elements[i])) {
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
                        std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << " contains neighbour element " << neighbour_element_index << " but it is not found in any process. Exiting." << std::endl;
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
        std::vector<bool> is_point_needed(mesh.xy_[0].size());
        for (cgsize_t element_index = 0; element_index < n_elements[i]; ++element_index) {
            for (cgsize_t side_index = 0; side_index < 4; ++side_index) {
                is_point_needed[elements_in_proc[4 * element_index + side_index] - 1] = true;
            }
        }
        for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
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

        for (cgsize_t node_index = 0; node_index < mesh.xy_[0].size(); ++node_index) {
            if (is_point_needed[node_index]) {
                xy_in_proc[0].push_back(mesh.xy_[0][node_index]);
                xy_in_proc[1].push_back(mesh.xy_[1][node_index]);

                // Replacing point indices CHECK do with node_to_elem
                for (auto element_index : node_to_element[node_index]) {
                    if (element_index < mesh.n_elements_domain_ ) {
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
                        for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
                            if (element_index + 1 >= mesh.section_ranges_[k][0] && element_index + 1 <= mesh.section_ranges_[k][1]) {
                                section_index = k;
                                break;
                            }
                        }
                        if (section_index == -1) {
                            std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << ", node to elements of point " << node_index << " contains element " << element_index << " but it is not found in any mesh section. Exiting." << std::endl;
                            exit(52);
                        }

                        const cgsize_t element_section_index = element_index + 1 - mesh.section_ranges_[section_index][0];
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
        cg_zone_write(index_out_file, index_out_base, ss.str().c_str(), isize.data(), mesh.zone_type_, &index_out_zone[i]);

        /* write grid coordinates (user must use SIDS-standard names here) */
        int index_out_coord = 0;
        cg_coord_write(index_out_file, index_out_base, index_out_zone[i], DataType_t::RealDouble, mesh.coord_names_[0].data(), xy_in_proc[0].data(), &index_out_coord);
        cg_coord_write(index_out_file, index_out_base, index_out_zone[i], DataType_t::RealDouble, mesh.coord_names_[1].data(), xy_in_proc[1].data(), &index_out_coord);

        // Writing domain sections to file
        cgsize_t section_start = 0;
        cgsize_t section_end = 0;
        cgsize_t element_index = starting_elements[i];
        cgsize_t remaining_elements = n_elements[i];
        cgsize_t domain_index_start = 0;
        cgsize_t domain_index_end = 0;
        for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
            if (mesh.section_is_domain_[k]) {
                section_end += mesh.section_ranges_[k][1] - mesh.section_ranges_[k][0] + 1;

                if (element_index >= section_start && element_index < section_end) {
                    const cgsize_t section_offset = element_index - section_start;
                    const cgsize_t n_elements_in_this_section = mesh.section_ranges_[k][1] - mesh.section_ranges_[k][0] + 1;
                    const cgsize_t section_end_offset = (n_elements_in_this_section - section_offset > remaining_elements) ? n_elements_in_this_section - section_offset - remaining_elements: 0;
                    const cgsize_t n_elements_from_this_section = n_elements_in_this_section - section_offset - section_end_offset;

                    int index_out_section = 0;
                    domain_index_end += n_elements_from_this_section;
                    const cgsize_t n_boundary_elem = 0; // No boundary elements yet
                    cg_section_write(index_out_file, index_out_base, index_out_zone[i], mesh.section_names_[k].data(), ElementType_t::QUAD_4, domain_index_start + 1, domain_index_end, n_boundary_elem, elements_in_proc.data() + element_index - starting_elements[i], &index_out_section);
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
        for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
            if (!mesh.section_is_domain_[k]) {
                if (n_boundaries_in_proc[k] > 0) {
                    boundary_index_end += n_boundaries_in_proc[k];
                    int index_out_section = 0;
                    const cgsize_t n_boundary_elem = 0; // No boundary elements yet
                    cg_section_write(index_out_file, index_out_base, index_out_zone[i], mesh.section_names_[k].data(), ElementType_t::BAR_2, boundary_index_start + 1, boundary_index_end, n_boundary_elem, boundaries_in_proc[k].data(), &index_out_section);
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
        for (int index_boundary = 0; index_boundary < mesh.boundary_sizes_.size(); ++index_boundary) {
            cgsize_t n_boundary_elements_in_proc = 0;
            for (cgsize_t j = 0; j < mesh.boundary_sizes_[index_boundary]; ++j) {
                int section_index = -1;
                for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
                    if ((mesh.boundary_elements_[index_boundary][j] >= mesh.section_ranges_[k][0]) && (mesh.boundary_elements_[index_boundary][j] <= mesh.section_ranges_[k][1])) {
                        section_index = k;
                        break;
                    }
                }
                if (section_index == -1) {
                    std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << ", boundary " << index_boundary << " contains element " << mesh.boundary_elements_[index_boundary][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(45);
                }

                const cgsize_t new_element_index = new_boundary_indices[section_index][mesh.boundary_elements_[index_boundary][j] - mesh.section_ranges_[section_index][0]];
                if (new_element_index != 0) {
                    ++n_boundary_elements_in_proc;
                }
            }

            if (n_boundary_elements_in_proc > 0) {
                std::vector<cgsize_t> boundary_conditions;
                boundary_conditions.reserve(n_boundary_elements_in_proc);

                for (cgsize_t j = 0; j < mesh.boundary_sizes_[index_boundary]; ++j) {
                    int section_index = -1;
                    for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
                        if ((mesh.boundary_elements_[index_boundary][j] >= mesh.section_ranges_[k][0]) && (mesh.boundary_elements_[index_boundary][j] <= mesh.section_ranges_[k][1])) {
                            section_index = k;
                            break;
                        }
                    }
                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << ", boundary " << index_boundary << " contains element " << mesh.boundary_elements_[index_boundary][j] << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(45);
                    }

                    const cgsize_t new_element_index = new_boundary_indices[section_index][mesh.boundary_elements_[index_boundary][j] - mesh.section_ranges_[section_index][0]];
                    if (new_element_index != 0) {
                        boundary_conditions.push_back(new_element_index + 1);
                    }
                }

                int index_out_boundary = 0;
                cg_boco_write(index_out_file, index_out_base, index_out_zone[i], mesh.boundary_names_[index_boundary].data(), mesh.boundary_types_[index_boundary], mesh.boundary_point_set_types_[index_boundary], n_boundary_elements_in_proc, boundary_conditions.data(), &index_out_boundary);
                cg_boco_gridlocation_write(index_out_file, index_out_base, index_out_zone[i], index_out_boundary, mesh.boundary_grid_locations_[index_boundary]);
            }
        }

        // Self interfaces
        for (int index_connectivity = 0; index_connectivity < mesh.connectivity_sizes_.size(); ++index_connectivity) {
            cgsize_t n_connectivity_elements_in_proc = 0;
            for (cgsize_t j = 0; j < mesh.connectivity_sizes_[index_connectivity]; ++j) {
                const cgsize_t element_index = mesh.interface_elements_[index_connectivity][j];
                const cgsize_t element_donor_index = mesh.interface_donor_elements_[index_connectivity][j];
                
                int section_index = -1;
                for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
                    if ((element_index >= mesh.section_ranges_[k][0]) && (element_index <= mesh.section_ranges_[k][1])) {
                        section_index = k;
                        break;
                    }
                }
                if (section_index == -1) {
                    std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << ", connectivity " << index_connectivity << " contains element " << element_index << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(46);
                }

                const cgsize_t new_element_index = new_boundary_indices[section_index][element_index - mesh.section_ranges_[section_index][0]];

                section_index = -1;
                for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
                    if ((element_donor_index >= mesh.section_ranges_[k][0]) && (element_donor_index <= mesh.section_ranges_[k][1])) {
                        section_index = k;
                        break;
                    }
                }
                if (section_index == -1) {
                    std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << ", connectivity " << index_connectivity << " contains donor element " << element_donor_index << " but it is not found in any mesh section. Exiting." << std::endl;
                    exit(47);
                }

                const cgsize_t new_element_donor_index = new_boundary_indices[section_index][element_donor_index - mesh.section_ranges_[section_index][0]];

                if (new_element_index != 0) {
                    if (new_element_donor_index != 0) {
                        ++n_connectivity_elements_in_proc;
                    }
                    else {
                        const cgsize_t domain_element = element_to_element[4 * mesh.n_elements_domain_ + element_index - mesh.n_elements_domain_ - 1];
                        const cgsize_t donor_domain_element = element_to_element[4 * mesh.n_elements_domain_ + element_donor_index - mesh.n_elements_domain_ - 1];

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
                            std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << " contains neighbour element " << donor_domain_element << " but it is not found in any process. Exiting." << std::endl;
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

                for (cgsize_t j = 0; j < mesh.connectivity_sizes_[index_connectivity]; ++j) {
                    const cgsize_t element_index = mesh.interface_elements_[index_connectivity][j];
                    const cgsize_t element_donor_index = mesh.interface_donor_elements_[index_connectivity][j];
                    
                    int section_index = -1;
                    for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
                        if ((element_index >= mesh.section_ranges_[k][0]) && (element_index <= mesh.section_ranges_[k][1])) {
                            section_index = k;
                            break;
                        }
                    }
                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << ", connectivity " << index_connectivity << " contains element " << element_index << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(46);
                    }

                    const cgsize_t new_element_index = new_boundary_indices[section_index][element_index - mesh.section_ranges_[section_index][0]];

                    section_index = -1;
                    for (int k = 0; k < mesh.section_is_domain_.size(); ++k) {
                        if ((element_donor_index >= mesh.section_ranges_[k][0]) && (element_donor_index <= mesh.section_ranges_[k][1])) {
                            section_index = k;
                            break;
                        }
                    }
                    if (section_index == -1) {
                        std::cerr << "Error: CGNS mesh, base " << mesh.index_in_base_ << ", zone " << mesh.index_in_zone_ << ", connectivity " << index_connectivity << " contains donor element " << element_donor_index << " but it is not found in any mesh section. Exiting." << std::endl;
                        exit(47);
                    }

                    const cgsize_t new_element_donor_index = new_boundary_indices[section_index][element_donor_index - mesh.section_ranges_[section_index][0]];

                    if (new_element_index != 0 && new_element_donor_index != 0) {
                        local_connectivity_elements.push_back(new_element_index + 1);
                        local_connectivity_donor_elements.push_back(new_element_donor_index + 1);
                    }
                }

                int index_out_connectivity = 0;
                cg_conn_write(index_out_file, index_out_base, index_out_zone[i], mesh.connectivity_names_[index_connectivity].data(), mesh.connectivity_grid_locations_[index_connectivity], mesh.connectivity_types_[index_connectivity], mesh.connectivity_point_set_types_[index_connectivity], n_connectivity_elements_in_proc, local_connectivity_elements.data(), ss.str().c_str(), mesh.connectivity_donor_zone_types_[index_connectivity], mesh.connectivity_donor_point_set_types_[index_connectivity], mesh.connectivity_donor_data_types_[index_connectivity], n_connectivity_elements_in_proc, local_connectivity_donor_elements.data(), &index_out_connectivity);
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
                ss2 << "Connectivity" << std::setfill('0') << std::setw(n_digits) << i + 1 << "to" << std::setfill('0') << std::setw(n_digits) << j + 1;
                std::stringstream ss3;
                ss3 << "Zone " << std::setfill('0') << std::setw(n_digits) << j + 1;
                cg_conn_write(index_out_file, index_out_base, index_out_zone[i], ss2.str().c_str(), GridLocation_t::FaceCenter, GridConnectivityType_t::Abutting1to1, PointSetType_t::PointList, connectivity_elements.size(), connectivity_elements.data(), ss3.str().c_str(), ZoneType_t::Unstructured, PointSetType_t::PointListDonor, DataType_t::Integer, connectivity_donor_elements.size(), connectivity_donor_elements.data(), &index_out_connectivity);
            }
        }
    }

    const int close_out_error = cg_close(index_out_file);
    if (close_out_error != CG_OK) {
        std::cerr << "Error: output file '" << out_file << "' could not be closed with error '" << cg_get_error() << "'. Exiting." << std::endl;
        exit(42);
    }
}

auto main(int argc, char* argv[]) -> int {
    const SEM::Helpers::InputParser_t input_parser(argc, argv);
    if (input_parser.cmdOptionExists("--help") || input_parser.cmdOptionExists("-h")) {
        std::cout << "usage: mesh_partitioner.exe [-h] [--in_path IN_PATH | [--in_filename IN_FILENAME] [--in_directory IN_DIRECTORY]] [--out_path OUT_PATH | [--out_filename OUT_FILENAME] [--out_directory OUT_DIRECTORY]] [--n N] [-v]" << std::endl << std::endl;
        std::cout << "Unstructured mesh partitioner. Generates multi-block 2D unstructured meshes in the CGNS HDF5 format from single-block meshes using the CGNS HDF5 or SU2 format." << std::endl << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << '\t' <<  "-h, --help"      <<  '\t' <<  "show this help message and exit" << std::endl;
        std::cout << '\t' <<  "--in_path"       <<  '\t' <<  "full path of the input mesh file, overrides filename and directory if set." << std::endl;
        std::cout << '\t' <<  "--in_filename"   <<  '\t' <<  "file name of the input mesh file, (default: mesh.cgns)" << std::endl;
        std::cout << '\t' <<  "--in_directory"  <<  '\t' <<  "directory of the input mesh file (default: ./meshes/)" << std::endl;
        std::cout << '\t' <<  "--out_path"      <<  '\t' <<  "full path of the output mesh file, overrides filename and directory if set" << std::endl;
        std::cout << '\t' <<  "--out_filename"  <<  '\t' <<  "file name of the output mesh file (default: mesh_partitioned.cgns)" << std::endl;
        std::cout << '\t' <<  "--out_directory" <<  '\t' <<  "directory of the output mesh file (default: ./meshes/)" << std::endl;
        std::cout << '\t' <<  "--n"             <<  '\t' <<  "number of blocks in the output mesh (default: 4)" << std::endl;
        std::cout << '\t' <<  "-v, --version"   <<  '\t' <<  "show program's version number and exit" << std::endl;
        exit(0);
    }
    else if (input_parser.cmdOptionExists("--version") || input_parser.cmdOptionExists("-v")) {
        std::cout << "mesh_partitioner.exe 1.0.0" << std::endl;
        exit(0);
    }

    // Argument parsing
    const fs::path in_file = get_input_file(input_parser);
    const fs::path out_file = get_output_file(input_parser);
    const cgsize_t n_proc = input_parser.getCmdOptionOr("--n", static_cast<cgsize_t>(4));

    if (in_file.extension().compare(".cgns") == 0) {
        write_cgns_mesh(n_proc, read_cgns_mesh(in_file), out_file);
    }
    else if (in_file.extension().compare(".su2") == 0) {
        write_cgns_mesh(n_proc, read_su2_mesh(in_file), out_file);
    }
    else {
        std::cerr << "Error: in file '" << in_file << "' has unknown extension '" << in_file.extension() << "'. Known extensions are: '.cgns', '.su2'. Exiting." << std::endl;
        exit(43);
    }

    return 0;
}