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
