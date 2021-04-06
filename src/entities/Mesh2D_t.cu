#include "entities/Mesh2D_t.cuh"
#include "ChebyshevPolynomial_t.cuh"
#include "LegendrePolynomial_t.cuh"
#include "ProgressBar_t.h"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

SEM::Entities::Mesh2D_t::Mesh2D_t(std::filesystem::path filename, int initial_N, cudaStream_t &stream) :       
        stream_(stream) {

    
}

void SEM::Entities::Mesh2D_t::read_su2(std::filesystem::path filename){
    /*std::string line;
    std::string token;
    size_t value;

    std::ifstream meshfile(filename);
    if (!meshfile.is_open()) {
        std::cerr << "Error: file '" << filename << "' could not be opened. Exiting." << std::endl;
        exit(7);
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
        exit(8);
    }

    if (value != 2) {
        std::cerr << "Error: program only works for 2 dimensions, found '" << value << "'. Exiting." << std::endl;
        exit(9);
    }

    std::vector<Cell_t> farfield;
    std::vector<Cell_t> wall;
    std::vector<Cell_t> inlet;

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
            nodes_ = std::vector<Node_t>(value);

            for (size_t i = 0; i < nodes_.size(); ++i) {
                std::getline(meshfile, line);
                std::istringstream liness2(line);
                liness2 >> nodes_[i].pos_[0] >> nodes_[i].pos_[1];
            }
        }
        else if (token == "NELEM=") {
            liness >> value;
            cells_ = std::vector<Cell_t>(value);
            n_cells_ = value;

            for (size_t i = 0; i < cells_.size(); ++i) {
                int n_sides;
                size_t val[4];

                std::getline(meshfile, line);
                std::istringstream liness2(line);
                liness2 >> token;
                if (token == "9") {
                    n_sides = 4;
                    liness2 >> val[0] >> val[1] >> val[2] >> val[3];

                    cells_[i] = Cell_t(n_sides);
                    for (int j = 0; j < n_sides; ++j) {
                        cells_[i].nodes_[j] = val[j];
                    }
                }
                else if (token == "5") {
                    n_sides = 3;
                    liness2 >> val[0] >> val[1] >> val[2];

                    cells_[i] = Cell_t(n_sides);
                    for (int j = 0; j < n_sides; ++j) {
                        cells_[i].nodes_[j] = val[j];
                    }
                }
                else {
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

                    n_farfield_ += value;
                    farfield.reserve(n_farfield_);

                    for (size_t j = 0; j < value; ++j) {

                        std::getline(meshfile, line);
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
                        std::getline(meshfile, line);
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

                        std::getline(meshfile, line);
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
                        std::getline(meshfile, line);
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

                        std::getline(meshfile, line);
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
            if (!meshfile.eof()) {
                std::cerr << "Error: expected marker 'NPOIN=', 'NELEM=' or 'NMARK=', found '" << token << "'. Exiting." << std::endl;
                exit(13);
            }
        }
    }

    meshfile.close();

    cells_.insert(std::end(cells_), std::begin(farfield), std::end(farfield));
    cells_.insert(std::end(cells_), std::begin(wall), std::end(wall));
    cells_.insert(std::end(cells_), std::begin(inlet), std::end(inlet));*/
}

void SEM::Entities::Mesh2D_t::set_initial_conditions(const deviceFloat* nodes) {

}

void SEM::Entities::Mesh2D_t::print() {
    std::vector<Face_t> host_faces(N_faces_);
    std::vector<Element_t> host_elements(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_);
    std::vector<size_t> host_local_boundary_to_element(N_local_boundaries_);

    cudaMemcpy(host_faces.data(), faces_, N_faces_ * sizeof(Face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_elements.data(), elements_, (N_elements_ + N_local_boundaries_ + N_MPI_boundaries_) * sizeof(Element_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_local_boundary_to_element.data(), local_boundary_to_element_, N_local_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_to_element_.data(), MPI_boundary_to_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_MPI_boundary_from_element_.data(), MPI_boundary_from_element_, N_MPI_boundaries_ * sizeof(size_t), cudaMemcpyDeviceToHost);

    // Invalidate GPU pointers, or else they will be deleted on the CPU, where they point to random stuff
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        host_elements[i].phi_ = nullptr;
        host_elements[i].q_ = nullptr;
        host_elements[i].ux_ = nullptr;
        host_elements[i].phi_prime_ = nullptr;
        host_elements[i].intermediate_ = nullptr;
    }

    std::cout << "N elements global: " << N_elements_global_ << std::endl;
    std::cout << "N elements local: " << N_elements_ << std::endl;
    std::cout << "N faces: " << N_faces_ << std::endl;
    std::cout << "N local boundaries: " << N_local_boundaries_ << std::endl;
    std::cout << "N MPI boundaries: " << N_MPI_boundaries_ << std::endl;
    std::cout << "Global element offset: " << global_element_offset_ << std::endl;
    std::cout << "Number of elements per process: " << N_elements_per_process_ << std::endl;
    std::cout << "Initial N: " << initial_N_ << std::endl;

    std::cout << std::endl << "Phi interpolated: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].phi_L_ << " ";
        std::cout << host_elements[i].phi_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime interpolated: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].phi_prime_L_ << " ";
        std::cout << host_elements[i].phi_prime_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "x: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].x_[0] << " ";
        std::cout << host_elements[i].x_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "Neighbouring faces: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].faces_[0] << " ";
        std::cout << host_elements[i].faces_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "N: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].N_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "delta x: " << std::endl;
    for (size_t i = 0; i < N_elements_ + N_local_boundaries_ + N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_elements[i].delta_x_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "Fluxes: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].flux_ << std::endl;
    }

    std::cout << std::endl << "Derivative fluxes: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].derivative_flux_ << std::endl;
    }

    std::cout << std::endl << "Non linear fluxes: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].nl_flux_ << std::endl;
    }

    std::cout << std::endl << "Elements: " << std::endl;
    for (size_t i = 0; i < N_faces_; ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << host_faces[i].elements_[0] << " ";
        std::cout << host_faces[i].elements_[1] << std::endl;
    }

    std::cout << std::endl << "Local boundaries elements: " << std::endl;
    for (size_t i = 0; i < N_local_boundaries_; ++i) {
        std::cout << '\t' << "Local boundary " << i << ": ";
        std::cout << '\t';
        std::cout << host_local_boundary_to_element[i] << std::endl;
    }

    std::cout << std::endl << "MPI boundaries to elements: " << std::endl;
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "MPI boundary " << i << ": ";
        std::cout << '\t';
        std::cout << host_MPI_boundary_to_element_[N_local_boundaries_ + i] << std::endl;
    }

    std::cout << std::endl << "MPI boundaries from elements: " << std::endl;
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "MPI boundary " << i << ": ";
        std::cout << '\t';
        std::cout << host_MPI_boundary_from_element_[N_local_boundaries_ + i] << std::endl;
    }
    std::cout << std::endl;
}

void SEM::Entities::Mesh2D_t::write_file_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& coordinates, const std::vector<deviceFloat>& velocity, const std::vector<deviceFloat>& du_dx, const std::vector<deviceFloat>& intermediate, const std::vector<deviceFloat>& x_L, const std::vector<deviceFloat>& x_R, const std::vector<int>& N, const std::vector<deviceFloat>& sigma, const bool* refine, const bool* coarsen, const std::vector<deviceFloat>& error) {
    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    std::stringstream ss;
    std::ofstream file;
    ss << "output_t" << std::setprecision(9) << std::fixed << time << "_proc" << std::setfill('0') << std::setw(6) << rank << ".dat";
    file.open(save_dir / ss.str());

    file << "TITLE = \"Velocity at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\", \"U_x_prime\", \"intermediate\"" << std::endl;

    for (size_t i = 0; i < N_elements; ++i) {
        file << "ZONE T= \"Zone " << i + 1 << "\",  I= " << N_interpolation_points << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

        for (size_t j = 0; j < N_interpolation_points; ++j) {
            file       << std::setw(12) << coordinates[i*N_interpolation_points + j] 
                << " " << std::setw(12) << velocity[i*N_interpolation_points + j]
                << " " << std::setw(12) << du_dx[i*N_interpolation_points + j]
                << " " << std::setw(12) << intermediate[i*N_interpolation_points + j] << std::endl;
        }
    }

    file.close();

    std::stringstream ss_element;
    std::ofstream file_element;
    ss_element << "output_element_t" << std::setprecision(9) << std::fixed << time << "_proc" << std::setfill('0') << std::setw(6) << rank << ".dat";
    file_element.open(save_dir / ss_element.str());

    file_element << "TITLE = \"Element values at t= " << time << "\"" << std::endl
                 << "VARIABLES = \"X\", \"X_L\", \"X_R\", \"N\", \"sigma\", \"refine\", \"coarsen\", \"error\"" << std::endl
                 << "ZONE T= \"Zone     1\",  I= " << N_elements << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (size_t j = 0; j < N_elements; ++j) {
        file_element << std::setw(12) << (x_L[j] + x_R[j]) * 0.5
              << " " << std::setw(12) << x_L[j]
              << " " << std::setw(12) << x_R[j]
              << " " << std::setw(12) << N[j]
              << " " << std::setw(12) << sigma[j]
              << " " << std::setw(12) << refine[j]
              << " " << std::setw(12) << coarsen[j]
              << " " << std::setw(12) << error[j] << std::endl;
    }

    file_element.close();
}

void SEM::Entities::Mesh2D_t::write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices) {
    deviceFloat* x;
    deviceFloat* phi;
    deviceFloat* phi_prime;
    deviceFloat* intermediate;
    deviceFloat* x_L;
    deviceFloat* x_R;
    int* N;
    deviceFloat* sigma;
    bool* refine;
    bool* coarsen;
    deviceFloat* error;
    std::vector<deviceFloat> host_x(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_phi(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_phi_prime(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_intermediate(N_elements_ * N_interpolation_points);
    std::vector<deviceFloat> host_x_L(N_elements_);
    std::vector<deviceFloat> host_x_R(N_elements_);
    std::vector<int> host_N(N_elements_);
    std::vector<deviceFloat> host_sigma(N_elements_);
    bool* host_refine = new bool[N_elements_]; // Vectors of bools can be messed-up by some implementations
    bool* host_coarsen = new bool[N_elements_]; // Like they won't be an array of bools but packed in integers, in which case getting them from Cuda will fail.
    std::vector<deviceFloat> host_error(N_elements_);
    cudaMalloc(&x, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&phi_prime, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&intermediate, N_elements_ * N_interpolation_points * sizeof(deviceFloat));
    cudaMalloc(&x_L, N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&x_R, N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&N, N_elements_ * sizeof(int));
    cudaMalloc(&sigma, N_elements_ * sizeof(deviceFloat));
    cudaMalloc(&refine, N_elements_ * sizeof(bool));
    cudaMalloc(&coarsen, N_elements_ * sizeof(bool));
    cudaMalloc(&error, N_elements_ * sizeof(deviceFloat));

    SEM::get_solution<<<elements_numBlocks_, elements_blockSize_, 0, stream_>>>(N_elements_, N_interpolation_points, elements_, interpolation_matrices, x, phi, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);
    
    cudaMemcpy(host_x.data(), x , N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi.data(), phi, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime.data(), phi_prime, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_intermediate.data(), intermediate, N_elements_ * N_interpolation_points * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_L.data(), x_L, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_x_R.data(), x_R, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_N.data(), N, N_elements_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sigma.data(), sigma, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_refine, refine, N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_coarsen, coarsen, N_elements_ * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_error.data(), error, N_elements_ * sizeof(deviceFloat), cudaMemcpyDeviceToHost);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    write_file_data(N_interpolation_points, N_elements_, time, global_rank, host_x, host_phi, host_phi_prime, host_intermediate, host_x_L, host_x_R, host_N, host_sigma, host_refine, host_coarsen, host_error);

    delete[] host_refine;
    delete[] host_coarsen;
    cudaFree(x);
    cudaFree(phi);
    cudaFree(phi_prime);
    cudaFree(intermediate);
    cudaFree(x_L);
    cudaFree(x_R);
    cudaFree(N);
    cudaFree(sigma);
    cudaFree(refine);
    cudaFree(coarsen);
    cudaFree(error);
}

template void SEM::Entities::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<ChebyshevPolynomial_t> &NDG, deviceFloat viscosity); // Get with the times c++, it's crazy I have to do this
template void SEM::Entities::Mesh2D_t::solve(const deviceFloat delta_t, const std::vector<deviceFloat> output_times, const NDG_t<LegendrePolynomial_t> &NDG, deviceFloat viscosity);

template<typename Polynomial>
void SEM::Entities::Mesh2D_t::solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG, deviceFloat viscosity) {
    
}

deviceFloat SEM::Entities::Mesh2D_t::get_delta_t(const deviceFloat CFL) {   
    return 0.0;
}

void SEM::Entities::Mesh2D_t::adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights) {
    
}

void SEM::Entities::Mesh2D_t::boundary_conditions() {
    
}
