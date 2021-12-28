#include "meshes/Mesh_t.h"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

constexpr hostFloat pi = 3.14159265358979323846;

SEM::Host::Meshes::Mesh_t::Mesh_t(size_t N_elements, int initial_N, hostFloat delta_x_min, hostFloat x_min, hostFloat x_max, int adaptivity_interval) : 
        N_elements_global_(N_elements),
        delta_x_min_(delta_x_min),
        adaptivity_interval_(adaptivity_interval),
        initial_N_(initial_N) {
    // CHECK N_faces = N_elements only for periodic BC.

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    N_elements_per_process_ = (N_elements_global_ + global_size - 1)/global_size;
    N_elements_ = (global_rank == global_size - 1) ? N_elements_per_process_ + N_elements_global_ - N_elements_per_process_ * global_size : N_elements_per_process_;
    if (N_elements_ == N_elements_global_) {
        N_local_boundaries_ = 2;
        N_MPI_boundaries_ = 0;
    }
    else {
        N_local_boundaries_ = 0;
        N_MPI_boundaries_ = 2;
    }

    const size_t N_faces = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 1; 
    global_element_offset_ = global_rank * N_elements_per_process_;

    faces_ = std::vector<SEM::Host::Entities::Face_t>(N_faces);
    elements_ = std::vector<SEM::Host::Entities::Element_t>(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_);

    local_boundary_to_element_ = std::vector<size_t>(N_local_boundaries_);
    MPI_boundary_to_element_ = std::vector<size_t>(N_MPI_boundaries_);
    MPI_boundary_from_element_ = std::vector<size_t>(N_MPI_boundaries_);
    send_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    receive_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    requests_ = std::vector<MPI_Request>(N_MPI_boundaries_*2);
    statuses_ = std::vector<MPI_Status>(N_MPI_boundaries_*2);
    refine_array_ = std::vector<size_t>(N_elements_);

    const hostFloat delta_x = (x_max - x_min)/N_elements_global_;
    const hostFloat x_min_local = x_min + delta_x * global_rank * N_elements_per_process_;
    const hostFloat x_max_local = x_min_local + N_elements_ * delta_x;
    
    build_elements(x_min_local, x_max_local);
    build_boundaries();
    build_faces(); // CHECK
}

void SEM::Host::Meshes::Mesh_t::set_initial_conditions(const std::vector<std::vector<hostFloat>>& nodes) {
    for (size_t i = 0; i < N_elements_; ++i) {
        for (int j = 0; j <= elements_[i].N_; ++j) {
            const hostFloat x = (0.5 + nodes[elements_[i].N_][j]/2.0f) * (elements_[i].x_[1] - elements_[i].x_[0]) + elements_[i].x_[0];
            elements_[i].phi_[j] = g(x);
        }
    }
}

void SEM::Host::Meshes::Mesh_t::print() {
    std::cout << "N elements global: " << N_elements_global_ << std::endl;
    std::cout << "N elements local: " << N_elements_ << std::endl;
    std::cout << "N faces: " << faces_.size() << std::endl;
    std::cout << "N local boundaries: " << N_local_boundaries_ << std::endl;
    std::cout << "N MPI boundaries: " << N_MPI_boundaries_ << std::endl;
    std::cout << "Global element offset: " << global_element_offset_ << std::endl;
    std::cout << "Number of elements per process: " << N_elements_per_process_ << std::endl;
    std::cout << "Initial N: " << initial_N_ << std::endl;

    std::cout << std::endl << "Phi: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (auto phi: elements_[i].phi_) {
            std::cout << phi << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (auto phi_prime: elements_[i].phi_prime_) {
            std::cout << phi_prime << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi interpolated: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << elements_[i].phi_L_ << " ";
        std::cout << elements_[i].phi_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime interpolated: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << elements_[i].phi_prime_L_ << " ";
        std::cout << elements_[i].phi_prime_R_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "x: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << elements_[i].x_[0] << " ";
        std::cout << elements_[i].x_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "Neighbouring faces: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << elements_[i].faces_[0] << " ";
        std::cout << elements_[i].faces_[1];
        std::cout << std::endl;
    }

    std::cout << std::endl << "N: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << elements_[i].N_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "delta x: " << std::endl;
    for (size_t i = 0; i < elements_.size(); ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << elements_[i].delta_x_;
        std::cout << std::endl;
    }

    std::cout << std::endl << "Fluxes: " << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << faces_[i].flux_ << std::endl;
    }

    std::cout << std::endl << "Derivative fluxes: " << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << faces_[i].derivative_flux_ << std::endl;
    }

    std::cout << std::endl << "Non linear fluxes: " << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << faces_[i].nl_flux_ << std::endl;
    }

    std::cout << std::endl << "Elements: " << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << faces_[i].elements_[0] << " ";
        std::cout << faces_[i].elements_[1] << std::endl;
    }

    std::cout << std::endl << "Local boundaries elements: " << std::endl;
    for (size_t i = 0; i < N_local_boundaries_; ++i) {
        std::cout << '\t' << "Local boundary " << i << ": ";
        std::cout << '\t';
        std::cout << local_boundary_to_element_[i] << std::endl;
    }

    std::cout << std::endl << "MPI boundaries to elements: " << std::endl;
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "MPI boundary " << i << ": ";
        std::cout << '\t';
        std::cout << MPI_boundary_to_element_[N_local_boundaries_ + i] << std::endl;
    }

    std::cout << std::endl << "MPI boundaries from elements: " << std::endl;
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        std::cout << '\t' << "MPI boundary " << i << ": ";
        std::cout << '\t';
        std::cout << MPI_boundary_from_element_[N_local_boundaries_ + i] << std::endl;
    }
    std::cout << std::endl;
}

void SEM::Host::Meshes::Mesh_t::write_file_data(size_t n_interpolation_points, size_t N_elements, hostFloat time, int rank, const std::vector<hostFloat>& velocity, const std::vector<hostFloat>& coordinates, const std::vector<hostFloat>& du_dx, const std::vector<hostFloat>& intermediate, const std::vector<hostFloat>& x_L, const std::vector<hostFloat>& x_R, const std::vector<int>& N, const std::vector<hostFloat>& sigma, const std::vector<bool>& refine, const std::vector<bool>& coarsen, const std::vector<hostFloat>& error) {
    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    std::stringstream ss;
    std::ofstream file;
    ss << "output_t" << std::setprecision(9) << std::fixed << time << "_proc" << std::setfill('0') << std::setw(6) << rank << ".dat";
    file.open(save_dir / ss.str());

    file << "TITLE = \"Velocity at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\", \"U_x_prime\", \"intermediate\"" << std::endl;

    for (size_t i = 0; i < N_elements; ++i) {
        file << "ZONE T= \"Zone " << i + 1 << "\",  I= " << n_interpolation_points << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

        for (size_t j = 0; j < n_interpolation_points; ++j) {
            file       << std::setw(12) << coordinates[i*n_interpolation_points + j] 
                << " " << std::setw(12) << velocity[i*n_interpolation_points + j]
                << " " << std::setw(12) << du_dx[i*n_interpolation_points + j]
                << " " << std::setw(12) << intermediate[i*n_interpolation_points + j] << std::endl;
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

void SEM::Host::Meshes::Mesh_t::write_data(hostFloat time, size_t n_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices) {
    std::vector<hostFloat> phi(N_elements_ * n_interpolation_points);
    std::vector<hostFloat> x(N_elements_ * n_interpolation_points);
    std::vector<hostFloat> phi_prime(N_elements_ * n_interpolation_points);
    std::vector<hostFloat> intermediate(N_elements_ * n_interpolation_points);
    std::vector<hostFloat> x_L(N_elements_);
    std::vector<hostFloat> x_R(N_elements_);
    std::vector<int> N(N_elements_);
    std::vector<hostFloat> sigma(N_elements_);
    std::vector<bool> refine(N_elements_);
    std::vector<bool> coarsen(N_elements_);
    std::vector<hostFloat> error(N_elements_);

    get_solution(n_interpolation_points, interpolation_matrices, phi, x, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    write_file_data(n_interpolation_points, N_elements_, time, global_rank, phi, x, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);
}

void SEM::Host::Meshes::Mesh_t::build_elements(hostFloat x_min, hostFloat x_max) {
    for (size_t i = 0; i < N_elements_; ++i) {
        const size_t face_L = i;
        const size_t face_R = i + 1;
        const hostFloat delta_x = (x_max - x_min)/N_elements_;
        const hostFloat element_x_min = x_min + i * delta_x;
        const hostFloat element_x_max = x_min + (i + 1) * delta_x;

        elements_[i] = SEM::Host::Entities::Element_t(initial_N_, face_L, face_R, element_x_min, element_x_max);
    }
}

void SEM::Host::Meshes::Mesh_t::build_boundaries() {
    for (int i = 0; i < N_local_boundaries_; ++i) {
        size_t face_L;
        size_t face_R;
        hostFloat element_x_min;
        hostFloat element_x_max;

        if (i == 0) { // CHECK this is hardcoded for 1D
            face_L = 0;
            face_R = 0;
            element_x_min = elements_[0].x_[0];
            element_x_max = elements_[0].x_[0];
            local_boundary_to_element_[i] = N_elements_ - 1;
        }
        else if (i == 1) {
            face_L = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            face_R = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            element_x_min = elements_[N_elements_ - 1].x_[1];
            element_x_max = elements_[N_elements_ - 1].x_[1];
            local_boundary_to_element_[i] = 0;
        }

        elements_[N_elements_ + i] = SEM::Host::Entities::Element_t(0, face_L, face_R, element_x_min, element_x_max);
    }

    for (int i = 0; i < N_MPI_boundaries_; ++i) {
        size_t face_L;
        size_t face_R;
        hostFloat element_x_min;
        hostFloat element_x_max;

        if (i == 0) { // CHECK this is hardcoded for 1D
            face_L = 0;
            face_R = 0;
            element_x_min = elements_[0].x_[0];
            element_x_max = elements_[0].x_[0];
            MPI_boundary_to_element_[i] = (global_element_offset_ == 0) ? N_elements_global_ - 1 : global_element_offset_ - 1;
            MPI_boundary_from_element_[i] = global_element_offset_;
        }
        else if (i == 1) {
            face_L = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            face_R = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            element_x_min = elements_[N_elements_ - 1].x_[1];
            element_x_max = elements_[N_elements_ - 1].x_[1];
            MPI_boundary_to_element_[i] = (global_element_offset_ + N_elements_ == N_elements_global_) ? 0 : global_element_offset_ + N_elements_;
            MPI_boundary_from_element_[i] = global_element_offset_ + N_elements_ - 1;
        }

        elements_[N_elements_ + N_local_boundaries_ + i] = SEM::Host::Entities::Element_t(0, face_L, face_R, element_x_min, element_x_max);
    }
}

void SEM::Host::Meshes::Mesh_t::adjust_boundaries() {
    for (int i = 0; i < N_MPI_boundaries_; ++i) {
        if (i == 0) { // CHECK this is hardcoded for 1D
            MPI_boundary_to_element_[i] = (global_element_offset_ == 0) ? N_elements_global_ - 1 : global_element_offset_ - 1;
            MPI_boundary_from_element_[i] = global_element_offset_;
        }
        else if (i == 1) {
            MPI_boundary_to_element_[i] = (global_element_offset_ + N_elements_ == N_elements_global_) ? 0 : global_element_offset_ + N_elements_;
            MPI_boundary_from_element_[i] = global_element_offset_ + N_elements_ - 1;
        }
    }
}

void SEM::Host::Meshes::Mesh_t::build_faces() {
    for (size_t i = 0; i < faces_.size(); ++i) {
        const size_t neighbour_L = (i > 0) ? i - 1 : faces_.size() - 1;
        const size_t neighbour_R = (i < faces_.size() - 1) ? i : faces_.size(); // Last face links last element to first element
        faces_[i] = SEM::Host::Entities::Face_t(neighbour_L, neighbour_R);
    }
}

hostFloat SEM::Host::Meshes::Mesh_t::g(hostFloat x) {
    //return (x < -0.2f || x > 0.2f) ? 0.2f : 0.8f;
    return -std::sin(pi * x);
}

void SEM::Host::Meshes::Mesh_t::get_solution(size_t n_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<hostFloat>& phi, std::vector<hostFloat>& x, std::vector<hostFloat>& phi_prime, std::vector<hostFloat>& intermediate, std::vector<hostFloat>& x_L, std::vector<hostFloat>& x_R, std::vector<int>& N, std::vector<hostFloat>& sigma, std::vector<bool>& refine, std::vector<bool>& coarsen, std::vector<hostFloat>& error) {
    for (size_t i = 0; i < N_elements_; ++i) {
        const size_t offset_interp_1D = i * n_interpolation_points;
        const size_t step = n_interpolation_points/(elements_[i].N_ + 1);

        for (size_t j = 0; j < n_interpolation_points; ++j) {
            phi[offset_interp_1D + j] = 0.0f;
            phi_prime[offset_interp_1D + j] = 0.0f;
            for (int k = 0; k <= elements_[i].N_; ++k) {
                phi[offset_interp_1D + j] += interpolation_matrices[elements_[i].N_][j * (elements_[i].N_ + 1) + k] * elements_[i].phi_[k];
                phi_prime[offset_interp_1D + j] += interpolation_matrices[elements_[i].N_][j * (elements_[i].N_ + 1) + k] * elements_[i].phi_prime_[k];
            }
            intermediate[offset_interp_1D + j] = elements_[i].intermediate_[std::min(static_cast<int>(j/step), elements_[i].N_)];
            x[offset_interp_1D + j] = j * (elements_[i].x_[1] - elements_[i].x_[0]) / (n_interpolation_points - 1) + elements_[i].x_[0];
        }

        x_L[i] = elements_[i].x_[0];
        x_R[i] = elements_[i].x_[1];
        N[i] = elements_[i].N_;
        sigma[i] = elements_[i].sigma_;
        refine[i] = elements_[i].refine_;
        coarsen[i] = elements_[i].coarsen_;
        error[i] = elements_[i].error_;
    }
}

void SEM::Host::Meshes::Mesh_t::rk3_first_step(hostFloat delta_t, hostFloat g) {
    for (size_t i = 0; i < N_elements_; ++i) {
        for (int j = 0; j <= elements_[i].N_; ++j){
            elements_[i].intermediate_[j] = elements_[i].phi_prime_[j];
            elements_[i].phi_[j] += g * delta_t * elements_[i].intermediate_[j];
        }
    }
}

void SEM::Host::Meshes::Mesh_t::rk3_step(hostFloat delta_t, hostFloat a, hostFloat g) {
    for (size_t i = 0; i < N_elements_; ++i) {
        for (int j = 0; j <= elements_[i].N_; ++j){
            elements_[i].intermediate_[j] = a * elements_[i].intermediate_[j] + elements_[i].phi_prime_[j];
            elements_[i].phi_[j] += g * delta_t * elements_[i].intermediate_[j];
        }
    }
}

hostFloat SEM::Host::Meshes::Mesh_t::get_delta_t(const hostFloat CFL) {   
    double delta_t_min_local = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N_elements_; ++i) {
        hostFloat phi_max = 0.0;
        for (int j = 0; j <= elements_[i].N_; ++j) {
            phi_max = std::max(phi_max, std::abs(elements_[i].phi_[j]));
        }
        const double delta_t = CFL * elements_[i].delta_x_ * elements_[i].delta_x_/(phi_max * elements_[i].N_ * elements_[i].N_);
        delta_t_min_local = std::min(delta_t_min_local, delta_t);
    }

    double delta_t_min;
    MPI_Allreduce(&delta_t_min_local, &delta_t_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    return delta_t_min;
}

void SEM::Host::Meshes::Mesh_t::boundary_conditions() {
    local_boundaries();
    get_MPI_boundaries();
    
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        const int destination = MPI_boundary_to_element_[i]/N_elements_per_process_;

        MPI_Irecv(&receive_buffers_[i][0], 4, MPI_DOUBLE, destination, MPI_boundary_from_element_[i], MPI_COMM_WORLD, &requests_[i]);
        MPI_Isend(&send_buffers_[i][0], 4, MPI_DOUBLE, destination, MPI_boundary_to_element_[i], MPI_COMM_WORLD, &requests_[i + N_MPI_boundaries_]);
    }

    MPI_Waitall(N_MPI_boundaries_, requests_.data(), statuses_.data()); // CHECK maybe MPI barrier?

    put_MPI_boundaries();
}

void SEM::Host::Meshes::Mesh_t::local_boundaries() {
    for (size_t i = 0; i < N_local_boundaries_; ++i) {
        elements_[N_elements_ + i].phi_L_ = elements_[local_boundary_to_element_[i]].phi_L_;
        elements_[N_elements_ + i].phi_R_ = elements_[local_boundary_to_element_[i]].phi_R_;
        elements_[N_elements_ + i].phi_prime_L_ = elements_[local_boundary_to_element_[i]].phi_prime_L_;
        elements_[N_elements_ + i].phi_prime_R_ = elements_[local_boundary_to_element_[i]].phi_prime_R_;
    }
}

void SEM::Host::Meshes::Mesh_t::get_MPI_boundaries() {
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        const SEM::Host::Entities::Element_t& boundary_element = elements_[N_elements_ + N_local_boundaries_ + i];
        const SEM::Host::Entities::Face_t& boundary_face = faces_[boundary_element.faces_[0]];
        const SEM::Host::Entities::Element_t& domain_element = elements_[boundary_face.elements_[boundary_face.elements_[0] == N_elements_ + N_local_boundaries_ + i]];
        
        send_buffers_[i] = {domain_element.phi_L_,
                            domain_element.phi_R_,
                            domain_element.phi_prime_L_,
                            domain_element.phi_prime_R_};
    }
}

void SEM::Host::Meshes::Mesh_t::put_MPI_boundaries() {
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        elements_[N_elements_ + N_local_boundaries_ + i].phi_L_ = receive_buffers_[i][0];
        elements_[N_elements_ + N_local_boundaries_ + i].phi_R_ = receive_buffers_[i][1];
        elements_[N_elements_ + N_local_boundaries_ + i].phi_prime_L_ = receive_buffers_[i][2];
        elements_[N_elements_ + N_local_boundaries_ + i].phi_prime_R_ = receive_buffers_[i][3];
    }
}

void  SEM::Host::Meshes::Mesh_t::move_elements(size_t N_elements, std::vector<SEM::Host::Entities::Element_t>& temp_elements, size_t source_start_index, size_t destination_start_index) {
    for (size_t i = 0; i < N_elements; ++i) {
        elements_[i + destination_start_index] = std::move(temp_elements[i + source_start_index]);

        elements_[i + destination_start_index].faces_[0] = elements_[i + destination_start_index].faces_[0] + destination_start_index - source_start_index;
        elements_[i + destination_start_index].faces_[1] = elements_[i + destination_start_index].faces_[1] + destination_start_index - source_start_index;
    }
}

void SEM::Host::Meshes::Mesh_t::calculate_fluxes() {
    for (auto& face: faces_) {
        hostFloat u;
        const hostFloat u_left = elements_[face.elements_[0]].phi_R_;
        const hostFloat u_right = elements_[face.elements_[1]].phi_L_;

        if (u_left < 0.0 && u_right > 0.0) { // In expansion fan
            u = 0.5 * (u_left + u_right);
        }
        else if (u_left >= u_right) { // Shock
            if (u_left > 0.0) {
                u = u_left;
            }
            else {
                u = u_right;
            }
        }
        else { // Expansion fan
            if (u_left > 0.0) {
                u = u_left;
            }
            else  {
                u = u_right;
            }
        }

        face.flux_ = u_right;
        face.nl_flux_ = 0.5f * u * u;
    }
}

void SEM::Host::Meshes::Mesh_t::calculate_q_fluxes() {
    for (auto& face: faces_) {
        const hostFloat u_prime_left = elements_[face.elements_[0]].phi_prime_R_;

        face.derivative_flux_ = u_prime_left;
    }
}

void SEM::Host::Meshes::Mesh_t::adapt(int N_max, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) {
    unsigned long long additional_elements = 0;
    for (size_t i = 0; i < N_elements_; ++i) {
        additional_elements += elements_[i].refine_ * (elements_[i].sigma_ < 1.0) * (elements_[i].delta_x_/2 >= delta_x_min_);
        refine_array_[i] = additional_elements - elements_[i].refine_ * (elements_[i].sigma_ < 1.0) * (elements_[i].delta_x_/2 >= delta_x_min_); // Current offset
    }

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    std::vector<unsigned long long> additional_elements_global(global_size);
    MPI_Allgather(&additional_elements, 1, MPI_UNSIGNED_LONG_LONG, additional_elements_global.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    size_t N_additional_elements_previous = 0;
    for (int i = 0; i < global_rank; ++i) {
        N_additional_elements_previous += additional_elements_global[i];
    }
    const size_t global_element_offset_current = global_element_offset_ + N_additional_elements_previous;
    size_t N_additional_elements_global = 0;
    for (int i = 0; i < global_size; ++i) {
        N_additional_elements_global += additional_elements_global[i];
    }
    N_elements_global_ += N_additional_elements_global;
    const size_t global_element_offset_end_current = global_element_offset_current + N_elements_ + additional_elements - 1;

    const size_t N_elements_per_process_old = N_elements_per_process_;
    N_elements_per_process_ = (N_elements_global_ + global_size - 1)/global_size;
    global_element_offset_ = global_rank * N_elements_per_process_;
    const size_t global_element_offset_end = std::min(global_element_offset_ + N_elements_per_process_ - 1, N_elements_global_ - 1);

    if ((additional_elements == 0) && (global_element_offset_ == global_element_offset_current) && (global_element_offset_end == global_element_offset_end_current)) {
        p_adapt(N_max, nodes, barycentric_weights);

        if (N_additional_elements_previous > 0 || ((global_element_offset_ == 0) && (N_additional_elements_global > 0))) {
            adjust_boundaries();
        }
        return;
    }

    std::vector<SEM::Host::Entities::Element_t> new_elements(N_elements_ + additional_elements);
    hp_adapt(N_max, new_elements, nodes, barycentric_weights);

    const size_t N_elements_old = N_elements_;
    N_elements_ = (global_rank == global_size - 1) ? N_elements_per_process_ + N_elements_global_ - N_elements_per_process_ * global_size : N_elements_per_process_;
    const size_t N_faces = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 1;

    faces_ = std::vector<SEM::Host::Entities::Face_t>(N_faces);
    build_faces();
    elements_ = std::vector<SEM::Host::Entities::Element_t>(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_);

    const size_t N_elements_send_left = (global_element_offset_ > global_element_offset_current) ? global_element_offset_ - global_element_offset_current : 0;
    const size_t N_elements_recv_left = (global_element_offset_current > global_element_offset_) ? global_element_offset_current - global_element_offset_ : 0;
    const size_t N_elements_send_right = (global_element_offset_end_current > global_element_offset_end) ? global_element_offset_end_current - global_element_offset_end : 0;
    const size_t N_elements_recv_right = (global_element_offset_end > global_element_offset_end_current) ? global_element_offset_end - global_element_offset_end_current : 0;

    if (N_elements_send_left + N_elements_recv_left + N_elements_send_right + N_elements_recv_right > 0) {
        std::vector<int> left_origins(N_elements_recv_left);
        std::vector<int> right_origins(N_elements_recv_right);
        for (int i = 0; i < N_elements_recv_left; ++i) {
            const int index = global_element_offset_ + i;
            int process_end_index = -1;
            for (int rank = 0; rank < global_rank; ++rank) { 
                process_end_index += N_elements_per_process_old + additional_elements_global[rank];
                if (process_end_index >= index) {
                    left_origins[i] = rank;
                    break;
                }
            }
        }
        for (int i = 0; i < N_elements_recv_right; ++i) {
            const int index = global_element_offset_end_current + i + 1;
            int process_start_index = 0;
            for (int rank = 1; rank < global_size; ++rank) { 
                process_start_index += N_elements_per_process_old + additional_elements_global[rank - 1];
                if (process_start_index >= index) {
                    right_origins[i] = rank;
                    break;
                }
            }
        }
        
        std::vector<MPI_Request> adaptivity_requests(3 * (N_elements_send_left + N_elements_recv_left + N_elements_send_right + N_elements_recv_right));
        std::vector<MPI_Status> adaptivity_statuses(3 * (N_elements_send_left + N_elements_recv_left + N_elements_send_right + N_elements_recv_right));
        const MPI_Datatype data_type = (sizeof(hostFloat) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

        for (int i = 0; i < N_elements_send_left; ++i) {
            const int index = global_element_offset_current + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&new_elements[i].N_, 1, MPI_INT, destination, 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right]);
        }

        for (int i = 0; i < N_elements_send_right; ++i) {
            const int index = global_element_offset_end + 1 + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&new_elements[new_elements.size() - N_elements_send_right + i].N_, 1, MPI_INT, destination, 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + N_elements_send_left]);
        }

        for (int i = 0; i < N_elements_recv_left; ++i) {
            const int index = global_element_offset_ + i;

            MPI_Irecv(&elements_[i].N_, 1, MPI_INT, left_origins[i], 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i]);
        }

        for (int i = 0; i < N_elements_recv_right; ++i) {
            const int index = global_element_offset_end_current + i + 1;

            MPI_Irecv(&elements_[elements_.size() - N_elements_recv_right + i].N_, 1, MPI_INT, right_origins[i], 3 * index, MPI_COMM_WORLD, &adaptivity_requests[i + N_elements_recv_left]);
        }

        MPI_Waitall(N_elements_recv_left + N_elements_recv_right, adaptivity_requests.data(), adaptivity_statuses.data());

        for (int i = 0; i < N_elements_recv_left; ++i) {
            elements_[i].phi_ = std::vector<hostFloat>(elements_[i].N_ + 1);
            elements_[i].q_ = std::vector<hostFloat>(elements_[i].N_ + 1);
            elements_[i].ux_ = std::vector<hostFloat>(elements_[i].N_ + 1);
            elements_[i].phi_prime_ = std::vector<hostFloat>(elements_[i].N_ + 1);
            elements_[i].intermediate_ = std::vector<hostFloat>(elements_[i].N_ + 1);
        }

        for (int i = 0; i < N_elements_recv_right; ++i) {
            const size_t index = elements_.size() - N_elements_recv_right + i;

            elements_[index].phi_ = std::vector<hostFloat>(elements_[index].N_ + 1);
            elements_[index].q_ = std::vector<hostFloat>(elements_[index].N_ + 1);
            elements_[index].ux_ = std::vector<hostFloat>(elements_[index].N_ + 1);
            elements_[index].phi_prime_ = std::vector<hostFloat>(elements_[index].N_ + 1);
            elements_[index].intermediate_ = std::vector<hostFloat>(elements_[index].N_ + 1);
        }

        for (int i = 0; i < N_elements_send_left; ++i) {
            const int index = global_element_offset_current + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&new_elements[i].x_[0], 2, data_type, destination, 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + N_elements_send_left + N_elements_send_right]);
            MPI_Isend(new_elements[i].phi_.data(), new_elements[i].N_ + 1, data_type, destination, 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + 2 * N_elements_send_left + 2 * N_elements_send_right]);
        }

        for (int i = 0; i < N_elements_send_right; ++i) {
            const int index = global_element_offset_end + 1 + i;
            const int destination = index/N_elements_per_process_;

            MPI_Isend(&new_elements[new_elements.size() - N_elements_send_right + i].x_[0], 2, data_type, destination, 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + 2 * N_elements_send_left + N_elements_send_right]);
            MPI_Isend(new_elements[new_elements.size() - N_elements_send_right + i].phi_.data(), new_elements[new_elements.size() - N_elements_send_right + i].N_ + 1, data_type, destination, 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 3 * N_elements_recv_right + 3 * N_elements_send_left + 2 * N_elements_send_right]);
        }

        for (int i = 0; i < N_elements_recv_left; ++i) {
            const int index = global_element_offset_ + i;

            MPI_Irecv(&elements_[i].x_[0], 2, data_type, left_origins[i], 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + N_elements_recv_left + N_elements_recv_right]);
            MPI_Irecv(elements_[i].phi_.data(), elements_[i].N_ + 1, data_type, left_origins[i], 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 2 * N_elements_recv_left + 2 * N_elements_recv_right]);
        }

        for (int i = 0; i < N_elements_recv_right; ++i) {
            const int index = global_element_offset_end_current + i + 1;

            MPI_Irecv(&elements_[elements_.size() - N_elements_recv_right + i].x_[0], 2, data_type, right_origins[i], 3 * index + 1, MPI_COMM_WORLD, &adaptivity_requests[i + 2 * N_elements_recv_left + N_elements_recv_right]);
            MPI_Irecv(elements_[elements_.size() - N_elements_recv_right + i].phi_.data(), elements_[elements_.size() - N_elements_recv_right + i].N_ + 1, data_type, right_origins[i], 3 * index + 2, MPI_COMM_WORLD, &adaptivity_requests[i + 3 * N_elements_recv_left + 2 * N_elements_recv_right]);
        }

        MPI_Waitall(2 * N_elements_recv_left + 2 * N_elements_recv_right, adaptivity_requests.data() + N_elements_recv_left + N_elements_recv_right, adaptivity_statuses.data() + N_elements_recv_left + N_elements_recv_right);

        for (int i = 0; i < N_elements_recv_left; ++i) {
            elements_[i].delta_x_ = elements_[i].x_[1] - elements_[i].x_[0];
            elements_[i].faces_ = {static_cast<size_t>(i), static_cast<size_t>(i) + 1};
        }

        for (int i = 0; i < N_elements_recv_right; ++i) {
            const size_t index = elements_.size() - N_elements_recv_right + i;

            elements_[index].delta_x_ = elements_[index].x_[1] - elements_[index].x_[0];
            elements_[index].faces_ = {index, index + 1};
        }

         // We also wait for the send requests
        MPI_Waitall(3 * N_elements_send_left + 3 * N_elements_send_right, adaptivity_requests.data() + 3 * N_elements_recv_left + 3 * N_elements_recv_right, adaptivity_statuses.data() + 3 * N_elements_recv_left + 3 * N_elements_recv_right);

    }

    move_elements(N_elements_ - N_elements_recv_left - N_elements_recv_right, new_elements, N_elements_send_left, N_elements_recv_left);


    local_boundary_to_element_ = std::vector<size_t>(N_local_boundaries_);
    MPI_boundary_to_element_ = std::vector<size_t>(N_MPI_boundaries_);
    MPI_boundary_from_element_ = std::vector<size_t>(N_MPI_boundaries_);
    send_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    receive_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    requests_ = std::vector<MPI_Request>(N_MPI_boundaries_*2);
    statuses_ = std::vector<MPI_Status>(N_MPI_boundaries_*2);
    refine_array_ = std::vector<size_t>(N_elements_);

    // CHECK create boundaries here.
    build_boundaries();
}

void SEM::Host::Meshes::Mesh_t::p_adapt(int N_max, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) {
    for (size_t i = 0; i < N_elements_; ++i) {
        if (elements_[i].refine_ && elements_[i].sigma_ >= 1.0 && elements_[i].N_ < N_max) {
            SEM::Host::Entities::Element_t new_element(std::min(elements_[i].N_ + 2, N_max), elements_[i].faces_[0], elements_[i].faces_[1], elements_[i].x_[0], elements_[i].x_[1]);
            new_element.interpolate_from(elements_[i], nodes, barycentric_weights);
            elements_[i] = std::move(new_element);
        }
    }
}

void SEM::Host::Meshes::Mesh_t::hp_adapt(int N_max, std::vector<SEM::Host::Entities::Element_t>& new_elements, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) {
    for (size_t i = 0; i < N_elements_; ++i) {
        const size_t element_index = i + refine_array_[i];

        if (elements_[i].refine_ && elements_[i].sigma_ < 1.0 && elements_[i].delta_x_/2 >= delta_x_min_) {
            new_elements[element_index] = SEM::Host::Entities::Element_t(elements_[i].N_, element_index, element_index + 1, elements_[i].x_[0], (elements_[i].x_[0] + elements_[i].x_[1]) * 0.5);
            new_elements[element_index + 1] = SEM::Host::Entities::Element_t(elements_[i].N_, element_index + 1, element_index + 2, (elements_[i].x_[0] + elements_[i].x_[1]) * 0.5, elements_[i].x_[1]);
            new_elements[element_index].interpolate_from(elements_[i], nodes, barycentric_weights);
            new_elements[element_index + 1].interpolate_from(elements_[i], nodes, barycentric_weights);
        }
        else if (elements_[i].refine_ && elements_[i].sigma_ >= 1.0 && elements_[i].N_ < N_max) {
            new_elements[element_index] = SEM::Host::Entities::Element_t(std::min(elements_[i].N_ + 2, N_max), element_index, element_index + 1, elements_[i].x_[0], elements_[i].x_[1]);
            new_elements[element_index].interpolate_from(elements_[i], nodes, barycentric_weights);
        }
        else {
            new_elements[element_index] = std::move(elements_[i]);
            new_elements[element_index].faces_ = {element_index, element_index + 1};
        }
    }
}

void SEM::Host::Meshes::matrix_vector_multiply(int N, const std::vector<hostFloat>& matrix, const std::vector<hostFloat>& vector, std::vector<hostFloat>& result) {
    for (int i = 0; i < vector.size(); ++i) {
        result[i] = 0.0f;
        for (int j = 0; j < vector.size(); ++j) {
            result[i] +=  matrix[i * (N + 1) + j] * vector[j];
        }
    }
}

// Algorithm 19
void SEM::Host::Meshes::matrix_vector_derivative(int N, const std::vector<hostFloat>& derivative_matrices_hat,  const std::vector<hostFloat>& phi, std::vector<hostFloat>& phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    
    for (size_t i = 0; i < phi.size(); ++i) {
        phi_prime[i] = 0.0f;
        for (size_t j = 0; j < phi.size(); ++j) {
            phi_prime[i] += derivative_matrices_hat[i * (N + 1) + j] * phi[j] * phi[j]/2;
        }
    }
}

// Algorithm 60 (not really anymore)
void SEM::Host::Meshes::Mesh_t::compute_dg_derivative(const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    for (size_t i = 0; i < N_elements_; ++i) {
        const hostFloat flux_L = faces_[elements_[i].faces_[0]].flux_;
        const hostFloat flux_R = faces_[elements_[i].faces_[1]].flux_;

        SEM::Host::Meshes::matrix_vector_multiply(elements_[i].N_, derivative_matrices_hat[elements_[i].N_], elements_[i].phi_, elements_[i].q_);

        for (int j = 0; j <= elements_[i].N_; ++j) {
            elements_[i].q_[j] = -elements_[i].q_[j] - (flux_R * lagrange_interpolant_right[elements_[i].N_][j]
                                                     - flux_L * lagrange_interpolant_left[elements_[i].N_][j]) / weights[elements_[i].N_][j];
            elements_[i].q_[j] *= 2.0f/elements_[i].delta_x_;
        }
    }
}

// Algorithm 60 (not really anymore)
void SEM::Host::Meshes::Mesh_t::compute_dg_derivative2(hostFloat viscosity, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    for (size_t i = 0; i < N_elements_; ++i) {
        const hostFloat derivative_flux_L = faces_[elements_[i].faces_[0]].derivative_flux_;
        const hostFloat derivative_flux_R = faces_[elements_[i].faces_[1]].derivative_flux_;
        const hostFloat nl_flux_L = faces_[elements_[i].faces_[0]].nl_flux_;
        const hostFloat nl_flux_R = faces_[elements_[i].faces_[1]].nl_flux_;

        SEM::Host::Meshes::matrix_vector_derivative(elements_[i].N_, derivative_matrices_hat[elements_[i].N_], elements_[i].phi_, elements_[i].ux_);
        SEM::Host::Meshes::matrix_vector_multiply(elements_[i].N_, derivative_matrices_hat[elements_[i].N_], elements_[i].q_, elements_[i].phi_prime_);

        for (int j = 0; j <= elements_[i].N_; ++j) {
            elements_[i].phi_prime_[j] = -elements_[i].phi_prime_[j] * viscosity
                                        - (derivative_flux_R * lagrange_interpolant_right[elements_[i].N_][j]
                                           - derivative_flux_L * lagrange_interpolant_left[elements_[i].N_][j]) * viscosity /weights[elements_[i].N_][j]
                                        - elements_[i].ux_[j]
                                        + (nl_flux_L * lagrange_interpolant_left[elements_[i].N_][j] 
                                            - nl_flux_R * lagrange_interpolant_right[elements_[i].N_][j]) / weights[elements_[i].N_][j];

            elements_[i].phi_prime_[j] *= 2.0f/elements_[i].delta_x_;
        }
    }
}

void SEM::Host::Meshes::Mesh_t::interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    for (size_t i = 0; i < N_elements_; ++i) {
        elements_[i].interpolate_to_boundaries(lagrange_interpolant_left, lagrange_interpolant_right);
    }
}

void SEM::Host::Meshes::Mesh_t::interpolate_q_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    for (size_t i = 0; i < N_elements_; ++i) {
        elements_[i].interpolate_q_to_boundaries(lagrange_interpolant_left, lagrange_interpolant_right);
    }
}
