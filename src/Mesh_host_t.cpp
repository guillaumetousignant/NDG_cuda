#include "Mesh_host_t.h"
#include "ChebyshevPolynomial_host_t.h"
#include "LegendrePolynomial_host_t.h"
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

constexpr hostFloat pi = 3.14159265358979323846;

SEM::Mesh_host_t::Mesh_host_t(size_t N_elements, int initial_N, hostFloat x_min, hostFloat x_max) : 
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

    faces_ = std::vector<Face_host_t>(N_faces);
    elements_ = std::vector<Element_host_t>(N_elements_ + N_local_boundaries_ + N_MPI_boundaries_);
    send_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    receive_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    requests_ = std::vector<MPI_Request>(N_MPI_boundaries_*2);
    statuses_ = std::vector<MPI_Status>(N_MPI_boundaries_*2);

    const hostFloat delta_x = (x_max - x_min)/N_elements_global_;
    const hostFloat x_min_local = x_min + delta_x * global_rank * N_elements_per_process_;
    const hostFloat x_max_local = x_min_local + N_elements_ * delta_x;
    
    build_elements(x_min_local, x_max_local);
    build_boundaries(x_min_local, x_max_local);
    build_faces(); // CHECK
}

SEM::Mesh_host_t::~Mesh_host_t() {}

void SEM::Mesh_host_t::set_initial_conditions(const std::vector<std::vector<hostFloat>>& nodes) {
    for (auto& element: elements_) {
        for (int j = 0; j <= element.N_; ++j) {
            const hostFloat x = (0.5 + nodes[element.N_][j]/2.0f) * (element.x_[1] - element.x_[0]) + element.x_[0];
            element.phi_[j] = g(x);
        }
    }
}

void SEM::Mesh_host_t::print() {
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

    std::cout << std::endl << "Elements: " << std::endl;
    for (size_t i = 0; i < faces_.size(); ++i) {
        std::cout << '\t' << "Face " << i << ": ";
        std::cout << '\t' << '\t';
        std::cout << faces_[i].elements_[0] << " ";
        std::cout << faces_[i].elements_[1] << std::endl;
    }
}

void SEM::Mesh_host_t::write_file_data(size_t N_interpolation_points, size_t N_elements, hostFloat time, int rank, const std::vector<hostFloat>& velocity, const std::vector<hostFloat>& coordinates, const std::vector<hostFloat>& du_dx, const std::vector<hostFloat>& intermediate, const std::vector<hostFloat>& x_L, const std::vector<hostFloat>& x_R, const std::vector<int>& N, const std::vector<hostFloat>& sigma, const std::vector<bool>& refine, const std::vector<bool>& coarsen, const std::vector<hostFloat>& error) {
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

void SEM::Mesh_host_t::write_data(hostFloat time, size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices) {
    std::vector<hostFloat> phi(N_elements_ * N_interpolation_points);
    std::vector<hostFloat> x(N_elements_ * N_interpolation_points);
    std::vector<hostFloat> phi_prime(N_elements_ * N_interpolation_points);
    std::vector<hostFloat> intermediate(N_elements_ * N_interpolation_points);
    std::vector<hostFloat> x_L(N_elements_);
    std::vector<hostFloat> x_R(N_elements_);
    std::vector<int> N(N_elements_);
    std::vector<hostFloat> sigma(N_elements_);
    std::vector<bool> refine(N_elements_);
    std::vector<bool> coarsen(N_elements_);
    std::vector<hostFloat> error(N_elements_);

    get_solution(N_interpolation_points, interpolation_matrices, phi, x, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    
    write_file_data(N_interpolation_points, N_elements_, time, global_rank, phi, x, phi_prime, intermediate, x_L, x_R, N, sigma, refine, coarsen, error);
}

template void SEM::Mesh_host_t::solve(const hostFloat delta_t, const std::vector<hostFloat> output_times, const NDG_host_t<ChebyshevPolynomial_host_t> &NDG); // Get with the times c++, it's crazy I have to do this
template void SEM::Mesh_host_t::solve(const hostFloat delta_t, const std::vector<hostFloat> output_times, const NDG_host_t<LegendrePolynomial_host_t> &NDG);

template<typename Polynomial>
void SEM::Mesh_host_t::solve(hostFloat delta_t, const std::vector<hostFloat> output_times, const NDG_host_t<Polynomial> &NDG) {
    hostFloat time = 0.0;
    hostFloat t_end = output_times.back();

    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

    while (time < t_end) {
        // Kinda algorithm 62
        hostFloat t = time;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        calculate_fluxes();
        compute_dg_derivative(NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step(delta_t, 0.0, 1.0/3.0);

        t = time + 0.33333333333 * delta_t;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        calculate_fluxes();
        compute_dg_derivative(NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step(delta_t, -5.0/9.0, 15.0/16.0);

        t = time + 0.75 * delta_t;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        calculate_fluxes();
        compute_dg_derivative(NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step(delta_t, -153.0/128.0, 8.0f/15.0);
              
        time += delta_t;
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
                break;
            }
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
        write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
    }
}

void SEM::Mesh_host_t::build_elements(hostFloat x_min, hostFloat x_max) {
    for (size_t i = 0; i < elements_.size(); ++i) {
        const size_t face_L = i;
        const size_t face_R = i + 1;
        const hostFloat delta_x = (x_max - x_min)/N_elements_;
        const hostFloat element_x_min = x_min + i * delta_x;
        const hostFloat element_x_max = x_min + (i + 1) * delta_x;

        elements_[i] = Element_host_t(initial_N_, face_L, face_R, element_x_min, element_x_max);
    }
}

void SEM::Mesh_host_t::build_boundaries(hostFloat x_min, hostFloat x_max) {
    for (int i = 0; i < N_local_boundaries_; ++i) {
        const hostFloat delta_x = (x_max - x_min)/N_elements_;
        size_t face_L;
        size_t face_R;
        hostFloat element_x_min;
        hostFloat element_x_max;

        if (i == 0) { // CHECK this is hardcoded for 1D
            face_L = 0;
            face_R = 0;
            element_x_min = x_min - delta_x;
            element_x_max = x_min;
            local_boundary_to_element_[i] = N_elements_ - 1;
        }
        else if (i == 1) {
            face_L = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            face_R = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            element_x_min = x_max;
            element_x_max = x_max + delta_x;
            local_boundary_to_element_[i] = 0;
        }

        elements_[N_elements_ + i] = SEM::Element_host_t(0, face_L, face_R, element_x_min, element_x_max);
    }

    for (int i = 0; i < N_MPI_boundaries_; ++i) {
        const hostFloat delta_x = (x_max - x_min)/N_elements_;
        size_t face_L;
        size_t face_R;
        hostFloat element_x_min;
        hostFloat element_x_max;

        if (i == 0) { // CHECK this is hardcoded for 1D
            face_L = 0;
            face_R = 0;
            element_x_min = x_min - delta_x;
            element_x_max = x_min;
            MPI_boundary_to_element_[i] = (global_element_offset_ == 0) ? N_elements_global_ - 1 : global_element_offset_ - 1;
            MPI_boundary_from_element_[i] = global_element_offset_;
        }
        else if (i == 1) {
            face_L = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            face_R = N_elements_ + N_local_boundaries_ + N_MPI_boundaries_ - 2;
            element_x_min = x_max;
            element_x_max = x_max + delta_x;
            MPI_boundary_to_element_[i] = (global_element_offset_ + N_elements_ == N_elements_global_) ? 0 : global_element_offset_ + N_elements_;
            MPI_boundary_from_element_[i] = global_element_offset_ + N_elements_ - 1;
        }

        elements_[N_elements_ + N_local_boundaries_ + i] = SEM::Element_host_t(0, face_L, face_R, element_x_min, element_x_max);
    }
}

void SEM::Mesh_host_t::build_faces() {
    for (size_t i = 0; i < faces_.size(); ++i) {
        const size_t neighbour_L = i;
        const size_t neighbour_R = (i < faces_.size() - 1) ? i + 1 : 0; // Last face links last element to first element
        faces_[i] = Face_host_t(neighbour_L, neighbour_R);
    }
}

hostFloat SEM::Mesh_host_t::g(hostFloat x) {
    //return (x < -0.2f || x > 0.2f) ? 0.2f : 0.8f;
    return -std::sin(pi * x);
}

void SEM::Mesh_host_t::get_solution(size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<hostFloat>& phi, std::vector<hostFloat>& x, std::vector<hostFloat>& phi_prime, std::vector<hostFloat>& intermediate, std::vector<hostFloat>& x_L, std::vector<hostFloat>& x_R, std::vector<int>& N, std::vector<hostFloat>& sigma, std::vector<bool>& refine, std::vector<bool>& coarsen, std::vector<hostFloat>& error) {
    for (size_t i = 0; i < N_elements_; ++i) {
        const size_t offset_interp_1D = i * N_interpolation_points;
        const size_t step = N_interpolation_points/(elements_[i].N_ + 1);

        for (size_t j = 0; j < N_interpolation_points; ++j) {
            phi[offset_interp_1D + j] = 0.0f;
            phi_prime[offset_interp_1D + j] = 0.0f;
            for (int k = 0; k <= elements_[i].N_; ++k) {
                phi[offset_interp_1D + j] += interpolation_matrices[elements_[i].N_][j * (elements_[i].N_ + 1) + k] * elements_[i].phi_[k];
                phi_prime[offset_interp_1D + j] += interpolation_matrices[elements_[i].N_][j * (elements_[i].N_ + 1) + k] * elements_[i].phi_prime_[k];
            }
            intermediate[offset_interp_1D + j] = elements_[i].intermediate_[std::min(static_cast<int>(j/step), elements_[i].N_)];
            x[offset_interp_1D + j] = j * (elements_[i].x_[1] - elements_[i].x_[0]) / (N_interpolation_points - 1) + elements_[i].x_[0];
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

void SEM::Mesh_host_t::rk3_step(hostFloat delta_t, hostFloat a, hostFloat g) {
    for (auto& element: elements_) {
        for (int j = 0; j <= element.N_; ++j){
            element.intermediate_[j] = a * element.intermediate_[j] + element.phi_prime_[j];
            element.phi_[j] += g * delta_t * element.intermediate_[j];
        }
    }
}

void SEM::Mesh_host_t::calculate_fluxes() {
    for (auto& face: faces_) {
        hostFloat u;
        const hostFloat u_left = elements_[face.elements_[0]].phi_R_;
        const hostFloat u_right = elements_[face.elements_[1]].phi_L_;

        if (u_left < 0.0 && u_right > 0.0) { // In expansion fan
            u = 0.5 * (u_left + u_right);
        }
        else if (u_left > u_right) { // Shock
            if (u_left > 0.0) {
                u = u_left;
            }
            else if (u_left < u_right) {
                u = u_right;
            }
            else { // ADDED
                u = 0.5 * (u_left + u_right);
            }
        }
        else if (u_left < u_right) { // Expansion fan
            if (u_left > 0.0) {
                u = u_left;
            }
            else if (u_left < 0.0) {
                u = u_right;
            }
            else { // ADDED
                u = 0.5 * (u_left + u_right);
            }
        }
        else { // ADDED
            u = 0.5 * (u_left + u_right);
        }

        face.flux_ = 0.5 * u * u;
    }
}

// Algorithm 19
void SEM::matrix_vector_derivative(const std::vector<hostFloat>& derivative_matrices_hat, const std::vector<hostFloat>& phi, std::vector<hostFloat>& phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    
    for (size_t i = 0; i < phi.size(); ++i) {
        phi_prime[i] = 0.0f;
        for (size_t j = 0; j < phi.size(); ++j) {
            phi_prime[i] += derivative_matrices_hat[i * phi.size() + j] * phi[j] * phi[j] * 0.5; // phi not squared in textbook, squared for Burger's
        }
    }
}

// Algorithm 60 (not really anymore)
void SEM::Mesh_host_t::compute_dg_derivative(const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    for (auto& element: elements_) {
        const hostFloat flux_L = faces_[element.faces_[0]].flux_;
        const hostFloat flux_R = faces_[element.faces_[1]].flux_;

        SEM::matrix_vector_derivative(derivative_matrices_hat[element.N_], element.phi_, element.phi_prime_);

        for (int j = 0; j <= element.N_; ++j) {
            element.phi_prime_[j] += (flux_L * lagrange_interpolant_left[element.N_][j] - flux_R * lagrange_interpolant_right[element.N_][j]) / weights[element.N_][j];
            element.phi_prime_[j] *= 2.0f/element.delta_x_;
        }
    }
}

void SEM::Mesh_host_t::interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    for (auto& element: elements_) {
        element.interpolate_to_boundaries(lagrange_interpolant_left, lagrange_interpolant_right);
    }
}