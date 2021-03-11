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
        N_elements_global_(N_elements),
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
    build_boundaries(x_min_local, x_max_local);
    build_faces(); // CHECK
}

SEM::Mesh_host_t::~Mesh_host_t() {}

void SEM::Mesh_host_t::set_initial_conditions(const std::vector<std::vector<hostFloat>>& nodes) {
    for (size_t i = 0; i < N_elements_; ++i) {
        for (int j = 0; j <= elements_[i].N_; ++j) {
            const hostFloat x = (0.5 + nodes[elements_[i].N_][j]/2.0f) * (elements_[i].x_[1] - elements_[i].x_[0]) + elements_[i].x_[0];
            elements_[i].phi_[j] = g(x);
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

template void SEM::Mesh_host_t::solve(const hostFloat CFL, const std::vector<hostFloat> output_times, const NDG_host_t<ChebyshevPolynomial_host_t> &NDG, hostFloat viscosity); // Get with the times c++, it's crazy I have to do this
template void SEM::Mesh_host_t::solve(const hostFloat CFL, const std::vector<hostFloat> output_times, const NDG_host_t<LegendrePolynomial_host_t> &NDG, hostFloat viscosity);

template<typename Polynomial>
void SEM::Mesh_host_t::solve(const hostFloat CFL, const std::vector<hostFloat> output_times, const NDG_host_t<Polynomial> &NDG, hostFloat viscosity) {
    hostFloat time = 0.0;
    hostFloat t_end = output_times.back();

    hostFloat delta_t = get_delta_t(CFL);
    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);

    while (time < t_end) {
        // Kinda algorithm 62
        hostFloat t = time;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_, NDG.lagrange_interpolant_derivative_left_, NDG.lagrange_interpolant_derivative_right_);
        boundary_conditions();
        calculate_fluxes();
        compute_dg_derivative(viscosity, NDG.weights_, NDG.derivative_matrices_hat_, NDG.g_hat_derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_first_step(delta_t, 1.0/3.0);

        t = time + 0.33333333333 * delta_t;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_, NDG.lagrange_interpolant_derivative_left_, NDG.lagrange_interpolant_derivative_right_);
        boundary_conditions();
        calculate_fluxes();
        compute_dg_derivative(viscosity, NDG.weights_, NDG.derivative_matrices_hat_, NDG.g_hat_derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step(delta_t, -5.0/9.0, 15.0/16.0);

        t = time + 0.75 * delta_t;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_, NDG.lagrange_interpolant_derivative_left_, NDG.lagrange_interpolant_derivative_right_);
        boundary_conditions();
        calculate_fluxes();
        compute_dg_derivative(viscosity, NDG.weights_, NDG.derivative_matrices_hat_, NDG.g_hat_derivative_matrices_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step(delta_t, -153.0/128.0, 8.0/15.0);
              
        time += delta_t;
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                estimate_error<Polynomial>(NDG.nodes_, NDG.weights_);
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
                adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
                delta_t = get_delta_t(CFL);
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
        estimate_error<Polynomial>(NDG.nodes_, NDG.weights_);
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
        const size_t neighbour_L = (i > 0) ? i - 1 : faces_.size() - 1;
        const size_t neighbour_R = (i < faces_.size() - 1) ? i : faces_.size(); // Last face links last element to first element
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

void SEM::Mesh_host_t::rk3_first_step(hostFloat delta_t, hostFloat g) {
    for (size_t i = 0; i < N_elements_; ++i) {
        for (int j = 0; j <= elements_[i].N_; ++j){
            elements_[i].intermediate_[j] = elements_[i].phi_prime_[j];
            elements_[i].phi_[j] += g * delta_t * elements_[i].intermediate_[j];
        }
    }
}

void SEM::Mesh_host_t::rk3_step(hostFloat delta_t, hostFloat a, hostFloat g) {
    for (size_t i = 0; i < N_elements_; ++i) {
        for (int j = 0; j <= elements_[i].N_; ++j){
            elements_[i].intermediate_[j] = a * elements_[i].intermediate_[j] + elements_[i].phi_prime_[j];
            elements_[i].phi_[j] += g * delta_t * elements_[i].intermediate_[j];
        }
    }
}

hostFloat SEM::Mesh_host_t::get_delta_t(const hostFloat CFL) {   
    double delta_t_min_local = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N_elements_; ++i) {
        deviceFloat phi_max = 0.0;
        for (int j = 0; j <= elements_[i].N_; ++j) {
            phi_max = std::max(phi_max, std::abs(elements_[i].phi_[j]));
        }
        const hostFloat delta_t = CFL * elements_[i].delta_x_ * elements_[i].delta_x_/(phi_max * elements_[i].N_ * elements_[i].N_);
        delta_t_min_local = std::min(delta_t_min_local, delta_t);
    }

    double delta_t_min;
    MPI_Allreduce(&delta_t_min_local, &delta_t_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    return delta_t_min;
}

void SEM::Mesh_host_t::boundary_conditions() {
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

void SEM::Mesh_host_t::local_boundaries() {
    for (size_t i = 0; i < N_local_boundaries_; ++i) {
        elements_[N_elements_ + i].phi_L_ = elements_[local_boundary_to_element_[i]].phi_L_;
        elements_[N_elements_ + i].phi_R_ = elements_[local_boundary_to_element_[i]].phi_R_;
        elements_[N_elements_ + i].phi_prime_L_ = elements_[local_boundary_to_element_[i]].phi_prime_L_;
        elements_[N_elements_ + i].phi_prime_R_ = elements_[local_boundary_to_element_[i]].phi_prime_R_;
    }
}

void SEM::Mesh_host_t::get_MPI_boundaries() {
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        const Element_host_t& boundary_element = elements_[N_elements_ + N_local_boundaries_ + i];
        const Face_host_t& boundary_face = faces_[boundary_element.faces_[0]];
        const Element_host_t& domain_element = elements_[boundary_face.elements_[boundary_face.elements_[0] == N_elements_ + N_local_boundaries_ + i]];
        
        send_buffers_[i] = {domain_element.phi_L_,
                            domain_element.phi_R_,
                            domain_element.phi_prime_L_,
                            domain_element.phi_prime_R_};
    }
}

void SEM::Mesh_host_t::put_MPI_boundaries() {
    for (size_t i = 0; i < N_MPI_boundaries_; ++i) {
        elements_[N_elements_ + N_local_boundaries_ + i].phi_L_ = receive_buffers_[i][0];
        elements_[N_elements_ + N_local_boundaries_ + i].phi_R_ = receive_buffers_[i][1];
        elements_[N_elements_ + N_local_boundaries_ + i].phi_prime_L_ = receive_buffers_[i][2];
        elements_[N_elements_ + N_local_boundaries_ + i].phi_prime_R_ = receive_buffers_[i][3];
    }
}

void SEM::Mesh_host_t::calculate_fluxes() {
    for (auto& face: faces_) {
        hostFloat u;
        const hostFloat u_left = elements_[face.elements_[0]].phi_R_;
        const hostFloat u_right = elements_[face.elements_[1]].phi_L_;
        const hostFloat u_prime_left = elements_[face.elements_[0]].phi_prime_R_;
        const hostFloat u_prime_right = elements_[face.elements_[1]].phi_prime_L_;

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

        face.flux_ = 0.5 * u * u;
        face.derivative_flux_ = 0.5 * (u_prime_left + u_prime_right);
    }
}

void SEM::Mesh_host_t::adapt(int N_max, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) {
    size_t additional_elements = 0;
    for (size_t i = 0; i < N_elements_; ++i) {
        additional_elements += elements_[i].refine_ * (elements_[i].sigma_ < 1.0);
        refine_array_[i] = additional_elements - elements_[i].refine_ * (elements_[i].sigma_ < 1.0); // Current offset
    }

    if (additional_elements == 0) {
        p_adapt(N_max, nodes, barycentric_weights);
        return;
    }

    std::vector<Element_host_t> new_elements(N_elements_ + additional_elements);
    std::vector<Face_host_t> new_faces(faces_.size() + additional_elements);

    copy_faces(new_faces);
    hp_adapt(N_max, new_elements, new_faces, nodes, barycentric_weights);

    elements_ = std::move(new_elements);
    faces_ = std::move(new_faces);
    
    N_elements_ += additional_elements;
    N_elements_per_process_ = N_elements_per_process_; // CHECK change

    local_boundary_to_element_ = std::vector<size_t>(N_local_boundaries_);
    MPI_boundary_to_element_ = std::vector<size_t>(N_MPI_boundaries_);
    MPI_boundary_from_element_ = std::vector<size_t>(N_MPI_boundaries_);
    send_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    receive_buffers_ = std::vector<std::array<double, 4>>(N_MPI_boundaries_);
    requests_ = std::vector<MPI_Request>(N_MPI_boundaries_*2);
    statuses_ = std::vector<MPI_Status>(N_MPI_boundaries_*2);
    refine_array_ = std::vector<size_t>(N_elements_);

    // CHECK create boundaries here.
}

void SEM::Mesh_host_t::p_adapt(int N_max, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) {
    for (size_t i = 0; i < N_elements_; ++i) {
        if (elements_[i].refine_ && elements_[i].sigma_ >= 1.0 && elements_[i].N_ < N_max) {
            SEM::Element_host_t new_element(std::min(elements_[i].N_ + 2, N_max), elements_[i].faces_[0], elements_[i].faces_[1], elements_[i].x_[0], elements_[i].x_[1]);
            new_element.interpolate_from(elements_[i], nodes, barycentric_weights);
            elements_[i] = std::move(new_element);
        }
    }
}

void SEM::Mesh_host_t::hp_adapt(int N_max, std::vector<Element_host_t>& new_elements, std::vector<Face_host_t>& new_faces, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) {
    for (size_t i = 0; i < N_elements_; ++i) {
        if (elements_[i].refine_ && elements_[i].sigma_ < 1.0) {
            size_t new_index = N_elements_ + refine_array_[i];

            new_elements[i] = SEM::Element_host_t(elements_[i].N_, elements_[i].faces_[0], new_index, elements_[i].x_[0], (elements_[i].x_[0] + elements_[i].x_[1]) * 0.5);
            new_elements[new_index] = SEM::Element_host_t(elements_[i].N_, new_index, elements_[i].faces_[1], (elements_[i].x_[0] + elements_[i].x_[1]) * 0.5, elements_[i].x_[1]);
            new_elements[i].interpolate_from(elements_[i], nodes, barycentric_weights);
            new_elements[new_index].interpolate_from(elements_[i], nodes, barycentric_weights);
            
            new_faces[new_index] = SEM::Face_host_t(i, new_index);
            new_faces[elements_[i].faces_[1]].elements_[0] = new_index;
        }
        else if (elements_[i].refine_ && elements_[i].N_ < N_max) {
            new_elements[i] = SEM::Element_host_t(std::min(elements_[i].N_ + 2, N_max), elements_[i].faces_[0], elements_[i].faces_[1], elements_[i].x_[0], elements_[i].x_[1]);
            new_elements[i].interpolate_from(elements_[i], nodes, barycentric_weights);
        }
        else {
            new_elements[i] = std::move(elements_[i]);
        }
    }
}


void SEM::Mesh_host_t::copy_faces(std::vector<Face_host_t>& new_faces) {
    for (size_t i = 0; i < faces_.size(); ++i) {
        new_faces[i] = std::move(faces_[i]);
    }
}

template void SEM::Mesh_host_t::estimate_error<SEM::ChebyshevPolynomial_host_t>(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);
template void SEM::Mesh_host_t::estimate_error<SEM::LegendrePolynomial_host_t>(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);

template<typename Polynomial>
void SEM::Mesh_host_t::estimate_error(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights) {
    for (size_t i = 0; i < N_elements_; ++i) {
        elements_[i].estimate_error<Polynomial>(nodes, weights);
    }
}

// Algorithm 19
void SEM::matrix_vector_derivative(hostFloat viscosity, int N, const std::vector<hostFloat>& derivative_matrices_hat,  const std::vector<hostFloat>& g_hat_derivative_matrices, const std::vector<hostFloat>& phi, std::vector<hostFloat>& phi_prime) {
    // s = 0, e = N (p.55 says N - 1)
    
    for (size_t i = 0; i < phi.size(); ++i) {
        phi_prime[i] = 0.0f;
        for (size_t j = 0; j < phi.size(); ++j) {
            //phi_prime[i] += derivative_matrices_hat[i * phi.size() + j] * phi[j] * phi[j] * 0.5; // phi not squared in textbook, squared for Burger's
            phi_prime[i] -= viscosity * g_hat_derivative_matrices[i * (N + 1) + j] * phi[j];
        }
    }
}

// Algorithm 60 (not really anymore)
void SEM::Mesh_host_t::compute_dg_derivative(hostFloat viscosity, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& g_hat_derivative_matrices, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    for (size_t i = 0; i < N_elements_; ++i) {
        const hostFloat flux_L = faces_[elements_[i].faces_[0]].flux_;
        const hostFloat flux_R = faces_[elements_[i].faces_[1]].flux_;
        const hostFloat derivative_flux_L = faces_[elements_[i].faces_[0]].derivative_flux_;
        const hostFloat derivative_flux_R = faces_[elements_[i].faces_[1]].derivative_flux_;

        SEM::matrix_vector_derivative(viscosity, elements_[i].N_, derivative_matrices_hat[elements_[i].N_], g_hat_derivative_matrices[elements_[i].N_], elements_[i].phi_, elements_[i].phi_prime_);

        for (int j = 0; j <= elements_[i].N_; ++j) {
            /*elements_[i].phi_prime_[j] += (flux_L * lagrange_interpolant_left[elements_[i].N_][j] - flux_R * lagrange_interpolant_right[elements_[i].N_][j]) / weights[elements_[i].N_][j];
            elements_[i].phi_prime_[j] *= 2.0f/elements_[i].delta_x_;*/

            elements_[i].phi_prime_[j] += (viscosity * derivative_flux_R * lagrange_interpolant_right[elements_[i].N_][j]
                                        - viscosity * derivative_flux_L * lagrange_interpolant_left[elements_[i].N_][j]) * elements_[i].delta_x_ * 0.5f / weights[elements_[i].N_][j];
            elements_[i].phi_prime_[j] *= 0.5f/(elements_[i].delta_x_ * elements_[i].delta_x_);
        }
    }
}

void SEM::Mesh_host_t::interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_derivative_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_derivative_right) {
    for (size_t i = 0; i < N_elements_; ++i) {
        elements_[i].interpolate_to_boundaries(lagrange_interpolant_left, lagrange_interpolant_right, lagrange_interpolant_derivative_left, lagrange_interpolant_derivative_right);
    }
}