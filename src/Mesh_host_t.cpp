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
        initial_N_(initial_N),
        elements_(N_elements),
        faces_(N_elements) {
    // CHECK N_faces = N_elements only for periodic BC.
    
    build_elements(x_min, x_max);
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

void SEM::Mesh_host_t::write_file_data(hostFloat time, const std::vector<hostFloat>& velocity, const std::vector<hostFloat>& coordinates) {
    std::stringstream ss;
    std::ofstream file;

    fs::path save_dir = fs::current_path() / "data";
    fs::create_directory(save_dir);

    ss << "output_t" << std::setprecision(4) << std::fixed << time << ".dat";
    file.open(save_dir / ss.str());

    file << "TITLE = \"Velocity  at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\"" << std::endl;
    file << "ZONE T= \"Zone     1\",  I= " << coordinates.size() << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (size_t i = 0; i < coordinates.size(); ++i) {
        file << std::setw(12) << coordinates[i] << " " << std::setw(12) << (std::isnan(velocity[i]) ? 0.0 : velocity[i]) << std::endl;
    }

    file.close();
}

void SEM::Mesh_host_t::write_data(hostFloat time, size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices) {
    std::vector<hostFloat> phi(elements_.size() * N_interpolation_points);
    std::vector<hostFloat> x(elements_.size() * N_interpolation_points);
    get_solution(N_interpolation_points, interpolation_matrices, phi, x);
    
    write_file_data(time, phi, x);
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
        const size_t face_L = (i > 0) ? i - 1 : elements_.size() - 1;
        const size_t face_R = i;
        const hostFloat delta_x = (x_max - x_min)/elements_.size();
        const hostFloat element_x_min = x_min + i * delta_x;
        const hostFloat element_y_min = x_min + (i + 1) * delta_x;
        elements_[i] = Element_host_t(initial_N_, face_L, face_R, element_x_min, element_y_min);
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

void SEM::Mesh_host_t::get_solution(size_t N_interpolation_points, const std::vector<std::vector<hostFloat>>& interpolation_matrices, std::vector<hostFloat>& phi, std::vector<hostFloat>& x) {
    for (size_t i = 0; i < elements_.size(); ++i) {
        const size_t offset_interp_1D = i * N_interpolation_points;

        for (size_t j = 0; j < N_interpolation_points; ++j) {
            phi[offset_interp_1D + j] = 0.0f;
            for (int k = 0; k <= elements_[i].N_; ++k) {
                phi[offset_interp_1D + j] += interpolation_matrices[elements_[i].N_][j * (elements_[i].N_ + 1) + k] * elements_[i].phi_[k];
            }
            x[offset_interp_1D + j] = j * (elements_[i].x_[1] - elements_[i].x_[0]) / (N_interpolation_points - 1) + elements_[i].x_[0];
        }
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