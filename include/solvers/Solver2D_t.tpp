#include "helpers/ProgressBar_t.h"
#include "helpers/constants.h"
#define NOMINMAX
#include "helpers/termcolor.hpp"

template<typename Polynomial>
auto SEM::Host::Solvers::Solver2D_t::solve(const SEM::Host::Entities::NDG_t<Polynomial> &NDG, SEM::Host::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) const -> void {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    hostFloat time{0};
    const hostFloat t_end = output_times_.back();
    SEM::Helpers::ProgressBar_t bar;
    size_t timestep{0};
    constexpr std::array<hostFloat, 3> am {0, -5.0/9.0, -153.0/128.0};
    constexpr std::array<hostFloat, 3> bm {0, 1.0/3.0, 0.75};
    constexpr std::array<hostFloat, 3> gm {1.0/3.0, 15.0/16.0, 8.0/15.0};

    hostFloat delta_t = get_delta_t(mesh);

    for (auto const& e : std::as_const(output_times_)) {
        if ((time >= e) && (time < e + delta_t)) {
            if (global_rank == 0) {
                bar.set_status_text("Writing solution");
                bar.update(0.0);
            }
            mesh.write_complete_data(time, NDG.nodes_, NDG.interpolation_matrices_, data_writer);
        }
    }
    
    if (global_rank == 0) {
        bar.set_status_text("Iteration 0");
        bar.update(0.0);
    }
    
    while (time < t_end) {
        ++timestep;
        delta_t = get_delta_t(mesh);
        if (time + delta_t > t_end) {
            delta_t = t_end - time;
        }

        // Kinda algorithm 62
        hostFloat t = time + bm[0] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_first_step(mesh.n_elements_, mesh.elements_.data(), delta_t, gm[0]);

        t = time + bm[1] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_step(mesh.n_elements_, mesh.elements_.data(), delta_t, am[1], gm[1]);

        t = time + bm[2] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_step(mesh.n_elements_, mesh.elements_.data(), delta_t, am[2], gm[2]);
        
        time += delta_t;
        for (auto const& e : std::as_const(output_times_)) {
            if ((time >= e) && (time < e + delta_t)) {
                if (global_rank == 0) {
                    bar.set_status_text("Writing solution");
                    bar.update(time/t_end);
                }
                mesh.estimate_error(NDG.polynomials_);
                mesh.write_complete_data(time, NDG.nodes_, NDG.interpolation_matrices_, data_writer);
                break;
            }
        }

        if (mesh.adaptivity_interval_ > 0 && timestep % mesh.adaptivity_interval_ == 0) {
            if (global_rank == 0) {
                bar.set_status_text("Adapting");
                bar.update(time/t_end);
            }

            mesh.estimate_error(NDG.polynomials_);
            mesh.adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
        }

        if (mesh.load_balancing_interval_ > 0 && global_size > 1 && timestep % mesh.load_balancing_interval_ == 0) {
            if (global_rank == 0) {
                bar.set_status_text("Load Balancing");
                bar.update(time/t_end);
            }

            mesh.load_balance(NDG.nodes_);
        }

        if (global_rank == 0) {
            std::stringstream ss;
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
            bar.update(time/t_end);
        }
    }

    bool did_write = false;
    for (auto const& e : std::as_const(output_times_)) {
        if ((time >= e) && (time < e + delta_t)) {
            did_write = true;
            break;
        }
    }

    if (!did_write) {
        mesh.estimate_error(NDG.polynomials_);
        if (global_rank == 0) {
            bar.set_status_text("Writing solution");
            bar.update(1.0);
        }
        mesh.write_complete_data(time, NDG.nodes_, NDG.interpolation_matrices_, data_writer);
    }
    if (global_rank == 0) {
        bar.set_status_text("Done");
        bar.update(1.0);
    }
    if (global_rank == 0) {
        std::cout << std::endl;
    }
}

template<typename Polynomial>
auto SEM::Host::Solvers::Solver2D_t::pre_condition(const SEM::Host::Entities::NDG_t<Polynomial> &NDG, SEM::Host::Meshes::Mesh2D_t& mesh, size_t n_adaptivity_steps, size_t pre_condition_interval) const -> void {
    if (pre_condition_interval == 0) {
        std::cerr << "Error, pre-condition should not be used with a pre-condition interval of 0. Exiting." << std::endl;
        exit(77);
    }
    
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    hostFloat time{0};
    const hostFloat t_end = output_times_.back();
    SEM::Helpers::ProgressBar_t bar;
    size_t timestep{0};
    size_t n_adaptivity{0};
    constexpr std::array<hostFloat, 3> am {0, -5.0/9.0, -153.0/128.0};
    constexpr std::array<hostFloat, 3> bm {0, 1.0/3.0, 0.75};
    constexpr std::array<hostFloat, 3> gm {1.0/3.0, 15.0/16.0, 8.0/15.0};

    hostFloat delta_t = get_delta_t(mesh);

    if (global_rank == 0) {
        bar.set_colour(termcolor::green);
        bar.set_status_text("Iteration 0");
        bar.update(0.0);
    }
    
    while (n_adaptivity < n_adaptivity_steps) {
        ++timestep;
        delta_t = get_delta_t(mesh);

        // Kinda algorithm 62
        hostFloat t = time + bm[0] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_first_step(mesh.n_elements_, mesh.elements_.data(), delta_t, gm[0]);

        t = time + bm[1] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_step(mesh.n_elements_, mesh.elements_.data(), delta_t, am[1], gm[1]);

        t = time + bm[2] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_step(mesh.n_elements_, mesh.elements_.data(), delta_t, am[2], gm[2]);
        
        time += delta_t;

        if (timestep % pre_condition_interval == 0) {
            if (global_rank == 0) {
                bar.set_status_text("Adapting");
                bar.update(static_cast<hostFloat>(timestep)/static_cast<hostFloat>(n_adaptivity_steps * pre_condition_interval)); // CHECK change if adaptivity is spaced out another way
            }

            mesh.estimate_error(NDG.polynomials_);
            mesh.adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
            ++n_adaptivity;
        }

        if (mesh.load_balancing_interval_ > 0 && global_size > 1 && timestep % mesh.load_balancing_interval_ == 0) {
            if (global_rank == 0) {
                bar.set_status_text("Load Balancing");
                bar.update(static_cast<hostFloat>(timestep)/static_cast<hostFloat>(n_adaptivity_steps * pre_condition_interval)); // CHECK change if adaptivity is spaced out another way
            }

            mesh.load_balance(NDG.nodes_);
        }

        if (global_rank == 0) {
            std::stringstream ss;
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
            bar.update(static_cast<hostFloat>(timestep)/static_cast<hostFloat>(n_adaptivity_steps * pre_condition_interval)); // CHECK change if adaptivity is spaced out another way
        }
    }

    mesh.initial_conditions(NDG.nodes_);

    if (global_rank == 0) {
        bar.set_status_text("Done");
        bar.update(1.0);
    }
    if (global_rank == 0) {
        std::cout << std::endl;
    }
}

template<typename Polynomial>
auto SEM::Host::Solvers::Solver2D_t::pre_condition_iterative(const SEM::Host::Entities::NDG_t<Polynomial> &NDG, SEM::Host::Meshes::Mesh2D_t& mesh, size_t n_adaptivity_steps, size_t pre_condition_interval) const -> void {
    if (pre_condition_interval == 0) {
        std::cerr << "Error, pre-condition should not be used with a pre-condition interval of 0. Exiting." << std::endl;
        exit(77);
    }
    
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    hostFloat time{0};
    const hostFloat t_end = output_times_.back();
    SEM::Helpers::ProgressBar_t bar;
    size_t timestep{0};
    size_t n_adaptivity{0};
    constexpr std::array<hostFloat, 3> am {0, -5.0/9.0, -153.0/128.0};
    constexpr std::array<hostFloat, 3> bm {0, 1.0/3.0, 0.75};
    constexpr std::array<hostFloat, 3> gm {1.0/3.0, 15.0/16.0, 8.0/15.0};

    hostFloat delta_t = get_delta_t(mesh);

    if (global_rank == 0) {
        bar.set_colour(termcolor::green);
        bar.set_status_text("Iteration 0");
        bar.update(0.0);
    }
    
    while (n_adaptivity < n_adaptivity_steps) {
        ++timestep;
        delta_t = get_delta_t(mesh);

        // Kinda algorithm 62
        hostFloat t = time + bm[0] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_first_step(mesh.n_elements_, mesh.elements_.data(), delta_t, gm[0]);

        t = time + bm[1] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_step(mesh.n_elements_, mesh.elements_.data(), delta_t, am[1], gm[1]);

        t = time + bm[2] * delta_t;
        mesh.interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        mesh.boundary_conditions(t, NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        mesh.project_to_faces(NDG.nodes_, NDG.barycentric_weights_);
        calculate_wave_fluxes(mesh.faces_);
        mesh.project_to_elements(NDG.nodes_, NDG.weights_, NDG.barycentric_weights_);
        SEM::Host::Solvers::compute_dg_wave_derivative(mesh.n_elements_, mesh.elements_.data(), mesh.faces_.data(), NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        SEM::Host::Solvers::rk3_step(mesh.n_elements_, mesh.elements_.data(), delta_t, am[2], gm[2]);
        
        time += delta_t;

        if (timestep % pre_condition_interval == 0) {
            if (global_rank == 0) {
                bar.set_status_text("Adapting");
                bar.update(static_cast<hostFloat>(timestep)/static_cast<hostFloat>(n_adaptivity_steps * pre_condition_interval)); // CHECK change if adaptivity is spaced out another way
            }

            mesh.estimate_error(NDG.polynomials_);
            mesh.adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
            ++n_adaptivity;

            mesh.initial_conditions(NDG.nodes_);
            time = deviceFloat{0};
        }

        if (mesh.load_balancing_interval_ > 0 && global_size > 1 && timestep % mesh.load_balancing_interval_ == 0) {
            if (global_rank == 0) {
                bar.set_status_text("Load Balancing");
                bar.update(static_cast<hostFloat>(timestep)/static_cast<hostFloat>(n_adaptivity_steps * pre_condition_interval)); // CHECK change if adaptivity is spaced out another way
            }

            mesh.load_balance(NDG.nodes_);
        }

        if (global_rank == 0) {
            std::stringstream ss;
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
            bar.update(static_cast<hostFloat>(timestep)/static_cast<hostFloat>(n_adaptivity_steps * pre_condition_interval)); // CHECK change if adaptivity is spaced out another way
        }
    }

    if (global_rank == 0) {
        bar.set_status_text("Done");
        bar.update(1.0);
    }
    if (global_rank == 0) {
        std::cout << std::endl;
    }
}