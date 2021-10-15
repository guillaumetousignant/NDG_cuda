
template<typename Polynomial>
void SEM::Host::Meshes::Mesh_t::solve(const hostFloat CFL, const std::vector<hostFloat> output_times, const SEM::Host::Entities::NDG_t<Polynomial> &NDG, hostFloat viscosity) {
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    hostFloat time = 0.0;
    hostFloat t_end = output_times.back();
    SEM::Helpers::ProgressBar_t bar;
    size_t timestep = 0;

    hostFloat delta_t = get_delta_t(CFL);
    write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
    if (global_rank == 0) {
        bar.update(0.0);
        bar.set_status_text("Iteration 0");
    }

    while (time < t_end) {
        ++timestep;
        delta_t = get_delta_t(CFL);
        if (time + delta_t > t_end) {
            delta_t = t_end - time;
        }

        // Kinda algorithm 62
        hostFloat t = time;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        calculate_fluxes();
        compute_dg_derivative(NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        interpolate_q_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        calculate_q_fluxes();
        compute_dg_derivative2(viscosity, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_first_step(delta_t, 1.0/3.0);

        t = time + 0.33333333333 * delta_t;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        calculate_fluxes();
        compute_dg_derivative(NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        interpolate_q_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        calculate_q_fluxes();
        compute_dg_derivative2(viscosity, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step(delta_t, -5.0/9.0, 15.0/16.0);

        t = time + 0.75 * delta_t;
        interpolate_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        calculate_fluxes();
        compute_dg_derivative(NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        interpolate_q_to_boundaries(NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        boundary_conditions();
        calculate_q_fluxes();
        compute_dg_derivative2(viscosity, NDG.weights_, NDG.derivative_matrices_hat_, NDG.lagrange_interpolant_left_, NDG.lagrange_interpolant_right_);
        rk3_step(delta_t, -153.0/128.0, 8.0/15.0);

        time += delta_t;
        if (global_rank == 0) {
            std::stringstream ss;
            bar.update(time/t_end);
            ss << "Iteration " << timestep;
            bar.set_status_text(ss.str());
        }
        for (auto const& e : std::as_const(output_times)) {
            if ((time >= e) && (time < e + delta_t)) {
                estimate_error<Polynomial>(NDG.nodes_, NDG.weights_);
                write_data(time, NDG.N_interpolation_points_, NDG.interpolation_matrices_);
                break;
            }
        }

        if (timestep % adaptivity_interval_ == 0) {
            estimate_error<Polynomial>(NDG.nodes_, NDG.weights_);
            adapt(NDG.N_max_, NDG.nodes_, NDG.barycentric_weights_);
        }
    }
    if (global_rank == 0) {
        std::cout << std::endl;
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


template<typename Polynomial>
void SEM::Host::Meshes::Mesh_t::estimate_error(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights) {
    for (size_t i = 0; i < N_elements_; ++i) {
        elements_[i].estimate_error<Polynomial>(nodes, weights);
    }
}