#ifndef NDG_SOLVERS_SOLVER2D_T_H
#define NDG_SOLVERS_SOLVER2D_T_H

#include "helpers/float_types.h"
#include "entities/NDG_t.h"
#include "helpers/DataWriter_t.h"
#include "meshes/Mesh2D_t.h"
#include "entities/Face2D_t.h"
#include "entities/Element2D_t.h"
#include <vector>
#include <array>

namespace SEM { namespace Host { namespace Solvers {
    class Solver2D_t {
        public:
            Solver2D_t(hostFloat CFL, std::vector<hostFloat> output_times, hostFloat viscosity);

            hostFloat CFL_;
            hostFloat viscosity_;
            std::vector<hostFloat> output_times_;

            template<typename Polynomial>
            auto solve(const SEM::Host::Entities::NDG_t<Polynomial> &NDG, SEM::Host::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) const -> void;

            template<typename Polynomial>
            auto pre_condition(const SEM::Host::Entities::NDG_t<Polynomial> &NDG, SEM::Host::Meshes::Mesh2D_t& mesh, size_t n_adaptivity_steps, size_t pre_condition_interval) const -> void;
            
            template<typename Polynomial>
            auto pre_condition_iterative(const SEM::Host::Entities::NDG_t<Polynomial> &NDG, SEM::Host::Meshes::Mesh2D_t& mesh, size_t n_adaptivity_steps, size_t pre_condition_interval) const -> void;


            auto get_delta_t(SEM::Host::Meshes::Mesh2D_t& mesh) const -> hostFloat;

            static auto x_flux(hostFloat p, hostFloat u, hostFloat v) -> std::array<hostFloat, 3>;

            static auto y_flux(hostFloat p, hostFloat u, hostFloat v) -> std::array<hostFloat, 3>;

            // Algorithm 19
            static auto matrix_vector_multiply(int N, const std::vector<hostFloat>& matrix, const std::vector<hostFloat>& vector, std::vector<hostFloat>& result) -> void;

            static auto calculate_wave_fluxes(std::vector<SEM::Host::Entities::Face2D_t>& faces) -> void;
    };

    // Algorithm 114
    auto compute_dg_wave_derivative(size_t N_elements, SEM::Host::Entities::Element2D_t* elements, const SEM::Host::Entities::Face2D_t* faces, const std::vector<std::vector<hostFloat>>& weights, const std::vector<std::vector<hostFloat>>& derivative_matrices_hat, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) -> void;

    auto rk3_first_step(size_t N_elements, SEM::Host::Entities::Element2D_t* elements, hostFloat delta_t, hostFloat g) -> void;

    auto rk3_step(size_t N_elements, SEM::Host::Entities::Element2D_t* elements, hostFloat delta_t, hostFloat a, hostFloat g) -> void;
}}}

#include "solvers/Solver2D_t.tpp"

#endif
