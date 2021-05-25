#ifndef NDG_ELEMENT2D_T_H
#define NDG_ELEMENT2D_T_H

#include "helpers/float_types.h"
#include "entities/cuda_vector.cuh"
#include "entities/Vec2.cuh"
#include <array>

namespace SEM { namespace Entities {
    class Element2D_t { // Turn this into separate vectors, because cache exists
        public:
            __device__ 
            Element2D_t(int N, std::array<SEM::Entities::cuda_vector<size_t>, 4> faces, std::array<size_t, 4> nodes);

            __host__ __device__
            Element2D_t();

            int N_;

            // Connectivity
            std::array<SEM::Entities::cuda_vector<size_t>, 4> faces_;
            std::array<size_t, 4> nodes_;

            // Geometry
            deviceFloat delta_xy_min_;
            SEM::Entities::cuda_vector<deviceFloat> dxi_dx_;
            SEM::Entities::cuda_vector<deviceFloat> deta_dx_;
            SEM::Entities::cuda_vector<deviceFloat> dxi_dy_;
            SEM::Entities::cuda_vector<deviceFloat> deta_dy_;
            SEM::Entities::cuda_vector<deviceFloat> jacobian_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> scaling_factor_;

            // Solution
            SEM::Entities::cuda_vector<deviceFloat> p_;
            SEM::Entities::cuda_vector<deviceFloat> u_;
            SEM::Entities::cuda_vector<deviceFloat> v_;
            SEM::Entities::cuda_vector<deviceFloat> G_p_;
            SEM::Entities::cuda_vector<deviceFloat> G_u_;
            SEM::Entities::cuda_vector<deviceFloat> G_v_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> p_extrapolated_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> u_extrapolated_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> v_extrapolated_;
            SEM::Entities::cuda_vector<deviceFloat> p_flux_;
            SEM::Entities::cuda_vector<deviceFloat> u_flux_;
            SEM::Entities::cuda_vector<deviceFloat> v_flux_;
            SEM::Entities::cuda_vector<deviceFloat> p_flux_derivative_;
            SEM::Entities::cuda_vector<deviceFloat> u_flux_derivative_;
            SEM::Entities::cuda_vector<deviceFloat> v_flux_derivative_;

            deviceFloat sigma_;
            bool refine_;
            bool coarsen_;
            deviceFloat error_;

            // Algorithm 61
            __device__
            auto interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

            // Algorithm 61
            __device__
            auto interpolate_q_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

            template<typename Polynomial>
            __device__
            auto estimate_error(const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

            __device__
            auto interpolate_from(const Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

            __device__
            auto interpolate_solution(size_t N_interpolation_points, const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void;

            __device__
            auto allocate_storage() -> void;

            __device__
            auto allocate_boundary_storage() -> void;

        private:
            __device__
            auto exponential_decay() -> deviceFloat;
    };
}}

#endif
