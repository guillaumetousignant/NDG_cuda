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
            Element2D_t(int N, int split_level, const std::array<SEM::Entities::cuda_vector<size_t>, 4>& faces, std::array<size_t, 4> nodes);

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
            SEM::Entities::cuda_vector<deviceFloat> p_; /**< @brief Pressure in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            SEM::Entities::cuda_vector<deviceFloat> u_; /**< @brief x velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            SEM::Entities::cuda_vector<deviceFloat> v_; /**< @brief y velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            SEM::Entities::cuda_vector<deviceFloat> G_p_;
            SEM::Entities::cuda_vector<deviceFloat> G_u_;
            SEM::Entities::cuda_vector<deviceFloat> G_v_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> p_extrapolated_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> u_extrapolated_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> v_extrapolated_;
            SEM::Entities::cuda_vector<deviceFloat> p_flux_; /**< @brief Pressure flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Entities::cuda_vector<deviceFloat> u_flux_; /**< @brief x velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Entities::cuda_vector<deviceFloat> v_flux_; /**< @brief y velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Entities::cuda_vector<deviceFloat> p_flux_derivative_;
            SEM::Entities::cuda_vector<deviceFloat> u_flux_derivative_;
            SEM::Entities::cuda_vector<deviceFloat> v_flux_derivative_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> p_flux_extrapolated_; /**< @brief Pressure flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> u_flux_extrapolated_; /**< @brief x velocity flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 4> v_flux_extrapolated_; /**< @brief y velocity flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            SEM::Entities::cuda_vector<deviceFloat> p_intermediate_;
            SEM::Entities::cuda_vector<deviceFloat> u_intermediate_;
            SEM::Entities::cuda_vector<deviceFloat> v_intermediate_;
            SEM::Entities::cuda_vector<deviceFloat> spectrum_;

            bool refine_;
            bool coarsen_;
            deviceFloat p_error_;
            deviceFloat u_error_;
            deviceFloat v_error_;
            deviceFloat p_sigma_;
            deviceFloat u_sigma_;
            deviceFloat v_sigma_;
            int split_level_;

            // Algorithm 61
            __device__
            auto interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

            // Algorithm 61
            __device__
            auto interpolate_q_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

            template<typename Polynomial>
            __device__
            auto estimate_error(const deviceFloat* polynomial_nodes, const deviceFloat* weights) -> void;

            // This is used when the elements have different points
            __device__
            auto interpolate_from(const std::array<Vec2<deviceFloat>, 4>& points, const std::array<Vec2<deviceFloat>, 4>& points_other, const Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

            // This is used when the elements have the same points
            __device__
            auto interpolate_from(const Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

            __device__
            auto interpolate_solution(size_t N_interpolation_points, const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) const -> void;

            __device__
            auto interpolate_complete_solution(size_t N_interpolation_points, const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt) const -> void;

            __device__
            auto allocate_storage() -> void;

            __device__
            auto allocate_boundary_storage() -> void;

            __device__
            auto resize_boundary_storage(int N) -> void;

            __device__
            auto compute_element_geometry(const std::array<Vec2<deviceFloat>, 4>& points, const deviceFloat* polynomial_nodes) -> void;

            // From cppreference.com
            __device__
            static auto almost_equal(deviceFloat x, deviceFloat y) -> bool;

        private:
            __device__
            auto exponential_decay(int n_points_least_squares) -> std::array<deviceFloat, 2>;
    };
}}

#endif
