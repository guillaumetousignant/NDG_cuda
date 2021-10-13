#ifndef NDG_ELEMENT2D_HOST_T_H
#define NDG_ELEMENT2D_HOST_T_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include "functions/Hilbert_splitting.cuh"
#include <array>
#include <vector>
#include <mpi.h>

namespace SEM { namespace Entities {
    class Element2D_host_t { // Turn this into separate vectors, because cache exists
        public: 
            Element2D_host_t(int N, int split_level, SEM::Hilbert::Status status, int rotation, const std::array<std::vector<size_t>, 4>& faces, std::array<size_t, 4> nodes);

            Element2D_host_t();

            int N_;

            // Connectivity
            std::array<std::vector<size_t>, 4> faces_;
            std::array<size_t, 4> nodes_;

            // Geometry
            SEM::Hilbert::Status status_;
            hostFloat delta_xy_min_;
            SEM::Entities::Vec2<hostFloat> center_;
            std::vector<hostFloat> dxi_dx_;
            std::vector<hostFloat> deta_dx_;
            std::vector<hostFloat> dxi_dy_;
            std::vector<hostFloat> deta_dy_;
            std::vector<hostFloat> jacobian_;
            std::array<std::vector<hostFloat>, 4> scaling_factor_;
            int rotation_;

            // Solution
            std::vector<hostFloat> p_; /**< @brief Pressure in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            std::vector<hostFloat> u_; /**< @brief x velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            std::vector<hostFloat> v_; /**< @brief y velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            std::vector<hostFloat> G_p_;
            std::vector<hostFloat> G_u_;
            std::vector<hostFloat> G_v_;
            std::array<std::vector<hostFloat>, 4> p_extrapolated_;
            std::array<std::vector<hostFloat>, 4> u_extrapolated_;
            std::array<std::vector<hostFloat>, 4> v_extrapolated_;
            std::vector<hostFloat> p_flux_; /**< @brief Pressure flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            std::vector<hostFloat> u_flux_; /**< @brief x velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            std::vector<hostFloat> v_flux_; /**< @brief y velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            std::vector<hostFloat> p_flux_derivative_;
            std::vector<hostFloat> u_flux_derivative_;
            std::vector<hostFloat> v_flux_derivative_;
            std::array<std::vector<hostFloat>, 4> p_flux_extrapolated_; /**< @brief Pressure flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            std::array<std::vector<hostFloat>, 4> u_flux_extrapolated_; /**< @brief x velocity flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            std::array<std::vector<hostFloat>, 4> v_flux_extrapolated_; /**< @brief y velocity flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            std::vector<hostFloat> p_intermediate_;
            std::vector<hostFloat> u_intermediate_;
            std::vector<hostFloat> v_intermediate_;
            std::vector<hostFloat> spectrum_;

            // Error
            bool refine_;
            bool coarsen_;
            hostFloat p_error_;
            hostFloat u_error_;
            hostFloat v_error_;
            hostFloat p_sigma_;
            hostFloat u_sigma_;
            hostFloat v_sigma_;

            // Adaptivity
            int split_level_;
            std::array<bool, 4> additional_nodes_;

            class Datatype {
                private:
                   MPI_Datatype datatype_;
                   
                public:
                    Datatype();

                    ~Datatype();

                    auto data() const -> const MPI_Datatype&;

                    auto data() -> MPI_Datatype&;
            };

            // Algorithm 61
            auto interpolate_to_boundaries(const std::vector<hostFloat>& lagrange_interpolant_minus, const std::vector<hostFloat>& lagrange_interpolant_plus) -> void;

            // Algorithm 61
            auto interpolate_q_to_boundaries(const std::vector<hostFloat>& lagrange_interpolant_minus, const std::vector<hostFloat>& lagrange_interpolant_plus) -> void;

            template<typename Polynomial>
            auto estimate_error(hostFloat tolerance_min, hostFloat tolerance_max, const std::vector<hostFloat>& polynomial_nodes, const std::vector<hostFloat>& weights) -> void;

            // This is used when the elements have different points
            auto interpolate_from(const std::array<SEM::Entities::Vec2<hostFloat>, 4>& points, const std::array<SEM::Entities::Vec2<hostFloat>, 4>& points_other, const Element2D_host_t& other, const std::vector<hostFloat>& polynomial_nodes, const std::vector<hostFloat>& barycentric_weights) -> void;

            // This is used when the elements have the same points
            auto interpolate_from(const Element2D_host_t& other, const std::vector<hostFloat>& polynomial_nodes, const std::vector<hostFloat>& barycentric_weights) -> void;

            auto interpolate_solution(size_t n_interpolation_points, const std::array<SEM::Entities::Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& interpolation_matrices, std::vector<hostFloat>& x, std::vector<hostFloat>& y, std::vector<hostFloat>& p, std::vector<hostFloat>& u, std::vector<hostFloat>& v) const -> void;

            auto interpolate_complete_solution(size_t n_interpolation_points, hostFloat time, const std::array<SEM::Entities::Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& polynomial_nodes, const std::vector<hostFloat>& interpolation_matrices, std::vector<hostFloat>& x, std::vector<hostFloat>& y, std::vector<hostFloat>& p, std::vector<hostFloat>& u, std::vector<hostFloat>& v, std::vector<hostFloat>& dp_dt, std::vector<hostFloat>& du_dt, std::vector<hostFloat>& dv_dt, std::vector<hostFloat>& p_analytical_error, std::vector<hostFloat>& u_analytical_error, std::vector<hostFloat>& v_analytical_error) const -> void;

            auto allocate_storage() -> void;

            auto allocate_boundary_storage() -> void;

            auto resize_boundary_storage(int N) -> void;

            auto compute_geometry(const std::array<SEM::Entities::Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& polynomial_nodes) -> void;

            auto compute_boundary_geometry(const std::array<SEM::Entities::Vec2<hostFloat>, 4>& points, const std::vector<hostFloat>& polynomial_nodes) -> void;

            // From cppreference.com
            static auto almost_equal(hostFloat x, hostFloat y) -> bool;

            auto would_p_refine(int max_N) const -> bool;

            auto would_h_refine(int max_split_level) const -> bool;

        private:
            auto exponential_decay(int n_points_least_squares) -> std::array<hostFloat, 2>;
    };
}}

#endif
