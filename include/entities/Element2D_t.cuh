#ifndef NDG_ENTITIES_ELEMENT2D_T_CUH
#define NDG_ENTITIES_ELEMENT2D_T_CUH

#include "helpers/float_types.h"
#include "entities/cuda_vector.cuh"
#include "entities/Vec2.cuh"
#include "functions/Hilbert_splitting.cuh"
#include <array>
#include <mpi.h>

namespace SEM { namespace Device { namespace Entities {
    class Element2D_t { // Turn this into separate vectors, because cache exists
        public:
            __device__ 
            Element2D_t(int N, int split_level, SEM::Device::Hilbert::Status status, int rotation, const std::array<SEM::Device::Entities::cuda_vector<size_t>, 4>& faces, std::array<size_t, 4> nodes);

            __host__ __device__
            Element2D_t();

            int N_;

            // Connectivity
            std::array<SEM::Device::Entities::cuda_vector<size_t>, 4> faces_;
            std::array<size_t, 4> nodes_;

            // Geometry
            SEM::Device::Hilbert::Status status_;
            deviceFloat delta_xy_min_;
            SEM::Device::Entities::Vec2<deviceFloat> center_;
            SEM::Device::Entities::cuda_vector<deviceFloat> dxi_dx_;
            SEM::Device::Entities::cuda_vector<deviceFloat> deta_dx_;
            SEM::Device::Entities::cuda_vector<deviceFloat> dxi_dy_;
            SEM::Device::Entities::cuda_vector<deviceFloat> deta_dy_;
            SEM::Device::Entities::cuda_vector<deviceFloat> jacobian_;
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 4> scaling_factor_;
            int rotation_;

            // Solution
            SEM::Device::Entities::cuda_vector<deviceFloat> p_; /**< @brief Pressure in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> u_; /**< @brief x velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> v_; /**< @brief y velocity in the element. Sized N + 1 by N + 1, index with i * (N + 1) + j.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> G_p_;
            SEM::Device::Entities::cuda_vector<deviceFloat> G_u_;
            SEM::Device::Entities::cuda_vector<deviceFloat> G_v_;
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 4> p_extrapolated_;
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 4> u_extrapolated_;
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 4> v_extrapolated_;
            SEM::Device::Entities::cuda_vector<deviceFloat> p_flux_; /**< @brief Pressure flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> u_flux_; /**< @brief x velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> v_flux_; /**< @brief y velocity flux in the element, used to extrapolate line by line to boundaries. Sized N + 1.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> p_flux_derivative_;
            SEM::Device::Entities::cuda_vector<deviceFloat> u_flux_derivative_;
            SEM::Device::Entities::cuda_vector<deviceFloat> v_flux_derivative_;
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 4> p_flux_extrapolated_; /**< @brief Pressure flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 4> u_flux_extrapolated_; /**< @brief x velocity flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            std::array<SEM::Device::Entities::cuda_vector<deviceFloat>, 4> v_flux_extrapolated_; /**< @brief y velocity flux on the element edges, projected from the faces. One vector per side, sized N + 1.*/
            SEM::Device::Entities::cuda_vector<deviceFloat> p_intermediate_;
            SEM::Device::Entities::cuda_vector<deviceFloat> u_intermediate_;
            SEM::Device::Entities::cuda_vector<deviceFloat> v_intermediate_;
            SEM::Device::Entities::cuda_vector<deviceFloat> spectrum_;

            // Error
            bool refine_;
            bool coarsen_;
            deviceFloat p_error_;
            deviceFloat u_error_;
            deviceFloat v_error_;
            deviceFloat p_sigma_;
            deviceFloat u_sigma_;
            deviceFloat v_sigma_;

            // Adaptivity
            int split_level_;
            std::array<bool, 4> additional_nodes_;

            class Datatype {
                private:
                   MPI_Datatype datatype_;
                   
                public:
                    __host__
                    Datatype();

                    __host__
                    ~Datatype();

                    __host__
                    auto data() const -> const MPI_Datatype&;

                    __host__
                    auto data() -> MPI_Datatype&;
            };

            // Algorithm 61
            __device__
            auto interpolate_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

            // Algorithm 61
            __device__
            auto interpolate_q_to_boundaries(const deviceFloat* lagrange_interpolant_minus, const deviceFloat* lagrange_interpolant_plus) -> void;

            template<typename Polynomial>
            __device__
            auto estimate_error(deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomials) -> void;

            template<typename Polynomial>
            __device__
            auto estimate_p_error(deviceFloat tolerance_min, deviceFloat tolerance_max, const deviceFloat* polynomials) -> void;

            // This is used when the elements have different points
            __device__
            auto interpolate_from(const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 4>& points, const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 4>& points_other, const Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

            // This is used when the elements have the same points
            __device__
            auto interpolate_from(const Element2D_t& other, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void;

            __device__
            auto interpolate_solution(size_t n_interpolation_points, const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 4>& points, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) const -> void;

            __device__
            auto interpolate_complete_solution(size_t n_interpolation_points, deviceFloat time, const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 4>& points, const deviceFloat* polynomial_nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v, deviceFloat* dp_dt, deviceFloat* du_dt, deviceFloat* dv_dt, deviceFloat* p_analytical_error, deviceFloat* u_analytical_error, deviceFloat* v_analytical_error) const -> void;

            __device__
            auto allocate_storage() -> void;

            __device__
            auto allocate_boundary_storage() -> void;

            __device__
            auto resize_boundary_storage(int N) -> void;

            __device__
            auto compute_geometry(const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 4>& points, const deviceFloat* polynomial_nodes) -> void;

            __device__
            auto compute_boundary_geometry(const std::array<SEM::Device::Entities::Vec2<deviceFloat>, 4>& points, const deviceFloat* polynomial_nodes) -> void;

            // From cppreference.com
            __device__
            static auto almost_equal(deviceFloat x, deviceFloat y) -> bool;

            __device__
            auto would_p_refine(int max_N) const -> bool;

            __device__
            auto would_h_refine(int max_split_level) const -> bool;

            __host__ __device__
            auto clear_storage() -> void;

        private:
            __device__
            auto exponential_decay(int n_points_least_squares) -> std::array<deviceFloat, 2>;
    };
}}}

#include "entities/Element2D_t.tcu"

#endif
