#ifndef NDG_ELEMENT2D_T_H
#define NDG_ELEMENT2D_T_H

#include "helpers/float_types.h"
#include "entities/cuda_vector.cuh"
#include <array>

namespace SEM { namespace Entities {
    class Element2D_t { // Turn this into separate vectors, because cache exists
        public:
            __device__ 
            Element2D_t(int N, std::array<size_t, 4> faces, std::array<size_t, 4> nodes);

            __host__ __device__
            Element2D_t();

            int N_;
            std::array<size_t, 4> faces_; // Could also be pointers. bottom, right, top, left
            std::array<size_t, 4> nodes_; // Could also be pointers. bottom, right, top, left
            std::array<deviceFloat, 2> delta_xy_;
            std::array<deviceFloat, 4> phi_extrapolated_;
            std::array<deviceFloat, 4> phi_prime_extrapolated_;
            SEM::Entities::cuda_vector<deviceFloat> p_;
            SEM::Entities::cuda_vector<deviceFloat> u_;
            SEM::Entities::cuda_vector<deviceFloat> v_;
            SEM::Entities::cuda_vector<deviceFloat> G_p_;
            SEM::Entities::cuda_vector<deviceFloat> G_u_;
            SEM::Entities::cuda_vector<deviceFloat> G_v_;

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
            auto estimate_error(const deviceFloat* nodes, const deviceFloat* weights) -> void;

            __device__
            auto interpolate_from(const Element2D_t& other, const deviceFloat* nodes, const deviceFloat* barycentric_weights) -> void;

        private:
            __device__
            auto exponential_decay() -> deviceFloat;
    };
}}

#endif
