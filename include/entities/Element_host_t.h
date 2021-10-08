#ifndef NDG_ELEMENT_HOST_T_H
#define NDG_ELEMENT_HOST_T_H

#include "helpers/float_types.h"
#include <array>
#include <vector>

namespace SEM { namespace Entities {
    class Element_host_t { // Turn this into separate vectors, because cache exists
        public:
            Element_host_t(int N, std::size_t face_L, std::size_t face_R, hostFloat x_L, hostFloat x_R);
            Element_host_t() = default;

            int N_;
            std::array<std::size_t, 2> faces_; // Could also be pointers. left, right
            std::array<hostFloat, 2> x_;
            hostFloat delta_x_;
            hostFloat phi_L_;
            hostFloat phi_R_;
            hostFloat phi_prime_L_;
            hostFloat phi_prime_R_;
            std::vector<hostFloat> phi_; // Solution
            std::vector<hostFloat> q_;
            std::vector<hostFloat> ux_;
            std::vector<hostFloat> phi_prime_;
            std::vector<hostFloat> intermediate_;

            hostFloat sigma_;
            bool refine_;
            bool coarsen_;
            hostFloat error_;

            // Algorithm 61
            void interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

            // Algorithm 61
            void interpolate_q_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

            template<typename Polynomial>
            void estimate_error(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);

            void interpolate_from(const Element_host_t& other, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights);

        private: 
            // Basically useless, find better solution when multiple elements.
            static void get_elements_data(std::size_t N_elements, const Element_host_t* elements, hostFloat* phi, hostFloat* phi_prime);

            // Basically useless, find better solution when multiple elements.
            static void get_phi(std::size_t N_elements, const Element_host_t* elements, hostFloat* phi);

            hostFloat exponential_decay();

            // From cppreference.com
            static bool almost_equal(hostFloat x, hostFloat y);
    };
}}

#endif
