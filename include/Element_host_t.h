#ifndef NDG_ELEMENT_HOST_T_H
#define NDG_ELEMENT_HOST_T_H

#include "float_types.h"
#include <array>
#include <vector>

class Element_t { // Turn this into separate vectors, because cache exists
    public:
        Element_t(int N, size_t neighbour_L, size_t neighbour_R, size_t face_L, size_t face_R, hostFloat x_L, hostFloat x_R);
        Element_t();
        ~Element_t();

        int N_;
        std::array<size_t, 2> neighbours_; // Could also be pointers
        std::array<size_t, 2> faces_; // Could also be pointers. left, right
        std::array<hostFloat, 2> x_;
        hostFloat delta_x_;
        hostFloat phi_L_;
        hostFloat phi_R_;
        std::vector<hostFloat> phi_; // Solution
        std::vector<hostFloat> phi_prime_;
        std::vector<hostFloat> intermediate_;

        // Algorithm 61
        void interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right);

    private: 
        // Basically useless, find better solution when multiple elements.
        static void get_elements_data(int N_elements, const Element_t* elements, hostFloat* phi, hostFloat* phi_prime);

        // Basically useless, find better solution when multiple elements.
        static void get_phi(int N_elements, const Element_t* elements, hostFloat* phi);

        static void interpolate_to_boundaries(int N_elements, Element_t* elements, const hostFloat* lagrange_interpolant_left, const hostFloat* lagrange_interpolant_right);
};

#endif