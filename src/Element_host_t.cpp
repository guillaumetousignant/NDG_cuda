#include "Element_host_t.h"

Element_host_t::Element_host_t(int N, size_t face_L, size_t face_R, hostFloat x_L, hostFloat x_R) : 
        N_(N),
        faces_{face_L, face_R},
        x_{x_L, x_R},
        delta_x_(x_R - x_L),
        phi_(N_ + 1),
        phi_prime_(N_ + 1),
        intermediate_(N_ + 1, 0.0) {}

Element_host_t::Element_host_t() {};

Element_host_t::~Element_host_t() {}

// Basically useless, find better solution when multiple elements.
void Element_host_t::get_elements_data(size_t N_elements, const Element_host_t* elements, hostFloat* phi, hostFloat* phi_prime) {
    for (size_t i = 0; i < N_elements; ++i) {
        const size_t element_offset = i * (elements[i].N_ + 1);
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[element_offset + j] = elements[i].phi_[j];
            phi_prime[element_offset + j] = elements[i].phi_prime_[j];
        }
    }
}

// Basically useless, find better solution when multiple elements.
void Element_host_t::get_phi(size_t N_elements, const Element_host_t* elements, hostFloat* phi) {
    for (size_t i = 0; i < N_elements; ++i) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
        }
    }
}

// Algorithm 61
void Element_host_t::interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    phi_L_ = 0.0;
    phi_R_ = 0.0;

    for (int j = 0; j <= N_; ++j) {
        phi_L_ += lagrange_interpolant_left[N_][j] * phi_[j];
        phi_R_ += lagrange_interpolant_right[N_][j] * phi_[j];
    }
}
