#include "Element_host_t.h"
#include "ChebyshevPolynomial_host_t.h"
#include "LegendrePolynomial_host_t.h"
#include <cmath>

SEM::Element_host_t::Element_host_t(int N, size_t face_L, size_t face_R, hostFloat x_L, hostFloat x_R) : 
        N_(N),
        faces_{face_L, face_R},
        x_{x_L, x_R},
        delta_x_(x_R - x_L),
        phi_(N_ + 1),
        phi_prime_(N_ + 1),
        intermediate_(N_ + 1, 0.0),
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {}

SEM::Element_host_t::Element_host_t() {};

SEM::Element_host_t::~Element_host_t() {}

// Basically useless, find better solution when multiple elements.
void SEM::Element_host_t::get_elements_data(size_t N_elements, const Element_host_t* elements, hostFloat* phi, hostFloat* phi_prime) {
    for (size_t i = 0; i < N_elements; ++i) {
        const size_t element_offset = i * (elements[i].N_ + 1);
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[element_offset + j] = elements[i].phi_[j];
            phi_prime[element_offset + j] = elements[i].phi_prime_[j];
        }
    }
}

// Basically useless, find better solution when multiple elements.
void SEM::Element_host_t::get_phi(size_t N_elements, const Element_host_t* elements, hostFloat* phi) {
    for (size_t i = 0; i < N_elements; ++i) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
        }
    }
}

// Algorithm 61
void SEM::Element_host_t::interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    phi_L_ = 0.0;
    phi_R_ = 0.0;

    for (int j = 0; j <= N_; ++j) {
        phi_L_ += lagrange_interpolant_left[N_][j] * phi_[j];
        phi_R_ += lagrange_interpolant_right[N_][j] * phi_[j];
    }
}

template void SEM::Element_host_t::estimate_error<SEM::ChebyshevPolynomial_host_t>(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);
template void SEM::Element_host_t::estimate_error<SEM::LegendrePolynomial_host_t>(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);

template<typename Polynomial>
void SEM::Element_host_t::estimate_error<Polynomial>(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights) {
    for (int k = 0; k <= N_; ++k) {
        intermediate_[k] = 0.0;
        for (int i = 0; i <= N_; ++i) {
            const hostFloat L_N = Polynomial::polynomial(k, nodes[N_][i]);

            intermediate_[k] += (2 * k + 1) * 0.5 * phi_[i] * L_N * weights[N_][i];
        }
        intermediate_[k] = std::abs(intermediate_[k]);
    }

    constexpr hostFloat tolerance_min = 1e25;     // Refine above this
    constexpr hostFloat tolerance_max = 1e-25;    // Coarsen below this

    const hostFloat C = exponential_decay();

    // sum of error
    error_ = std::sqrt(std::pow(C, 2) * 0.5/sigma_) * std::exp(-sigma_ * (N_ + 1));

    if(error_ > tolerance_min) {	// need refine
        refine_ = true;
        coarsen_ = false;
    }
    else if(error_ <= tolerance_max ) {	// need coarsen
        refine_ = false;
        coarsen_ = true;
    }
    else {	// if error in between then do nothing
        refine_ = false;
        coarsen_ = false;
    }
}

hostFloat SEM::Element_host_t::exponential_decay() {
    const int n_points_least_squares = std::min(N_, 4); // Number of points to use for thew least squares reduction, but don't go above N.

    hostFloat x_avg = 0.0;
    hostFloat y_avg = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        x_avg += N_ - i;
        y_avg += std::log(intermediate_[N_ - i]);
    }

    x_avg /= n_points_least_squares;
    y_avg /= n_points_least_squares;

    hostFloat numerator = 0.0;
    hostFloat denominator = 0.0;

    for (int i = 0; i < n_points_least_squares; ++i) {
        numerator += (N_ - i - x_avg) * (std::log(intermediate_[N_ - i]) - y_avg);
        denominator += (N_ - i - x_avg) * (N_ - i - x_avg);
    }

    sigma_ = numerator/denominator;

    const hostFloat C = std::exp(y_avg - sigma_ * x_avg);
    sigma_ = std::abs(sigma_);
    return C;
}
