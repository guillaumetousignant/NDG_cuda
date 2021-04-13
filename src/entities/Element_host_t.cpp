#include "entities/Element_host_t.h"
#include "polynomials/ChebyshevPolynomial_host_t.h"
#include "polynomials/LegendrePolynomial_host_t.h"
#include <cmath>

SEM::Entities::Element_host_t::Element_host_t(int N, size_t face_L, size_t face_R, hostFloat x_L, hostFloat x_R) : 
        N_(N),
        faces_{face_L, face_R},
        x_{x_L, x_R},
        delta_x_(x_R - x_L),
        phi_(N_ + 1),
        q_(N_ + 1),
        ux_(N_ + 1),
        phi_prime_(N_ + 1),
        intermediate_(N_ + 1, 0.0),
        sigma_(0.0),
        refine_(false),
        coarsen_(false),
        error_(0.0) {}

// Basically useless, find better solution when multiple elements.
void SEM::Entities::Element_host_t::get_elements_data(size_t N_elements, const Element_host_t* elements, hostFloat* phi, hostFloat* phi_prime) {
    for (size_t i = 0; i < N_elements; ++i) {
        const size_t element_offset = i * (elements[i].N_ + 1);
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[element_offset + j] = elements[i].phi_[j];
            phi_prime[element_offset + j] = elements[i].phi_prime_[j];
        }
    }
}

// Basically useless, find better solution when multiple elements.
void SEM::Entities::Element_host_t::get_phi(size_t N_elements, const Element_host_t* elements, hostFloat* phi) {
    for (size_t i = 0; i < N_elements; ++i) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
        }
    }
}

// Algorithm 61
void SEM::Entities::Element_host_t::interpolate_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    phi_L_ = 0.0;
    phi_R_ = 0.0;

    for (int j = 0; j <= N_; ++j) {
        phi_L_ += lagrange_interpolant_left[N_][j] * phi_[j];
        phi_R_ += lagrange_interpolant_right[N_][j] * phi_[j];
    }
}

// Algorithm 61
void SEM::Entities::Element_host_t::interpolate_q_to_boundaries(const std::vector<std::vector<hostFloat>>& lagrange_interpolant_left, const std::vector<std::vector<hostFloat>>& lagrange_interpolant_right) {
    phi_prime_L_ = 0.0;
    phi_prime_R_ = 0.0;

    for (int j = 0; j <= N_; ++j) {
        phi_prime_L_ += lagrange_interpolant_left[N_][j] * q_[j];
        phi_prime_R_ += lagrange_interpolant_right[N_][j] * q_[j];
    }
}

template void SEM::Entities::Element_host_t::estimate_error<SEM::Polynomials::ChebyshevPolynomial_host_t>(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);
template void SEM::Entities::Element_host_t::estimate_error<SEM::Polynomials::LegendrePolynomial_host_t>(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights);

template<typename Polynomial>
void SEM::Entities::Element_host_t::estimate_error(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights) {
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

hostFloat SEM::Entities::Element_host_t::exponential_decay() {
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

void SEM::Entities::Element_host_t::interpolate_from(const SEM::Entities::Element_host_t& other, const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& barycentric_weights) {
    for (int i = 0; i <= N_; ++i) {
        const hostFloat x = (x_[1] - x_[0]) * (nodes[N_][i] + 1) * 0.5 + x_[0];
        const hostFloat node = (2 * x - other.x_[0] - other.x_[1])/(other.x_[1] - other.x_[0]);
        hostFloat numerator = 0.0;
        hostFloat denominator = 0.0;
        for (int j = 0; j <= other.N_; ++j) {
            if (almost_equal(node, nodes[N_][j])) {
                numerator = other.phi_[j];
                denominator = 1.0;
                break;
            }
            const hostFloat t = barycentric_weights[other.N_][j]/(node - nodes[N_][j]);
            numerator += t * other.phi_[j];
            denominator += t;
        }
        phi_[i] = numerator/denominator;
    }
}

bool SEM::Entities::Element_host_t::almost_equal(hostFloat x, hostFloat y) {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= FLT_EPSILON * std::abs(x+y) * ulp // CHECK change this to double equivalent if using double instead of float
        // unless the result is subnormal
        || std::abs(x-y) < FLT_MIN; // CHECK change this to 64F if using double instead of float
}
