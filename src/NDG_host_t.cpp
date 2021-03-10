#include "NDG_host_t.h"
#include "ChebyshevPolynomial_host_t.h"
#include "LegendrePolynomial_host_t.h"
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>

template class SEM::NDG_host_t<SEM::ChebyshevPolynomial_host_t>; // Like, I understand why I need this, but man is it crap.
template class SEM::NDG_host_t<SEM::LegendrePolynomial_host_t>;

template<typename Polynomial>
SEM::NDG_host_t<Polynomial>::NDG_host_t(int N_max, size_t N_interpolation_points) : 
        N_max_(N_max), 
        N_interpolation_points_(N_interpolation_points),
        nodes_(N_max + 1),
        weights_(N_max + 1),
        barycentric_weights_(N_max + 1),
        lagrange_interpolant_left_(N_max + 1),
        lagrange_interpolant_right_(N_max + 1),
        lagrange_interpolant_derivative_left_(N_max + 1),
        lagrange_interpolant_derivative_right_(N_max + 1),
        derivative_matrices_(N_max + 1),
        derivative_matrices_hat_(N_max + 1),
        g_hat_derivative_matrices_(N_max + 1),
        interpolation_matrices_(N_max + 1) {

    for(int N = 0; N <= N_max; ++N) {
        nodes_[N] = std::vector<hostFloat>(N + 1);
        weights_[N] = std::vector<hostFloat>(N + 1);
        barycentric_weights_[N] = std::vector<hostFloat>(N + 1);
        lagrange_interpolant_left_[N] = std::vector<hostFloat>(N + 1);
        lagrange_interpolant_right_[N] = std::vector<hostFloat>(N + 1);
        lagrange_interpolant_derivative_left_[N] = std::vector<hostFloat>(N + 1);
        lagrange_interpolant_derivative_right_[N] = std::vector<hostFloat>(N + 1);
        derivative_matrices_[N] = std::vector<hostFloat>(std::pow(N + 1, 2));
        derivative_matrices_hat_[N] = std::vector<hostFloat>(std::pow(N + 1, 2));
        g_hat_derivative_matrices_[N] = std::vector<hostFloat>(std::pow(N + 1, 2));
        interpolation_matrices_[N] = std::vector<hostFloat>((N + 1) * N_interpolation_points_);
    }

    for(int N = 0; N <= N_max; ++N) {
        Polynomial::nodes_and_weights(N, nodes_[N], weights_[N]);
        calculate_barycentric_weights(N, nodes_[N], barycentric_weights_[N]);
        polynomial_derivative_matrices(N, nodes_[N], barycentric_weights_[N], derivative_matrices_[N]);
        create_interpolation_matrices(N, N_interpolation_points_, nodes_[N], barycentric_weights_[N], interpolation_matrices_[N]);
        lagrange_interpolating_polynomials(-1.0, N, nodes_[N], barycentric_weights_[N], lagrange_interpolant_left_[N]);
        lagrange_interpolating_polynomials(1.0, N, nodes_[N], barycentric_weights_[N], lagrange_interpolant_right_[N]);
        lagrange_interpolating_derivative_polynomials(-1.0, N, nodes_[N], barycentric_weights_[N], lagrange_interpolant_derivative_left_[N]);
        lagrange_interpolating_derivative_polynomials(1.0, N, nodes_[N], barycentric_weights_[N], lagrange_interpolant_derivative_right_[N]);
        normalize_lagrange_interpolating_polynomials(N, lagrange_interpolant_left_[N]);
        normalize_lagrange_interpolating_polynomials(N, lagrange_interpolant_right_[N]);
        normalize_lagrange_interpolating_derivative_polynomials(-1.0, N, nodes_[N], barycentric_weights_[N], lagrange_interpolant_derivative_left_[N]);
        normalize_lagrange_interpolating_derivative_polynomials(1.0, N, nodes_[N], barycentric_weights_[N], lagrange_interpolant_derivative_right_[N]);
        polynomial_derivative_matrices_diagonal(N, derivative_matrices_[N]);
        polynomial_derivative_matrices_hat(N, weights_[N], derivative_matrices_[N], derivative_matrices_hat_[N]);
        polynomial_cg_derivative_matrices(N, weights_[N], derivative_matrices_[N], g_hat_derivative_matrices_[N]);
    }
}

template<typename Polynomial>
SEM::NDG_host_t<Polynomial>::~NDG_host_t() {}
   
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::print() {
        std::cout << "Nodes: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << nodes_[N][i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Weights: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << weights_[N][i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl << "Barycentric weights: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << barycentric_weights_[N][i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants -1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << lagrange_interpolant_left_[N][i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants +1: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << lagrange_interpolant_right_[N][i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Derivative matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << derivative_matrices_[N][i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "CG derivative matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << g_hat_derivative_matrices_[N][i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "Derivative matrices hat: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (int i = 0; i <= N; ++i) {
            std::cout << '\t' << '\t';
            for (int j = 0; j <= N; ++j) {
                std::cout << derivative_matrices_hat_[N][i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "Interpolation matrices: " << std::endl;
    for (int N = 0; N <= N_max_; ++N) {
        std::cout << '\t' << "N = " << N << ": " << std::endl;
        for (size_t i = 0; i < N_interpolation_points_; ++i) {
            std::cout << '\t' << '\t';
            for (size_t j = 0; j <= N; ++j) {
                std::cout << interpolation_matrices_[N][i * (N + 1) + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

// Algorithm 30
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::calculate_barycentric_weights(int N, const std::vector<hostFloat>& nodes, std::vector<hostFloat>& barycentric_weights) {
    for (int j = 0; j <= N; ++j) {
        hostFloat xjxi = 1.0;
        for (int i = 0; i < j; ++i) {
            xjxi *= nodes[j] - nodes[i];
        }
        for (int i = j + 1; i <= N; ++i) {
            xjxi *= nodes[j] - nodes[i];
        }

        barycentric_weights[j] = 1.0/xjxi;
    }
}

/*__device__
bool almost_equal(float a, float b) {
    return (std::abs(a) > std::numeric_limits<float>::min()) * (std::abs(b) > std::numeric_limits<float>::min()) * ((std::abs(a - b) <= std::numeric_limits<float>::epsilon() * a) * (std::abs(a - b) <= std::numeric_limits<float>::epsilon() * b)) 
    + (1 - (std::abs(a) > std::numeric_limits<float>::min()) * (std::abs(b) > std::numeric_limits<float>::min())) * (std::abs(a - b) <= std::numeric_limits<float>::epsilon() * 2);
}*/

// From cppreference.com
template<typename Polynomial>
bool SEM::NDG_host_t<Polynomial>::almost_equal(hostFloat x, hostFloat y) {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= FLT_EPSILON * std::abs(x+y) * ulp // CHECK change this to double equivalent if using double instead of float
        // unless the result is subnormal
        || std::abs(x-y) < FLT_MIN; // CHECK change this to 64F if using double instead of float
}

// This will not work if we are on a node, or at least be pretty inefficient
// Algorithm 34
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::lagrange_interpolating_polynomials(hostFloat x, int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& lagrange_interpolant) {
    for (int i = 0; i <= N; ++i) {
        lagrange_interpolant[i] = barycentric_weights[i] / (x - nodes[i]);
    }
}

// Algorithm 34
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::normalize_lagrange_interpolating_polynomials(int N, std::vector<hostFloat>& lagrange_interpolant) {
    hostFloat sum = 0.0;
    for (int i = 0; i <= N; ++i) {
        sum += lagrange_interpolant[i];
    }
    for (int i = 0; i <= N; ++i) {
        lagrange_interpolant[i] /= sum;
    }
}

// This will not work if we are on a node, or at least be pretty inefficient
// Algorithm 36
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::lagrange_interpolating_derivative_polynomials(hostFloat x, int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& lagrange_derivative_interpolant) {
    for (int i = 0; i <= N; ++i) {
        lagrange_derivative_interpolant[i] = barycentric_weights[i] / ((x - nodes[i]) * (x - nodes[i]));
    }
}

// Algorithm 36
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::normalize_lagrange_interpolating_derivative_polynomials(hostFloat x, int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& lagrange_derivative_interpolant) {
    deviceFloat sum = 0.0;
    for (int i = 0; i <= N; ++i) {
        sum += barycentric_weights[i]/(x - nodes[i]);
    }
    for (int i = 0; i <= N; ++i) {
        lagrange_derivative_interpolant[i] /= sum;
    }
}

// Be sure to compute the diagonal afterwards
// Algorithm 37
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::polynomial_derivative_matrices(int N, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& derivative_matrices) {
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            if (i != j) { // CHECK remove for branchless, i == j will be overwritten anyway
                derivative_matrices[i * (N + 1) + j] = barycentric_weights[j] / (barycentric_weights[i] * (nodes[i] - nodes[j]));
            }
        }
    }
}

// Algorithm 37
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::polynomial_derivative_matrices_diagonal(int N, std::vector<hostFloat>& derivative_matrices) {
    for (int i = 0; i <= N; ++i) {
        derivative_matrices[i * (N + 2)] = 0.0;
        for (int j = 0; j < i; ++j) {
            derivative_matrices[i * (N + 2)] -= derivative_matrices[i * (N + 1) + j];
        }
        for (int j = i + 1; j <= N; ++j) {
            derivative_matrices[i * (N + 2)] -= derivative_matrices[i * (N + 1) + j];
        }
    }
}

template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::polynomial_derivative_matrices_hat(int N, const std::vector<hostFloat>& weights, const std::vector<hostFloat>& derivative_matrices, std::vector<hostFloat>& derivative_matrices_hat) {
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            derivative_matrices_hat[i * (N + 1) + j] = derivative_matrices[j * (N + 1) + i] * weights[j] / weights[i];
        }
    }
}

// Algorithm 57
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::polynomial_cg_derivative_matrices(int N, const std::vector<hostFloat>& weights, const std::vector<hostFloat>& derivative_matrices, std::vector<hostFloat>& g_hat_derivative_matrices) {
    for (int j = 0; j <= N; ++j) {
        for (int n = 0; n <= N; ++n) {
            hostFloat s = 0.0;
            for (int k = 0; k <= N; ++k) {
                s += derivative_matrices[k * (N + 1) + n] * derivative_matrices[k * (N + 1) + j] * weights[k];
            }
            g_hat_derivative_matrices[j * (N + 1) + n] = s/weights[j];
        }
    }
}

// Will interpolate N_interpolation_points between -1 and 1
template<typename Polynomial>
void SEM::NDG_host_t<Polynomial>::create_interpolation_matrices(int N, size_t N_interpolation_points, const std::vector<hostFloat>& nodes, const std::vector<hostFloat>& barycentric_weights, std::vector<hostFloat>& interpolation_matrices) {
    for (size_t j = 0; j < N_interpolation_points; ++j) {
        bool row_has_match = false;
        const hostFloat x_coord = 2.0 * j / (N_interpolation_points - 1) - 1.0;

        for (int k = 0; k <= N; ++k) {
            interpolation_matrices[j * (N + 1) + k] = 0.0;
            if (almost_equal(x_coord, nodes[k])) {
                interpolation_matrices[j * (N + 1) + k] = 1.0;
                row_has_match = true;
            }
        }

        if (!row_has_match) {
            hostFloat total = 0.0;
            for (int k = 0; k <= N; ++k) {
                interpolation_matrices[j * (N + 1) + k] = barycentric_weights[k] / (x_coord - nodes[k]);
                total += interpolation_matrices[j * (N + 1) + k];
            }
            for (int k = 0; k <= N; ++k) {
                interpolation_matrices[j * (N + 1) + k] /= total;
            }
        }
    }
}