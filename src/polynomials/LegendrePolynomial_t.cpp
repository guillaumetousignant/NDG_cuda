#include "polynomials/LegendrePolynomial_t.h"
#include <cmath>
#include <limits>

constexpr hostFloat pi{3.14159265358979323846};

// Algorithm 22
void SEM::Host::Polynomials::LegendrePolynomial_t::legendre_polynomial_and_derivative(int N, hostFloat x, hostFloat &L_N, hostFloat &L_N_prime) {
    if (N == 0) {
        L_N = hostFloat{1};
        L_N_prime = hostFloat{0};
    }
    else if (N == 1) {
        L_N = x;
        L_N_prime = hostFloat{1};
    }
    else {
        hostFloat L_N_2{1};
        hostFloat L_N_1 = x;
        hostFloat L_N_2_prime{0};
        hostFloat L_N_1_prime{1};

        for (int k = 2; k <= N; ++k) {
            L_N = (2 * k - 1) * x * L_N_1/k - (k - 1) * L_N_2/k; // L_N_1(x) ??
            L_N_prime = L_N_2_prime + (2 * k - 1) * L_N_1;
            L_N_2 = L_N_1;
            L_N_1 = L_N;
            L_N_2_prime = L_N_1_prime;
            L_N_1_prime = L_N_prime;
        }
    }
}

hostFloat SEM::Host::Polynomials::LegendrePolynomial_t::polynomial(int N, hostFloat x) {
    if (N == 0) {
        return hostFloat{1};
    }
    if (N == 1) {
        return x;
    }
    
    hostFloat L_N_2{1};
    hostFloat L_N_1 = x;
    hostFloat L_N_2_prime{0};
    hostFloat L_N_1_prime{1};
    hostFloat L_N;

    for (int k = 2; k <= N; ++k) {
        L_N = (2 * k - 1) * x * L_N_1/k - (k - 1) * L_N_2/k; // L_N_1(x) ??
        const hostFloat L_N_prime = L_N_2_prime + (2 * k - 1) * L_N_1;
        L_N_2 = L_N_1;
        L_N_1 = L_N;
        L_N_2_prime = L_N_1_prime;
        L_N_1_prime = L_N_prime;
    }
    return L_N;
}

// Algorithm 23
void SEM::Host::Polynomials::LegendrePolynomial_t::nodes_and_weights(int N, std::vector<hostFloat>& nodes, std::vector<hostFloat>& weights) {
    for (int i = 0; i < (N + 1)/2; ++i) {
        if (N == 1) { // CHECK will enter loop above
            nodes[0] = -std::sqrt(hostFloat{1}/hostFloat{3});
            weights[0] = hostFloat{1};
            nodes[1] = -nodes[0];
            weights[1] = weights[0];
        }
        else {
            nodes[i] = -std::cos(pi * (2 * i + 1)/(2 * N + 2));
                    
            for (int k = 0; k < 1000; ++k) {
                hostFloat L_N_plus1, L_N_plus1_prime;
                legendre_polynomial_and_derivative(N + 1, nodes[i], L_N_plus1, L_N_plus1_prime);
                hostFloat delta = -L_N_plus1/L_N_plus1_prime;
                nodes[i] += delta;
                if (std::abs(delta) <= std::numeric_limits<hostFloat>::min() * 2 * std::abs(nodes[i])) {
                    break;
                }

            }

            hostFloat dummy, L_N_plus1_prime_final;
            legendre_polynomial_and_derivative(N + 1, nodes[i], dummy, L_N_plus1_prime_final);
            nodes[N - i] = -nodes[i];
            weights[i] = hostFloat{2}/((1 - std::pow(nodes[i], 2)) * std::pow(L_N_plus1_prime_final, 2));
            weights[N - i] = weights[i];
        }
    }

    if (N == 0) {
        nodes[0] = hostFloat{0};
        weights[0] = hostFloat{2};
    }

    if (N % 2 == 0) {
        hostFloat dummy, L_N_plus1_prime_final;
        legendre_polynomial_and_derivative(N + 1, hostFloat{0}, dummy, L_N_plus1_prime_final);
        nodes[N/2] = hostFloat{0};
        weights[N/2] = 2/std::pow(L_N_plus1_prime_final, 2);
    }
}
