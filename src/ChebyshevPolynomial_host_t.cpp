#include "ChebyshevPolynomial_host_t.h"
#include <cmath>

constexpr hostFloat pi = 3.14159265358979323846;

// Algorithm 26
void SEM::ChebyshevPolynomial_host_t::nodes_and_weights(int N, std::vector<hostFloat>& nodes, std::vector<hostFloat>& weights) {
    for (int i = 0; i <= N; ++i) {
        nodes[i] = -std::cos(pi * (2 * i + 1) / (2 * N + 2));
        weights[i] = pi / (N + 1);
    }
}

hostFloat SEM::ChebyshevPolynomial_host_t::polynomial(int N, hostFloat x) {
    return std::cos(N * std::acos(x));
}
