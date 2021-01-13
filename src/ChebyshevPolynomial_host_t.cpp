#include "ChebyshevPolynomial_host_t.h"
#include <cmath>

constexpr float pi = 3.14159265358979323846f;

// Algorithm 26
void ChebyshevPolynomial_host_t::chebyshev_gauss_nodes_and_weights(int N, float* nodes, float* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        nodes[offset + i] = -std::cos(pi * (2 * i + 1) / (2 * N + 2));
        weights[offset + i] = pi / (N + 1);
    }
}

void ChebyshevPolynomial_host_t::nodes_and_weights(int N_max, int blockSize, float* nodes, float* weights) {
    for (int N = 0; N <= N_max; ++N) {
        const int numBlocks = (N + blockSize) / blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::chebyshev_gauss_nodes_and_weights<<<numBlocks, blockSize>>>(N, nodes, weights);
    }
}