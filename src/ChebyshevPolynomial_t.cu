#include "ChebyshevPolynomial_t.cuh"
#include <cmath>

constexpr deviceFloat pi = 3.14159265358979323846;

// Algorithm 26
__global__
void SEM::chebyshev_gauss_nodes_and_weights(int N, deviceFloat* nodes, deviceFloat* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        nodes[offset + i] = -std::cos(pi * (2 * i + 1) / (2 * N + 2));
        weights[offset + i] = pi / (N + 1);
    }
}

void SEM::ChebyshevPolynomial_t::nodes_and_weights(int N_max, int blockSize, deviceFloat* nodes, deviceFloat* weights, cudaStream_t &stream) {
    for (int N = 0; N <= N_max; ++N) {
        const int numBlocks = (N + blockSize) / blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::chebyshev_gauss_nodes_and_weights<<<numBlocks, blockSize, 0, stream>>>(N, nodes, weights);
    }
}

__device__
deviceFloat SEM::ChebyshevPolynomial_t::polynomial(int N, deviceFloat x) {
    return cos(N * acos(x));
}
