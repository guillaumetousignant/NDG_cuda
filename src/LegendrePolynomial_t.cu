#include "LegendrePolynomial_t.cuh"
#include <cmath>

constexpr deviceFloat pi = 3.14159265358979323846;

// Algorithm 22
__device__
void SEM::legendre_polynomial_and_derivative(int N, deviceFloat x, deviceFloat &L_N, deviceFloat &L_N_prime) {
    if (N == 0) {
        L_N = 1.0f;
        L_N_prime = 0.0f;
    }
    else if (N == 1) {
        L_N = x;
        L_N_prime = 1.0f;
    }
    else {
        deviceFloat L_N_2 = 1.0f;
        deviceFloat L_N_1 = x;
        deviceFloat L_N_2_prime = 0.0f;
        deviceFloat L_N_1_prime = 1.0f;

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

// Algorithm 23
__global__
void SEM::legendre_gauss_nodes_and_weights(int N, deviceFloat* nodes, deviceFloat* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int i = index; i < (N + 1)/2; i += stride) {
        if (N == 1) { // CHECK will enter loop above
            nodes[offset] = -std::sqrt(1.0f/3.0f);
            weights[offset] = 1.0f;
            nodes[offset + 1] = -nodes[offset];
            weights[offset + 1] = weights[offset];
        }
        else {
            nodes[offset + i] = -std::cos(pi * (2 * i + 1)/(2 * N + 2));
                    
            for (int k = 0; k < 1000; ++k) {
                deviceFloat L_N_plus1, L_N_plus1_prime;
                SEM::legendre_polynomial_and_derivative(N + 1, nodes[offset + i], L_N_plus1, L_N_plus1_prime);
                deviceFloat delta = -L_N_plus1/L_N_plus1_prime;
                nodes[offset + i] += delta;
                if (std::abs(delta) <= 0.00000001f * std::abs(nodes[offset + i])) {
                    break;
                }

            }

            deviceFloat dummy, L_N_plus1_prime_final;
            SEM::legendre_polynomial_and_derivative(N + 1, nodes[offset + i], dummy, L_N_plus1_prime_final);
            nodes[offset + N - i] = -nodes[offset + i];
            weights[offset + i] = 2.0f/((1.0f - std::pow(nodes[offset + i], 2)) * std::pow(L_N_plus1_prime_final, 2));
            weights[offset + N - i] = weights[offset + i];
        }
    }

    if (index == 0) {
        if (N == 0) {
            nodes[offset] = 0.0f;
            weights[offset] = 2.0f;
        }

        if (N % 2 == 0) {
            deviceFloat dummy, L_N_plus1_prime_final;
            SEM::legendre_polynomial_and_derivative(N + 1, 0.0f, dummy, L_N_plus1_prime_final);
            nodes[offset + N/2] = 0.0f;
            weights[offset + N/2] = 2/std::pow(L_N_plus1_prime_final, 2);
        }
    }
}

void LegendrePolynomial_t::nodes_and_weights(int N_max, int blockSize, deviceFloat* nodes, deviceFloat* weights) {
    for (int N = 0; N <= N_max; ++N) {
        const int numBlocks = ((N + 1)/2 + blockSize) / blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::legendre_gauss_nodes_and_weights<<<numBlocks, blockSize>>>(N, nodes, weights);
    }
}