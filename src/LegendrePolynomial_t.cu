#include "LegendrePolynomial_t.cuh"
#include <cmath>

constexpr float pi = 3.14159265358979323846f;

// Algorithm 22
__device__
void SEM::legendre_polynomial_and_derivative(int N, float x, float &L_N, float &L_N_prime) {
    if (N == 0) {
        L_N = 1.0f;
        L_N_prime = 0.0f;
    }
    else if (N == 1) {
        L_N = x;
        L_N_prime = 1.0f;
    }
    else {
        float L_N_2 = 1.0f;
        float L_N_1 = x;
        float L_N_2_prime = 0.0f;
        float L_N_1_prime = 1.0f;

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
void SEM::legendre_gauss_nodes_and_weights(int N, float* nodes, float* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = N * (N + 1) /2;

    if (index == 0) {
        if (N == 0) {
            nodes[offset] = 0.0f;
            weights[offset] = 2.0f;
        }
        else if (N == 1) {
            nodes[offset] = -std::sqrt(1.0f/3.0f);
            weights[offset] = 1.0f;
            nodes[offset + 1] = -nodes[offset];
            weights[offset + 1] = weights[offset];
        }
        else {
            for (int j = 0; j < (N + 1)/2; ++j) {
                nodes[offset + j] = -std::cos(pi * (2 * j + 1)/(2 * N + 2));
                
                for (int k = 0; k < 1000; ++k) {
                    float L_N_plus1, L_N_plus1_prime;
                    SEM::legendre_polynomial_and_derivative(N + 1, nodes[offset + j], L_N_plus1, L_N_plus1_prime);
                    float delta = -L_N_plus1/L_N_plus1_prime;
                    nodes[offset + j] += delta;
                    if (std::abs(delta) <= 0.00000001f * std::abs(nodes[offset + j])) {
                        break;
                    }

                }

                float dummy, L_N_plus1_prime_final;
                SEM::legendre_polynomial_and_derivative(N + 1, nodes[offset + j], dummy, L_N_plus1_prime_final);
                nodes[offset + N - j] = -nodes[offset + j];
                weights[offset + j] = 2.0f/((1.0f - std::pow(nodes[offset + j], 2)) * std::pow(L_N_plus1_prime_final, 2));
                weights[offset + N - j] = weights[offset + j];
            }
        }

        if (N % 2 == 0) {
            float dummy, L_N_plus1_prime_final;
            SEM::legendre_polynomial_and_derivative(N + 1, 0.0f, dummy, L_N_plus1_prime_final);
            nodes[offset + N/2] = 0.0f;
            weights[offset + N/2] = 2/std::pow(L_N_plus1_prime_final, 2);
        }
    }
}

void LegendrePolynomial_t::nodes_and_weights(int N_max, int blockSize, float* nodes, float* weights) {
    for (int N = 0; N <= N_max; ++N) {
        const int numBlocks = (N + blockSize) / blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::legendre_gauss_nodes_and_weights<<<numBlocks, blockSize>>>(N, nodes, weights);
    }
}