#include "polynomials/LegendrePolynomial_t.cuh"
#include <cmath>
#include <limits>

constexpr deviceFloat pi{3.14159265358979323846};

// Algorithm 23
__global__
void SEM::Device::Polynomials::legendre_gauss_nodes_and_weights(int N, deviceFloat* nodes, deviceFloat* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const size_t offset = N * (N + 1) /2;

    for (int i = index; i < (N + 1)/2; i += stride) {
        if (N == 1) { // CHECK will enter loop above
            nodes[offset] = -std::sqrt(deviceFloat{1}/deviceFloat{3});
            weights[offset] = deviceFloat{1};
            nodes[offset + 1] = -nodes[offset];
            weights[offset + 1] = weights[offset];
        }
        else {
            nodes[offset + i] = -std::cos(pi * (2 * i + 1)/(2 * N + 2));
                    
            for (int k = 0; k < 1000; ++k) {
                deviceFloat L_N_plus1, L_N_plus1_prime;
                LegendrePolynomial_t::polynomial_and_derivative(N + 1, nodes[offset + i], L_N_plus1, L_N_plus1_prime);
                deviceFloat delta = -L_N_plus1/L_N_plus1_prime;
                nodes[offset + i] += delta;
                if (std::abs(delta) < std::numeric_limits<deviceFloat>::min() * 2 * std::abs(nodes[offset + i])) {
                    break;
                }

            }

            deviceFloat dummy, L_N_plus1_prime_final;
            LegendrePolynomial_t::polynomial_and_derivative(N + 1, nodes[offset + i], dummy, L_N_plus1_prime_final);
            nodes[offset + N - i] = -nodes[offset + i];
            weights[offset + i] = deviceFloat{2}/((1 - nodes[offset + i] * nodes[offset + i]) * L_N_plus1_prime_final * L_N_plus1_prime_final);
            weights[offset + N - i] = weights[offset + i];
        }
    }

    if (index == 0) {
        if (N == 0) {
            nodes[offset] = deviceFloat{0};
            weights[offset] = deviceFloat{2};
        }

        if (N % 2 == 0) {
            deviceFloat dummy, L_N_plus1_prime_final;
            LegendrePolynomial_t::polynomial_and_derivative(N + 1, deviceFloat{0}, dummy, L_N_plus1_prime_final);
            nodes[offset + N/2] = deviceFloat{0};
            weights[offset + N/2] = 2/(L_N_plus1_prime_final * L_N_plus1_prime_final);
        }
    }
}

void SEM::Device::Polynomials::LegendrePolynomial_t::nodes_and_weights(int N_max, int blockSize, deviceFloat* nodes, deviceFloat* weights, const cudaStream_t &stream) {
    for (int N = 0; N <= N_max; ++N) {
        const int numBlocks = ((N + 1)/2 + blockSize) / blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        SEM::Device::Polynomials::legendre_gauss_nodes_and_weights<<<numBlocks, blockSize, 0, stream>>>(N, nodes, weights);
    }
}

// Algorithm 22
__device__
void SEM::Device::Polynomials::LegendrePolynomial_t::polynomial_and_derivative(int N, deviceFloat x, deviceFloat &L_N, deviceFloat &L_N_prime) {
    if (N == 0) {
        L_N = deviceFloat{1};
        L_N_prime = deviceFloat{0};
    }
    else if (N == 1) {
        L_N = x;
        L_N_prime = deviceFloat{1};
    }
    else {
        deviceFloat L_N_2{1};
        deviceFloat L_N_1 = x;
        deviceFloat L_N_2_prime{0};
        deviceFloat L_N_1_prime{1};

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

__device__
deviceFloat SEM::Device::Polynomials::LegendrePolynomial_t::polynomial(int N, deviceFloat x) {
    if (N == 0) {
        return deviceFloat{1};
    }
    if (N == 1) {
        return x;
    }
    
    deviceFloat L_N_2{1};
    deviceFloat L_N_1 = x;
    deviceFloat L_N_2_prime{0};
    deviceFloat L_N_1_prime{1};
    deviceFloat L_N;

    for (int k = 2; k <= N; ++k) {
        L_N = (2 * k - 1) * x * L_N_1/k - (k - 1) * L_N_2/k; // L_N_1(x) ??
        const deviceFloat L_N_prime = L_N_2_prime + (2 * k - 1) * L_N_1;
        L_N_2 = L_N_1;
        L_N_1 = L_N;
        L_N_2_prime = L_N_1_prime;
        L_N_1_prime = L_N_prime;
    }
    return L_N;
}
