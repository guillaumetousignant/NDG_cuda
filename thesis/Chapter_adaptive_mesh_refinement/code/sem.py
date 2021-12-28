import math
import numpy as np

def LegendrePolynomialAndDerivative(N, x):
    if N == 0:
        L_N = 1
        L_N_prime = 0

    elif N == 1:
        L_N = x
        L_N_prime = 1

    else:
        L_N_2 = 1
        L_N_1 = x
        L_N_2_prime = 0
        L_N_1_prime = 1

        for k in range(2, N + 1):
            L_N = (2 * k - 1) * x * L_N_1/k - (k - 1) * L_N_2/k
            L_N_prime = L_N_2_prime + (2 * k - 1) * L_N_1
            L_N_2 = L_N_1
            L_N_1 = L_N
            L_N_2_prime = L_N_1_prime
            L_N_1_prime = L_N_prime
        
    return L_N, L_N_prime

def LegendreGaussNodesAndWeights(N):
    nodes = np.zeros(N + 1)
    weights = np.zeros(N + 1)

    if N == 0:
        nodes[0] = 0
        weights[0] = 2
    elif N == 1:
        nodes[0] = -math.sqrt(1/3)
        weights[0] = 1
        nodes[1] = -nodes[0]
        weights[1] = weights[0]
    else:
        for j in range(int((N + 1)/2)):
            nodes[j] = -math.cos(math.pi * (2 * j + 1)/(2 * N + 2))

            for k in range(1000):
                L_N_plus1, L_N_plus1_prime = LegendrePolynomialAndDerivative(N + 1, nodes[j])
                delta = -L_N_plus1/L_N_plus1_prime
                nodes[j] = nodes[j] + delta
                if abs(delta) <= 0.00000001 * abs(nodes[j]):
                    break
                
            

            L_N_plus1_prime = LegendrePolynomialAndDerivative(N + 1, nodes[j])[1]
            nodes[N - j] = -nodes[j]
            weights[j] = 2/((1 - nodes[j]**2) * L_N_plus1_prime**2)
            weights[N - j] = weights[j]

    if N % 2 == 0:
        L_N_plus1_prime = LegendrePolynomialAndDerivative(N + 1, 0)[1]
        nodes[int(N/2)] = 0
        weights[int(N/2)] = 2/L_N_plus1_prime**2

    return nodes, weights
