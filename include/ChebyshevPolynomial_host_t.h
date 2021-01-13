#ifndef NDG_CHEBYSHEVPOLYNOMIAL_HOST_T_H
#define NDG_CHEBYSHEVPOLYNOMIAL_HOST_T_H

class ChebyshevPolynomial_t { 
    public: 
        static void nodes_and_weights(int N_max, int blockSize, float* nodes, float* weights);

    private:
        // Algorithm 26
        static void chebyshev_gauss_nodes_and_weights(int N, float* nodes, float* weights);
};

#endif