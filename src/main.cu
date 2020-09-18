#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>

constexpr double pi = 3.14159265358979323846;

class Node_t { // Turn this into separate vectors, because cache exists
    public:
        __device__ 
        Node_t(float coordinate, int neighbour0, int neighbour1, float velocity) 
            : coordinate_(coordinate), neighbour_{neighbour0, neighbour1}, velocity_(velocity), velocity_next_(0.0f) {}

        float coordinate_;
        int neighbour_[2];
        float velocity_;
        float velocity_next_;
};

class Edge_t {
    public:
        __device__ 
        Edge_t(int node0, int node1) : nodes_{node0, node1} {}

        int nodes_[2];
};

__global__
void create_nodes(int n, Node_t* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        float coordinate = i * 1.0f/static_cast<float>(n - 1);
        float velocity = sin(i * 2.0f * pi/static_cast<float>(n - 1));
        int neighbour0 = (i > 0) ? (i - 1) : i;
        int neighbour1 = (i < n - 1) ? (i + 1) : i;
        nodes[i] = Node_t(coordinate, neighbour0, neighbour1, velocity);
    }
}

__global__
void get_velocity(int n, float* velocity, Node_t* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        velocity[i] = nodes[i].velocity_;
    }
}

__global__
void get_coordinates(int n, float* coordinates, Node_t* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        coordinates[i + 1] = nodes[i].coordinate_;
    }
}

__global__
void timestep(int n, float delta_t, Node_t* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        float u = nodes[i].velocity_;
        float u_L = nodes[nodes[i].neighbour_[0]].velocity_;
        float u_R = nodes[nodes[i].neighbour_[1]].velocity_;
        float r_L = std::abs(nodes[i].coordinate_ - nodes[nodes[i].neighbour_[0]].coordinate_);
        float r_R = std::abs(nodes[i].coordinate_ - nodes[nodes[i].neighbour_[1]].coordinate_);

        nodes[i].velocity_next_ = u * (1 - delta_t * ((u_R - u_L - ((u_R + u_L - 2 * u) * (std::pow(r_R, 2) - std::pow(r_L, 2)))/(std::pow(r_R, 2) + std::pow(r_L, 2)))/(r_R + r_L) 
                    /(1 + (r_R - r_L) * (std::pow(r_R, 2) - std::pow(r_L, 2))/((r_R + r_L) * (std::pow(r_R, 2) + std::pow(r_L, 2))))));
    }
}

__global__
void update(int n, Node_t* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        nodes[i].velocity_ = nodes[i].velocity_next_;
    }
}

void write_data(int n, float time, float* velocity, float* coordinates) {
    std::stringstream ss;
    std::ofstream file;

    ss << "data/output_t" << time << ".dat";
    file.open (ss.str());

    file << "TITLE = \"Velocity  at t= " << time << "\"" << std::endl;
    file << "VARIABLES = \"X\", \"U_x\"" << std::endl;
    file << "ZONE T= \"Zone     1\",  I= " << n << ",  J= 1,  DATAPACKING = POINT, SOLUTIONTIME = " << time << std::endl;

    for (int i = 0; i < n; ++i) {
        file << std::setw(12) << coordinates[i] << " " << std::setw(12) << velocity[i] << std::endl;
    }

    file.close();
}

__global__
void chebyshev_gauss_nodes_and_weights(int N, float* nodes, float* weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        nodes[offset + i] = -cos(pi * (2 * i + 1) / (2 * N + 2));
        weights[offset + i] = pi / (N + 1);
    }
}

__global__
void barycentric_weights(int N, const float* nodes, float* barycentric_weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int j = index; j <= N; j += stride) {
        float xjxi = 1.0f;
        for (int i = 0; i < j; ++i) {
            xjxi *= nodes[offset + j] - nodes[offset + i];
        }
        for (int i = j + 1; i <= N; ++i) {
            xjxi *= nodes[offset + j] - nodes[offset + i];
        }

        barycentric_weights[offset + j] = 1.0f/xjxi;
    }
}

/*__device__
bool almost_equal(float a, float b) {
    return (std::abs(a) > std::numeric_limits<float>::min()) * (std::abs(b) > std::numeric_limits<float>::min()) * ((std::abs(a - b) <= std::numeric_limits<float>::epsilon() * a) * (std::abs(a - b) <= std::numeric_limits<float>::epsilon() * b)) 
    + (1 - (std::abs(a) > std::numeric_limits<float>::min()) * (std::abs(b) > std::numeric_limits<float>::min())) * (std::abs(a - b) <= std::numeric_limits<float>::epsilon() * 2);
}*/

// From cppreference.com
__device__
bool almost_equal(float x, float y) {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<float>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<float>::min();
}

// This will not work if we are on a node, or at least be pretty inefficient
__global__
void lagrange_integrating_polynomials(float x, int N, const float* nodes, const float* weights, float* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        lagrange_interpolant[offset + i] = weights[offset + i] / (x - nodes[offset + i]);
    }
}

__global__
void normalize_lagrange_integrating_polynomials(int N_max, const float* lagrange_interpolant) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int N = index; N <= N_max; N += stride) {
        const int offset = N * (N + 1) /2;
        float sum = 0.0f;
        for (int i = 0; i <= N; ++i) {
            sum += lagrange_interpolant[offset + i];
        }
        for (int i = 0; i <= N; ++i) {
            lagrange_interpolant[offset + i] /= sum;
        }
    }
}

int main(void) {
    const int N_max = 16;
    const int nodes_size = (N_max + 1) * (N_max + 2)/2;
    float* nodes;
    float* weights;
    float* barycentric_weights;
    float* lagrange_interpolant_left;
    float* lagrange_interpolant_right;

    // Allocate GPU Memory – accessible from GPU
    cudaMalloc(&nodes, nodes_size * sizeof(float));
    cudaMalloc(&weights, nodes_size * sizeof(float));
    cudaMalloc(&barycentric_weights, nodes_size * sizeof(float));
    cudaMalloc(&lagrange_interpolant_left, nodes_size * sizeof(float));
    cudaMalloc(&lagrange_interpolant_right, nodes_size * sizeof(float));

    auto t_start = std::chrono::high_resolution_clock::now(); 
    const int poly_blockSize = 16; // Small number of threads per block because N will never be huge
    for (int N = 0; N <= N_max; ++N) {
        const int numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        chebyshev_gauss_nodes_and_weights<<<numBlocks, poly_blockSize>>>(N, nodes, weights);
    }

    // Nodes are needed to compute barycentric weights
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max; ++N) {
        const int numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        barycentric_weights<<<numBlocks, poly_blockSize>>>(N, nodes, barycentric_weights);
        lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(-1.0f, N, nodes, weights, lagrange_interpolant_left);
        lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(1.0f, N, nodes, weights, lagrange_interpolant_right);
    }

    // We need to divide lagrange_integrating_polynomials by sum
    cudaDeviceSynchronize();
    const int numBlocks = (N_max + poly_blockSize) / poly_blockSize;
    normalize_lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(N_max, lagrange_interpolant_left);
    normalize_lagrange_integrating_polynomials<<<numBlocks, poly_blockSize>>>(N_max, lagrange_interpolant_right);
    
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 
            << "s." << std::endl;

    // Copy vectors from device memory to host memory
    float* host_nodes = new float[nodes_size];
    float* host_weights = new float[nodes_size];
    float* host_barycentric_weights = new float[nodes_size];
    float* host_lagrange_interpolant_left = new float[nodes_size];
    float* host_lagrange_interpolant_right = new float[nodes_size];

    cudaMemcpy(host_nodes, nodes, nodes_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights, weights, nodes_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_barycentric_weights, barycentric_weights, nodes_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_left, lagrange_interpolant_left, nodes_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_lagrange_interpolant_right, lagrange_interpolant_right, nodes_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Nodes: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_nodes[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Weights: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl << "Barycentric weights: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_barycentric_weights[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants -1: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_left[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Lagrange interpolants +1: " << std::endl;
    for (int N = 0; N <= N_max; ++N) {
        const int offset = N * (N + 1) /2;

        std::cout << '\t' << "N = " << N << ": ";
        std::cout << '\t' << '\t';
        for (int i = 0; i <= N; ++i) {
            std::cout << host_lagrange_interpolant_right[offset + i] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(nodes);
    cudaFree(weights);
    cudaFree(barycentric_weights);
    cudaFree(lagrange_interpolant_left);
    cudaFree(lagrange_interpolant_right);

    delete host_nodes;
    delete host_weights;
    delete host_barycentric_weights;
    delete host_lagrange_interpolant_left;
    delete host_lagrange_interpolant_right;

    
    /**
    // OLD
    const int N = 1000;
    float delta_t = 0.00001f;
    float time = 0.0f;
    int iter_max = 20000;
    int write_interval = 2000;
    Node_t* nodes;

    // Allocate GPU Memory – accessible from GPU
    cudaMalloc(&nodes, (N)*sizeof(Node_t));

    // Run kernel on 1000 elements on the GPU, initializing nodes
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    create_nodes<<<numBlocks, blockSize>>>(N, nodes);

    float* velocity;
    cudaMallocManaged(&velocity, (N)*sizeof(float));
    float* coordinates;
    cudaMallocManaged(&coordinates, (N)*sizeof(float));
    get_coordinates<<<numBlocks, blockSize>>>(N, coordinates, nodes);    

    // Wait for GPU to finish before accessing on host
    get_velocity<<<numBlocks, blockSize>>>(N, velocity, nodes);
    cudaDeviceSynchronize();
    write_data(N, time, velocity, coordinates);

    // Calculations
    auto t_start = std::chrono::high_resolution_clock::now(); 
    for (int iter = 1; iter <= iter_max; ++iter) {
        time += delta_t;
        timestep<<<numBlocks, blockSize>>>(N, delta_t, nodes);
        update<<<numBlocks, blockSize>>>(N, nodes);

        if (!(iter % write_interval)) {
            get_velocity<<<numBlocks, blockSize>>>(N, velocity, nodes);
            cudaDeviceSynchronize();
            write_data(N, time, velocity, coordinates);
        }
    }

    get_velocity<<<numBlocks, blockSize>>>(N, velocity, nodes);
    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << iter_max << " iterations done in " 
            << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0f 
            << "s." << std::endl;
    write_data(N, time, velocity, coordinates);

    // Free memory
    cudaFree(nodes);
    cudaFree(velocity);
    cudaFree(coordinates);
    */
    
    return 0;
}