#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <cmath>
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
void chebyshev_gauss_nodes_and_weights(int N, float* all_nodes, float* all_weights) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        all_nodes[offset + i] = -cos(pi * (2 * i + 1) / (2 * N + 2));
        all_weights[offset + i] = pi / (N + 1);
    }
}

int main(void) {
    const int N_max = 16;
    const int nodes_size = (N_max + 1) * (N_max + 2)/2;
    float* all_nodes;
    float* all_weights;
    float** nodes;
    float** weights;

    // Allocate GPU Memory – accessible from GPU
    cudaMalloc(&all_nodes, nodes_size * sizeof(float));
    cudaMalloc(&all_weights, nodes_size * sizeof(float));
    cudaMalloc(&nodes, N_max * sizeof(float*));
    cudaMalloc(&weights, N_max * sizeof(float*));

    int poly_blockSize = 16; // Small number of threads per block because N will never be huge
    for (int N = 0; N <= N_max; ++N) {
        int numBlocks = (N + poly_blockSize - 1) / poly_blockSize;
        chebyshev_gauss_nodes_and_weights<<<numBlocks, poly_blockSize>>>(N, all_nodes, all_weights);
    }
    
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();
    // Copy vectors from device memory to host memory
    float* host_nodes;
    float* host_weights;
    cudaMemcpy(host_nodes, all_nodes, nodes_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_weights, all_weights, nodes_size, cudaMemcpyDeviceToHost);


    // Free memory
    cudaFree(all_nodes);
    cudaFree(all_weights);
    cudaFree(nodes);
    cudaFree(weights);

    delete host_nodes;
    delete host_weights;

    
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