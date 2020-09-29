#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>
#include <cfloat>

constexpr float pi = 3.14159265358979323846;

/*class Node_t { // Turn this into separate vectors, because cache exists
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
}*/

class NDG_t { 
public:
    __host__ __device__ 
    NDG_t(int N_max) : N_max_(N_max) {
        const int vector_length = (N_max_ + 1) * (N_max_ + 2)/2; // Flattened length of all N one after the other
        const int matrix_length = (N_max_ + 1) * (N_max_ + 2) * (2 * N_max_ + 3)/6; // Flattened length of all N² one after the other

        nodes_ = new float[vector_length];
        weights_ = new float[vector_length];
        barycentric_weights_ = new float[vector_length];
        lagrange_interpolant_left_ = new float[vector_length];
        lagrange_interpolant_right_ = new float[vector_length];
        derivative_matrices_ = new float[matrix_length];
        derivative_matrices_hat_= new float[matrix_length];
    }

    __host__ __device__
    ~NDG_t() {
        delete [] nodes_;
        delete [] weights_;
        delete [] barycentric_weights_;
        delete [] lagrange_interpolant_left_;
        delete [] lagrange_interpolant_right_;
        delete [] derivative_matrices_;
        delete [] derivative_matrices_hat_;
    }

    int N_max_;
    float* nodes_;
    float* weights_;
    float* barycentric_weights_;
    float* lagrange_interpolant_left_;
    float* lagrange_interpolant_right_;
    float* derivative_matrices_;
    float* derivative_matrices_hat_;

    __host__
    void print() const {
        std::cout << "Nodes: " << std::endl;
        for (int N = 0; N <= N_max_; ++N) {
            const int offset = N * (N + 1) /2;
    
            std::cout << '\t' << "N = " << N << ": ";
            std::cout << '\t' << '\t';
            for (int i = 0; i <= N; ++i) {
                std::cout << nodes_[offset + i] << " ";
            }
            std::cout << std::endl;
        }
    
        std::cout << std::endl << "Weights: " << std::endl;
        for (int N = 0; N <= N_max_; ++N) {
            const int offset = N * (N + 1) /2;
    
            std::cout << '\t' << "N = " << N << ": ";
            std::cout << '\t' << '\t';
            for (int i = 0; i <= N; ++i) {
                std::cout << weights_[offset + i] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl << "Barycentric weights: " << std::endl;
        for (int N = 0; N <= N_max_; ++N) {
            const int offset = N * (N + 1) /2;
    
            std::cout << '\t' << "N = " << N << ": ";
            std::cout << '\t' << '\t';
            for (int i = 0; i <= N; ++i) {
                std::cout << barycentric_weights_[offset + i] << " ";
            }
            std::cout << std::endl;
        }
    
        std::cout << std::endl << "Lagrange interpolants -1: " << std::endl;
        for (int N = 0; N <= N_max_; ++N) {
            const int offset = N * (N + 1) /2;
    
            std::cout << '\t' << "N = " << N << ": ";
            std::cout << '\t' << '\t';
            for (int i = 0; i <= N; ++i) {
                std::cout << lagrange_interpolant_left_[offset + i] << " ";
            }
            std::cout << std::endl;
        }
    
        std::cout << std::endl << "Lagrange interpolants +1: " << std::endl;
        for (int N = 0; N <= N_max_; ++N) {
            const int offset = N * (N + 1) /2;
    
            std::cout << '\t' << "N = " << N << ": ";
            std::cout << '\t' << '\t';
            for (int i = 0; i <= N; ++i) {
                std::cout << lagrange_interpolant_right_[offset + i] << " ";
            }
            std::cout << std::endl;
        }
    
        std::cout << std::endl << "Derivative matrices: " << std::endl;
        for (int N = 0; N <= N_max_; ++N) {
            const int offset_2D = N * (N + 1) * (2 * N + 1) /6;
    
            std::cout << '\t' << "N = " << N << ": " << std::endl;
            for (int i = 0; i <= N; ++i) {
                std::cout << '\t' << '\t';
                for (int j = 0; j <= N; ++j) {
                    std::cout << derivative_matrices_[offset_2D + i * (N + 1) + j] << " ";
                }
                std::cout << std::endl;
            }
        }
    
        std::cout << std::endl << "Derivative matrices hat: " << std::endl;
        for (int N = 0; N <= N_max_; ++N) {
            const int offset_2D = N * (N + 1) * (2 * N + 1) /6;
    
            std::cout << '\t' << "N = " << N << ": " << std::endl;
            for (int i = 0; i <= N; ++i) {
                std::cout << '\t' << '\t';
                for (int j = 0; j <= N; ++j) {
                    std::cout << derivative_matrices_hat_[offset_2D + i * (N + 1) + j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
};

__global__
void build_NDG(int N_max, NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < 1; i += stride) {
        NDG[i] = NDG_t(N_max);
    }
}

// Keep same layout as NDG_t to transfer from gpu to cpu
class NDG_transfer_t { 
public:
    int N_max_;
    float* nodes_;
    float* weights_;
    float* barycentric_weights_;
    float* lagrange_interpolant_left_;
    float* lagrange_interpolant_right_;
    float* derivative_matrices_;
    float* derivative_matrices_hat_;
};

class Element_t { 
public:
    __device__ 
    Element_t(int N) : N_(N) {
        phi_ = new float[N_ + 1];
        phi_prime_ = new float[N_ + 1];
    }

    __device__
    ~Element_t() {
        delete [] phi_;
        delete [] phi_prime_;
    }

    int N_;
    float phi_L_;
    float phi_R_;
    float boundary_L_;
    float boundary_R_;
    float* phi_; // Solution
    float* phi_prime_;

    // Algorithm 19
    __device__
    void matrix_vector_derivative(const NDG_t* NDG) {
        // s = 0, e = N (p.55 says N - 1)
        const int offset_1D = N_ * (N_ + 1) /2;
        const int offset_2D = N_ * (N_ + 1) * (2 * N_ + 1) /6;

        for (int i = 0; i <= N_; ++i) {
            phi_prime_[offset_1D + i] = 0.0f;
            for (int j = 0; j <= N_; ++j) {
                phi_prime_[offset_1D + i] += NDG->derivative_matrices_[offset_2D + i * (N_ + 1) + j] * phi_[offset_1D + j] * phi_[offset_1D + j]; // phi not squared in textbook, squared for Burger's
            }
        }
    }

    // Algorithm 60
    __device__
    void compute_dg_derivative(const NDG_t* NDG) {
        const int offset_1D = N_ * (N_ + 1) /2;
        
        matrix_vector_derivative(NDG);

        for (int j = 0; j <= N_; ++j) {
            phi_prime_[j] += (phi_L_ * phi_L_ * NDG->lagrange_interpolant_left_[offset_1D + j] - phi_R_ * phi_R_ * NDG->lagrange_interpolant_right_[offset_1D + j]) / (2 * NDG->weights_[offset_1D + j]);
        }
    }
    // Algorithm 61
    __device__
    float interpolate_to_boundary(const float* lagrange_interpolant) {
        const int offset_1D = N_ * (N_ + 1) /2;
        float result = 0.0f;

        for (int j = 0; j <= N_; ++j) {
            result += lagrange_interpolant[offset_1D + j] * phi_[offset_1D + j];
        }

        return result;
    }
    

    // Algorithm 61
    __device__
    void compute_dg_time_derivative(const NDG_t* NDG) {
        phi_L_ = interpolate_to_boundary(NDG->lagrange_interpolant_left_);
        phi_R_ = interpolate_to_boundary(NDG->lagrange_interpolant_right_);

        if (phi_L_ > 0.0f) {
            phi_L_ = boundary_L_;
        }
        if (phi_R_ < 0.0f) {
            phi_R_ = boundary_R_;
        }
        
        compute_dg_derivative(NDG); // Multiplied by -c in textbook, here multiplied by phi in matrix_vector_derivative
    }
};

__global__
void build_elements(int N_elements, int N, Element_t* elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        elements[i] = Element_t(N);
    }
}

__global__
void initial_conditions(int N_elements, Element_t* elements, const NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        const int offset = elements[i].N_ * (elements[i].N_ + 1) /2;
        elements[i].boundary_L_ = -sin(-1.0f);
        elements[i].boundary_R_ = -sin(1.0f);
        elements[i].phi_L_ = elements[i].boundary_L_;
        elements[i].phi_R_ = elements[i].boundary_R_;
        for (int j = 0; j <= elements[i].N_; ++j) {
            elements[i].phi_[j] = -sin(NDG->nodes_[offset + j]);
        }
    }
}

// Basically useless, find better solution when multiple elements.
__global__
void get_solution(int N_elements, const Element_t* elements, float* phi, float* phi_prime) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_elements; i += stride) {
        for (int j = 0; j <= elements[i].N_; ++j) {
            phi[j] = elements[i].phi_[j];
            phi_prime[j] = elements[i].phi_prime_[j];
        }
    }
}

// Algorithm 26
__global__
void chebyshev_gauss_nodes_and_weights(int N, NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        NDG->nodes_[offset + i] = -cos(pi * (2 * i + 1) / (2 * N + 2));
        NDG->weights_[offset + i] = pi / (N + 1);
    }
}

// Algorithm 30
__global__
void calculate_barycentric_weights(int N, NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int j = index; j <= N; j += stride) {
        float xjxi = 1.0f;
        for (int i = 0; i < j; ++i) {
            xjxi *= NDG->nodes_[offset + j] - NDG->nodes_[offset + i];
        }
        for (int i = j + 1; i <= N; ++i) {
            xjxi *= NDG->nodes_[offset + j] - NDG->nodes_[offset + i];
        }

        NDG->barycentric_weights_[offset + j] = 1.0f/xjxi;
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
    return std::abs(x-y) <= FLT_EPSILON * std::abs(x+y) * ulp // CHECK change this to double equivalent if using double instead of float
        // unless the result is subnormal
        || std::abs(x-y) < FLT_MIN; // CHECK change this to 64F if using double instead of float
}

// This will not work if we are on a node, or at least be pretty inefficient
// Algorithm 34
__global__
void lagrange_integrating_polynomials_left(float x, int N, NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        NDG->lagrange_interpolant_left_[offset + i] = NDG->weights_[offset + i] / (x - NDG->nodes_[offset + i]);
    }
}
// Has to be duplicated for left and right because the interpolant arrays are on the gpu only now since in an object
__global__
void lagrange_integrating_polynomials_right(float x, int N, NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = N * (N + 1) /2;

    for (int i = index; i <= N; i += stride) {
        NDG->lagrange_interpolant_right_[offset + i] = NDG->weights_[offset + i] / (x - NDG->nodes_[offset + i]);
    }
}

// Algorithm 34
__global__
void normalize_lagrange_integrating_polynomials_left(NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int N = index; N <= NDG->N_max_; N += stride) {
        const int offset = N * (N + 1) /2;
        float sum = 0.0f;
        for (int i = 0; i <= N; ++i) {
            sum += NDG->lagrange_interpolant_left_[offset + i];
        }
        for (int i = 0; i <= N; ++i) {
            NDG->lagrange_interpolant_left_[offset + i] /= sum;
        }
    }
}
// Has to be duplicated for left and right because the interpolant arrays are on the gpu only now since in an object
__global__
void normalize_lagrange_integrating_polynomials_right(NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int N = index; N <= NDG->N_max_; N += stride) {
        const int offset = N * (N + 1) /2;
        float sum = 0.0f;
        for (int i = 0; i <= N; ++i) {
            sum += NDG->lagrange_interpolant_right_[offset + i];
        }
        for (int i = 0; i <= N; ++i) {
            NDG->lagrange_interpolant_right_[offset + i] /= sum;
        }
    }
}

// Be sure to compute the diagonal afterwards
// Algorithm 37
__global__
void polynomial_derivative_matrices(int N, NDG_t* NDG) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const int offset_1D = N * (N + 1) /2;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            if (i != j) { // CHECK remove for branchless, i == j will be overwritten anyway
                NDG->derivative_matrices_[offset_2D + i * (N + 1) + j] = NDG->barycentric_weights_[offset_1D + j] / (NDG->barycentric_weights_[offset_1D + i] * (NDG->nodes_[offset_1D + i] - NDG->nodes_[offset_1D + j]));
            }
        }
    }
}

// Algorithm 37
__global__
void polynomial_derivative_matrices_diagonal(int N, NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index; i <= N; i += stride) {
        NDG->derivative_matrices_[offset_2D + i * (N + 2)] = 0.0f;
        for (int j = 0; j < i; ++j) {
            NDG->derivative_matrices_[offset_2D + i * (N + 2)] -= NDG->derivative_matrices_[offset_2D + i * (N + 1) + j];
        }
        for (int j = i + 1; j <= N; ++j) {
            NDG->derivative_matrices_[offset_2D + i * (N + 2)] -= NDG->derivative_matrices_[offset_2D + i * (N + 1) + j];
        }
    }
}

__global__
void polynomial_derivative_matrices_hat(int N, NDG_t* NDG) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;
    const int offset_1D = N * (N + 1) /2;
    const int offset_2D = N * (N + 1) * (2 * N + 1) /6;

    for (int i = index_x; i <= N; i += stride_x) {
        for (int j = index_y; j <= N; j += stride_y) {
            NDG->derivative_matrices_hat_[offset_2D + i * (N + 1) + j] = -NDG->derivative_matrices_[offset_2D + j * (N + 1) + i] * NDG->weights_[offset_1D + j] / NDG->weights_[offset_1D + i];
        }
    }
}

// Algorithm 61
__global__
void compute_dg_time_derivative(int N_elements, Element_t* elements, const NDG_t* NDG) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i <= N_elements; i += stride) {
        elements[i].compute_dg_time_derivative(NDG);
    }
}

int main(void) {
    const int N_elements = 1;
    const int initial_N = 8;
    const int N_max = 16;
    const int vector_length = (N_max + 1) * (N_max + 2)/2; // Flattened length of all N one after the other
    const int matrix_length = (N_max + 1) * (N_max + 2) * (2 * N_max + 3)/6; // Flattened length of all N² one after the other

    Element_t* elements;
    NDG_t* NDG;

    // Allocate GPU Memory – accessible from GPU
    cudaMalloc(&elements, N_elements * sizeof(Element_t));
    cudaMalloc(&NDG, 1 * sizeof(NDG_t));

    auto t_start = std::chrono::high_resolution_clock::now(); 
    build_NDG<<<1, 1>>>(N_max, NDG);

    const int poly_blockSize = 16; // Small number of threads per block because N will never be huge
    for (int N = 0; N <= N_max; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        chebyshev_gauss_nodes_and_weights<<<vector_numBlocks, poly_blockSize>>>(N, NDG);
    }

    const int elements_blockSize = 32; // For when we'll have multiple elements
    const int elements_numBlocks = (N_elements + elements_blockSize - 1) / elements_blockSize;
    build_elements<<<elements_numBlocks, elements_blockSize>>>(N_elements, initial_N, elements);

    // Nodes are needed to compute barycentric weights
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        calculate_barycentric_weights<<<vector_numBlocks, poly_blockSize>>>(N, NDG);
        lagrange_integrating_polynomials_left<<<vector_numBlocks, poly_blockSize>>>(-1.0f, N, NDG);
        lagrange_integrating_polynomials_right<<<vector_numBlocks, poly_blockSize>>>(1.0f, N, NDG);
    }

    initial_conditions<<<elements_numBlocks, elements_blockSize>>>(N_elements, elements, NDG);

    // We need to divide lagrange_integrating_polynomials by sum, and barycentric weights for derivative matrix
    cudaDeviceSynchronize();
    const int numBlocks = (N_max + poly_blockSize) / poly_blockSize;
    normalize_lagrange_integrating_polynomials_left<<<numBlocks, poly_blockSize>>>(NDG);
    normalize_lagrange_integrating_polynomials_right<<<numBlocks, poly_blockSize>>>(NDG);
    const dim3 matrix_blockSize(16, 16); // Small number of threads per block because N will never be huge
    for (int N = 0; N <= N_max; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices<<<matrix_numBlocks, matrix_blockSize>>>(N, NDG);
    }

    // Then we calculate the derivative matrix diagonal
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max; ++N) {
        const int vector_numBlocks = (N + poly_blockSize) / poly_blockSize; // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices_diagonal<<<vector_numBlocks, poly_blockSize>>>(N, NDG);
    }

    // All the derivative matrix has to be computed before D^
    cudaDeviceSynchronize();
    for (int N = 0; N <= N_max; ++N) {
        const dim3 matrix_numBlocks((N +  matrix_blockSize.x) / matrix_blockSize.x, (N +  matrix_blockSize.y) / matrix_blockSize.y); // Should be (N + poly_blockSize - 1) if N is not inclusive
        polynomial_derivative_matrices_hat<<<matrix_numBlocks, matrix_blockSize>>>(N, NDG);
    }

    // Starting actual computation
    cudaDeviceSynchronize();
    compute_dg_time_derivative<<<elements_numBlocks, elements_blockSize>>>(N_elements, elements, NDG);
    
    // Wait for GPU to finish before copying to host
    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU computation time: " 
            << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 
            << "s." << std::endl;

    // Copy vectors from device memory to host memory
    NDG_transfer_t NDG_transfer;
    cudaMemcpy(&NDG_transfer, NDG, 1 * sizeof(NDG_t), cudaMemcpyDeviceToHost);

    NDG_t host_NDG(N_max);

    cudaMemcpy(host_NDG.nodes_, NDG_transfer.nodes_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_NDG.weights_, NDG_transfer.weights_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_NDG.barycentric_weights_, NDG_transfer.barycentric_weights_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_NDG.lagrange_interpolant_left_, NDG_transfer.lagrange_interpolant_left_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_NDG.lagrange_interpolant_right_, NDG_transfer.lagrange_interpolant_right_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_NDG.derivative_matrices_, NDG_transfer.derivative_matrices_, matrix_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_NDG.derivative_matrices_hat_, NDG_transfer.derivative_matrices_hat_, matrix_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Can't do that!
    //cudaMemcpy(host_phi, elements[0].phi_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_phi_prime, elements[0].phi_prime_, vector_length * sizeof(float), cudaMemcpyDeviceToHost);

    // CHECK find better solution for multiple elements
    float* phi;
    float* phi_prime;
    float* host_phi = new float[initial_N + 1];
    float* host_phi_prime = new float[initial_N + 1];
    cudaMalloc(&phi, (initial_N + 1) * sizeof(float));
    cudaMalloc(&phi_prime, (initial_N + 1) * sizeof(float));
    get_solution<<<elements_numBlocks, elements_blockSize>>>(N_elements, elements, phi, phi_prime);
    cudaDeviceSynchronize();
    cudaMemcpy(host_phi, phi, (initial_N + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_phi_prime, phi_prime, (initial_N + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Printing
    host_NDG.print();

    std::cout << std::endl << "Phi: " << std::endl;
    for (int i = 0; i < N_elements; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N; ++j) {
            std::cout << host_phi[j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Phi prime: " << std::endl;
    for (int i = 0; i < N_elements; ++i) {
        std::cout << '\t' << "Element " << i << ": ";
        std::cout << '\t' << '\t';
        for (int j = 0; j <= initial_N; ++j) {
            std::cout << host_phi_prime[j] << " ";
        }
        std::cout << std::endl;
    }


    
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