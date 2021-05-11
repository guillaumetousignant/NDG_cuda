#include "entities/Face2D_t.cuh"
#include <utility>

__device__ 
SEM::Entities::Face2D_t::Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements) : 
        N_{N},
        nodes_{nodes},
        elements_{elements},
        p_{N + 1},
        u_{N + 1},
        v_{N + 1} {}

__host__ __device__
SEM::Entities::Face2D_t::Face2D_t() :
        N_{0},
        nodes_{0, 0},
        elements_{0, 0} {}

__device__
auto SEM::Entities::Face2D_t::allocate_storage() -> void {
        p_ = cuda_vector<deviceFloat>(N_ + 1);
        u_ = cuda_vector<deviceFloat>(N_ + 1);
        v_ = cuda_vector<deviceFloat>(N_ + 1);
}
