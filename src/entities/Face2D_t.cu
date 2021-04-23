#include "entities/Face2D_t.cuh"
#include <utility>

__host__ __device__ 
SEM::Entities::Face2D_t::Face2D_t(std::array<size_t, 2> nodes, std::array<size_t, 2> elements) : 
        nodes_{nodes},
        elements_{elements},
        flux_{0.0},
        derivative_flux_{0.0},
        nl_flux_{0.0} {};

__host__ __device__
SEM::Entities::Face2D_t::Face2D_t() :
        nodes_{0, 0},
        elements_{0, 0},
        flux_{0.0},
        derivative_flux_{0.0},
        nl_flux_{0.0} {}
