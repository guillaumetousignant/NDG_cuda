#ifndef NDG_FACE2D_T_H
#define NDG_FACE2D_T_H

#include "helpers/float_types.h"
#include "entities/cuda_vector.cuh"
#include <array>

namespace SEM { namespace Entities {
    class Face2D_t {
        public:
            __device__ 
            Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements);

            __host__ __device__
            Face2D_t();

            int N_;
            std::array<size_t, 2> nodes_; // left, right
            std::array<size_t, 2> elements_; // left, right
            SEM::Entities::cuda_vector<deviceFloat> p_;
            SEM::Entities::cuda_vector<deviceFloat> u_;
            SEM::Entities::cuda_vector<deviceFloat> v_;

            __device__
            auto allocate_storage() -> void;
    };
}}

#endif
