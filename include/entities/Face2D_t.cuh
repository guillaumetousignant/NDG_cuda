#ifndef NDG_FACE2D_T_H
#define NDG_FACE2D_T_H

#include "helpers/float_types.h"
#include "entities/cuda_vector.cuh"
#include <array>

namespace SEM { namespace Entities {
    class Face2D_t {
        public:
            __device__ 
            Face2D_t(int N, std::array<size_t, 2> nodes, std::array<size_t, 2> elements, std::array<size_t, 2> elements_size);

            __host__ __device__
            Face2D_t();

            int N_;
            std::array<size_t, 2> nodes_; // left, right
            std::array<size_t, 2> elements_; // left, right
            std::array<size_t, 2> elements_side_; // left, right
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 2> p_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 2> u_;
            std::array<SEM::Entities::cuda_vector<deviceFloat>, 2> v_;
            SEM::Entities::cuda_vector<deviceFloat> p_flux_;
            SEM::Entities::cuda_vector<deviceFloat> u_flux_;
            SEM::Entities::cuda_vector<deviceFloat> v_flux_;

            __device__
            auto allocate_storage() -> void;
    };
}}

#endif
