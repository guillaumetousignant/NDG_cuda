#ifndef NDG_FACE2D_T_H
#define NDG_FACE2D_T_H

#include "helpers/float_types.h"
#include <array>

namespace SEM { namespace Entities {
    class Face2D_t {
        public:
            __host__ __device__ 
            Face2D_t(std::array<size_t, 2> nodes, std::array<size_t, 2> elements);

            __host__ __device__
            Face2D_t();

            std::array<size_t, 2> nodes_; // left, right
            std::array<size_t, 2> elements_; // left, right
            deviceFloat flux_;
            deviceFloat derivative_flux_;
            deviceFloat nl_flux_;
    };
}}

#endif
