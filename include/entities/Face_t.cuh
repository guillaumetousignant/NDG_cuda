#ifndef NDG_FACE_T_H
#define NDG_FACE_T_H

#include "helpers/float_types.h"
#include <array>

namespace SEM { namespace Entities {
    class Face_t {
        public:
            __host__ __device__ 
            Face_t(size_t element_L, size_t element_R);

            __host__ __device__
            Face_t();

            std::array<size_t, 2> elements_; // left, right
            deviceFloat flux_;
            deviceFloat derivative_flux_;
            deviceFloat nl_flux_;
    };

    __global__
    void build_faces(size_t N_faces, Face_t* faces);
}}

#endif
