#ifndef NDG_FACE_T_H
#define NDG_FACE_T_H

#include "float_types.h"
#include <array>

namespace SEM {
    class Face_t {
        public:
            __device__ 
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

    __global__
    void copy_faces(size_t N_faces, const Face_t* faces, Face_t* new_faces);
}

#endif