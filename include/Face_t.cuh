#ifndef NDG_FACE_T_H
#define NDG_FACE_T_H

#include "float_types.h"

class Face_t {
public:
    __device__ 
    Face_t(size_t element_L, size_t element_R);

    __device__
    Face_t(const Face_t& other);

    __device__
    Face_t(Face_t&& other);	

    __device__
    Face_t& operator=(const Face_t& other);

    __device__
    Face_t& operator=(Face_t&& other);

    __host__ __device__
    Face_t();

    __host__ __device__
    ~Face_t();

    size_t elements_[2]; // left, right
    deviceFloat flux_;
    deviceFloat derivative_flux_;
};

namespace SEM {
    __global__
    void build_faces(size_t N_faces, Face_t* faces);

    __global__
    void copy_faces(size_t N_faces, const Face_t* faces, Face_t* new_faces);
}

#endif