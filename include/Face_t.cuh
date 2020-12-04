#ifndef FACE_T_H
#define FACE_T_H

namespace SEM {
    __global__
    void build_faces(int N_faces, Face_t* faces);
}

class Face_t {
public:
    __device__ 
    Face_t(int element_L, int element_R) : elements_{element_L, element_R};

    __host__
    Face_t();

    __host__ __device__
    ~Face_t();

    int elements_[2]; // left, right
    float flux_;
};

#endif