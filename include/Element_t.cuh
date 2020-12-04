#ifndef ELEMENT_T_H
#define ELEMENT_T_H

class Element_t { // Turn this into separate vectors, because cache exists
public:
    __device__ 
    Element_t(int N, int neighbour_L, int neighbour_R, int face_L, int face_R, float x_L, float x_R);

    __host__ 
    Element_t();

    __host__ __device__
    ~Element_t();

    int N_;
    int neighbours_[2]; // Could also be pointers
    int faces_[2]; // Could also be pointers. left, right
    float x_[2];
    float delta_x_;
    float phi_L_;
    float phi_R_;
    float* phi_; // Solution
    float* phi_prime_;
    float* intermediate_;
};

#endif