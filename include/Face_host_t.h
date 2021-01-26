#ifndef NDG_FACE_HOST_T_H
#define NDG_FACE_HOST_T_H

#include "float_types.h"
#include <array>
#include <vector>

class Face_host_t {
    public:
        Face_host_t(size_t element_L, size_t element_R);

        Face_host_t();

        ~Face_host_t();

        std::array<size_t, 2> elements_; // left, right
        hostFloat flux_;
};

#endif