#ifndef NDG_FACE_HOST_T_H
#define NDG_FACE_HOST_T_H

#include "helpers/float_types.h"
#include <array>
#include <vector>

namespace SEM { namespace Entities {
    class Face_host_t {
        public:
            Face_host_t(size_t element_L, size_t element_R);

            Face_host_t() = default;

            std::array<size_t, 2> elements_; // left, right
            hostFloat flux_;
            hostFloat derivative_flux_;
            hostFloat nl_flux_;
    };
}}

#endif
