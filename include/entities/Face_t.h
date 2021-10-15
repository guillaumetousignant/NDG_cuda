#ifndef NDG_ENTITIES_FACE_T_H
#define NDG_ENTITIES_FACE_T_H

#include "helpers/float_types.h"
#include <array>
#include <vector>

namespace SEM { namespace Host { namespace Entities {
    class Face_t {
        public:
            Face_t(std::size_t element_L, std::size_t element_R);

            Face_t() = default;

            std::array<std::size_t, 2> elements_; // left, right
            hostFloat flux_;
            hostFloat derivative_flux_;
            hostFloat nl_flux_;
    };
}}}

#endif
