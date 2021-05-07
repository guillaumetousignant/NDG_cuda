#ifndef NDG_QUAD_MAP_H
#define NDG_QUAD_MAP_H

#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include <array>
#include <string>

namespace SEM {
    // Algorithm 95
    __host__ __device__
    auto quad_map(SEM::Entities::Vec2<deviceFloat> local_coordinates, const std::array<SEM::Entities::Vec2<deviceFloat>, 4>& points) -> SEM::Entities::Vec2<deviceFloat>;
}

#endif
