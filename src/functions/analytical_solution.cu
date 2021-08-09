#include "functions/analytical_solution.cuh"
#include "helpers/constants.h"
#include <cmath>

__host__ __device__
auto SEM::g(SEM::Entities::Vec2<deviceFloat> xy, deviceFloat t) -> std::array<deviceFloat, 3> {    
    const deviceFloat p = std::exp(-(SEM::Constants::k.x() * (xy.x() - SEM::Constants::xy0.x()) + SEM::Constants::k.y() * (xy.y() - SEM::Constants::xy0.y()) - SEM::Constants::c * t) * (SEM::Constants::k.x() * (xy.x() - SEM::Constants::xy0.x()) + SEM::Constants::k.y() * (xy.y() - SEM::Constants::xy0.y()) - SEM::Constants::c * t) / (SEM::Constants::d * SEM::Constants::d));

    return {p,
            p * SEM::Constants::k.x() / SEM::Constants::c,
            p * SEM::Constants::k.y() / SEM::Constants::c};
}