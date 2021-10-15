#include "helpers/constants.h"
#include <cmath>

template <class T>
auto SEM::Host::g(SEM::Host::Entities::Vec2<T> xy, T t) -> std::array<T, 3> {    
    const T p = std::exp(-(SEM::Host::Constants::k.x() * (xy.x() - SEM::Host::Constants::xy0.x()) + SEM::Host::Constants::k.y() * (xy.y() - SEM::Host::Constants::xy0.y()) - SEM::Host::Constants::c * t) * (SEM::Host::Constants::k.x() * (xy.x() - SEM::Host::Constants::xy0.x()) + SEM::Host::Constants::k.y() * (xy.y() - SEM::Host::Constants::xy0.y()) - SEM::Host::Constants::c * t) / (SEM::Host::Constants::d * SEM::Host::Constants::d));

    return {p,
            p * SEM::Host::Constants::k.x() / SEM::Host::Constants::c,
            p * SEM::Host::Constants::k.y() / SEM::Host::Constants::c};
}