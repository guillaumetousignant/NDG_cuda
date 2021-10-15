#ifndef NDG_FUNCTIONS_HILBERT_H
#define NDG_FUNCTIONS_HILBERT_H

#include <array>

namespace SEM { namespace Hilbert {
    //convert (x,y) to d
    auto xy2d(int n, std::array<int, 2> xy) -> int;

    //convert d to (x,y)
    auto d2xy(int n, int d) -> std::array<int, 2>;

    //rotate/flip a quadrant appropriately
    auto rot(int n, std::array<int, 2>& xy, std::array<int, 2> rxy) -> void;
}}

#endif