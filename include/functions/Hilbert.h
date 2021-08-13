#ifndef NDG_HILBERT_H
#define NDG_HILBERT_H

#include <array>

namespace SEM { namespace Hilbert {
    /**
     * @brief Describes the geometrical arrangement of a cell, one of four possible values.
     * 
     * This is necessary to the table-driven algorithm for the Hilbert curve, where the order
     * and status oh the next level cells is determined by its parent's status.
     */
    enum Status {H, A, R, B};

    //convert (x,y) to d
    auto xy2d(int n, std::array<int, 2> xy) -> int;

    //convert d to (x,y)
    auto d2xy(int n, int d) -> std::array<int, 2>;

    //rotate/flip a quadrant appropriately
    auto rot(int n, std::array<int, 2>& xy, std::array<int, 2> rxy) -> void;
}}

#endif