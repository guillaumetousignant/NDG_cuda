#include "functions/Hilbert.h"

auto SEM::Hilbert::deduct_first_element_status(size_t outgoing_side) -> SEM::Hilbert::Status {
    constexpr std::array<SEM::Hilbert::Status, 4> first_element_status {SEM::Hilbert::Status::B, SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::R};
    return first_element_status[outgoing_side];
}

auto SEM::Hilbert::deduct_last_element_status(size_t incoming_side) -> SEM::Hilbert::Status {
    constexpr std::array<SEM::Hilbert::Status, 4> last_element_status {SEM::Hilbert::Status::A, SEM::Hilbert::Status::R, SEM::Hilbert::Status::B, SEM::Hilbert::Status::H};
    return last_element_status[incoming_side];
}

auto SEM::Hilbert::deduct_element_status(size_t incoming_side, size_t outgoing_side) -> SEM::Hilbert::Status {
    constexpr std::array<std::array<SEM::Hilbert::Status, 4>, 4> element_status {{
        {SEM::Hilbert::Status::H, SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::A},
        {SEM::Hilbert::Status::B, SEM::Hilbert::Status::B, SEM::Hilbert::Status::R, SEM::Hilbert::Status::R},
        {SEM::Hilbert::Status::B, SEM::Hilbert::Status::B, SEM::Hilbert::Status::R, SEM::Hilbert::Status::R},
        {SEM::Hilbert::Status::H, SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::A}
    }};
    return element_status[incoming_side][outgoing_side];
}

auto SEM::Hilbert::child_order(SEM::Hilbert::Status parent_status) -> std::array<size_t, 4> {
    constexpr std::array<std::array<size_t, 4>, 4> child_orders {{
        {0, 3, 2, 1},
        {0, 1, 2, 3},
        {2, 1, 0, 3},
        {2, 3, 0, 1}
    }};
    return child_orders[parent_status];
}

auto SEM::Hilbert::child_statuses(SEM::Hilbert::Status parent_status) -> std::array<SEM::Hilbert::Status, 4> {
    constexpr std::array<std::array<SEM::Hilbert::Status, 4>, 4> child_statuses {{
        {SEM::Hilbert::Status::A, SEM::Hilbert::Status::B, SEM::Hilbert::Status::H, SEM::Hilbert::Status::H},
        {SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::A, SEM::Hilbert::Status::R},
        {SEM::Hilbert::Status::R, SEM::Hilbert::Status::R, SEM::Hilbert::Status::B, SEM::Hilbert::Status::A},
        {SEM::Hilbert::Status::B, SEM::Hilbert::Status::H, SEM::Hilbert::Status::R, SEM::Hilbert::Status::B}
    }};
    return child_statuses[parent_status];
}

//convert (x,y) to d
auto SEM::Hilbert::xy2d(int n, std::array<int, 2> xy) -> int {
    int d = 0;
    for (int s = n/2; s > 0; s /= 2) {
        const std::array<int, 2> rxy {(xy[0] & s) > 0,
                                      (xy[1] & s) > 0};

        d += s * s * ((3 * rxy[0]) ^ rxy[1]);
        rot(n, xy, rxy);
    }
    return d;
}

//convert d to (x,y)
auto SEM::Hilbert::d2xy(int n, int d) -> std::array<int, 2> {
    std::array<int, 2> xy{0, 0};
    for (int s = 1; s < n; s *= 2) {
        std::array<int, 2> rxy{1 & (d/2), 0};
        rxy[1] = 1 & (d ^ rxy[0]);
        rot(s, xy, rxy);
        xy[0] += s * rxy[0];
        xy[1] += s * rxy[1];
        d /= 4;
    }
    return xy;
}

//rotate/flip a quadrant appropriately
auto SEM::Hilbert::rot(int n, std::array<int, 2>& xy, std::array<int, 2> rxy) -> void {
    if (rxy[1] == 0) {
        if (rxy[0] == 1) {
            xy[0] = n-1 - xy[0];
            xy[1] = n-1 - xy[1];
        }

        //Swap x and y
        std::swap(xy[0], xy[1]);
    }
}