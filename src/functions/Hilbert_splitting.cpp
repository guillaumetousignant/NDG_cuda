#include "functions/Hilbert_splitting.h"

auto SEM::Host::Hilbert::deduct_first_element_status(size_t outgoing_side) -> SEM::Host::Hilbert::Status {
    constexpr std::array<SEM::Host::Hilbert::Status, 4> first_element_status {SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R};
    return first_element_status[outgoing_side];
}

auto SEM::Host::Hilbert::deduct_last_element_status(size_t incoming_side) -> SEM::Host::Hilbert::Status {
    constexpr std::array<SEM::Host::Hilbert::Status, 4> last_element_status {SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H};
    return last_element_status[incoming_side];
}

auto SEM::Host::Hilbert::deduct_element_status(size_t incoming_side, size_t outgoing_side) -> SEM::Host::Hilbert::Status {
    constexpr std::array<std::array<SEM::Host::Hilbert::Status, 4>, 4> element_status {{
        {SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::A},
        {SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::R},
        {SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::R},
        {SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::A}
    }};
    return element_status[incoming_side][outgoing_side];
}

auto SEM::Host::Hilbert::child_order(SEM::Host::Hilbert::Status parent_status, int rotation) -> std::array<size_t, 4> {
    constexpr std::array<std::array<std::array<size_t, 4>, 4>, 4> child_orders {{
        {{
            {0, 3, 2, 1},
            {0, 1, 2, 3},
            {2, 1, 0, 3},
            {2, 3, 0, 1}
        }},
        {{
            {3, 2, 1, 0},
            {1, 2, 3, 0},
            {1, 0, 3, 2},
            {3, 0, 1, 2}
        }},
        {{
            {2, 1, 0, 3},
            {2, 3, 0, 1},
            {0, 3, 2, 1},
            {0, 1, 2, 3}
        }},
        {{
            {1, 0, 3, 2},
            {3, 0, 1, 2},
            {3, 2, 1, 0},
            {1, 2, 3, 0}
        }}
    }};
    return child_orders[rotation][parent_status];
}

auto SEM::Host::Hilbert::child_statuses(SEM::Host::Hilbert::Status parent_status, int rotation) -> std::array<SEM::Host::Hilbert::Status, 4> {
    constexpr std::array<std::array<std::array<SEM::Host::Hilbert::Status, 4>, 4>, 4> child_statuses_array {{
        {{
            {SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::H},
            {SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R},
            {SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::A},
            {SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::B}
        }},
        {{
            {SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A},
            {SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::H},
            {SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R},
            {SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::B}
        }},
        {{
            {SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::B},
            {SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A},
            {SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::R},
            {SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H}
        }},
        {{
            {SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H},
            {SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::A},
            {SEM::Host::Hilbert::Status::A, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::R, SEM::Host::Hilbert::Status::B},
            {SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::B, SEM::Host::Hilbert::Status::H, SEM::Host::Hilbert::Status::R}
        }}
    }};
    return child_statuses_array[rotation][parent_status];
}
