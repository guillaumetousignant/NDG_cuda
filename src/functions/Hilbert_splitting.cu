#include "functions/Hilbert_splitting.cuh"

__host__ __device__
auto SEM::Hilbert::deduct_first_element_status(size_t outgoing_side) -> SEM::Hilbert::Status {
    constexpr std::array<SEM::Hilbert::Status, 4> first_element_status {SEM::Hilbert::Status::B, SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::R};
    return first_element_status[outgoing_side];
}

__host__ __device__
auto SEM::Hilbert::deduct_last_element_status(size_t incoming_side) -> SEM::Hilbert::Status {
    constexpr std::array<SEM::Hilbert::Status, 4> last_element_status {SEM::Hilbert::Status::A, SEM::Hilbert::Status::R, SEM::Hilbert::Status::B, SEM::Hilbert::Status::H};
    return last_element_status[incoming_side];
}

__host__ __device__
auto SEM::Hilbert::deduct_element_status(size_t incoming_side, size_t outgoing_side) -> SEM::Hilbert::Status {
    constexpr std::array<std::array<SEM::Hilbert::Status, 4>, 4> element_status {{
        {SEM::Hilbert::Status::H, SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::A},
        {SEM::Hilbert::Status::B, SEM::Hilbert::Status::B, SEM::Hilbert::Status::R, SEM::Hilbert::Status::R},
        {SEM::Hilbert::Status::B, SEM::Hilbert::Status::B, SEM::Hilbert::Status::R, SEM::Hilbert::Status::R},
        {SEM::Hilbert::Status::H, SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::A}
    }};
    return element_status[incoming_side][outgoing_side];
}

__host__ __device__
auto SEM::Hilbert::child_order(SEM::Hilbert::Status parent_status) -> std::array<size_t, 4> {
    constexpr std::array<std::array<size_t, 4>, 4> child_orders {{
        {0, 3, 2, 1},
        {0, 1, 2, 3},
        {2, 1, 0, 3},
        {2, 3, 0, 1}
    }};
    return child_orders[parent_status];
}

__host__ __device__
auto SEM::Hilbert::child_statuses(SEM::Hilbert::Status parent_status) -> std::array<SEM::Hilbert::Status, 4> {
    constexpr std::array<std::array<SEM::Hilbert::Status, 4>, 4> child_statuses {{
        {SEM::Hilbert::Status::A, SEM::Hilbert::Status::B, SEM::Hilbert::Status::H, SEM::Hilbert::Status::H},
        {SEM::Hilbert::Status::H, SEM::Hilbert::Status::A, SEM::Hilbert::Status::A, SEM::Hilbert::Status::R},
        {SEM::Hilbert::Status::R, SEM::Hilbert::Status::R, SEM::Hilbert::Status::B, SEM::Hilbert::Status::A},
        {SEM::Hilbert::Status::B, SEM::Hilbert::Status::H, SEM::Hilbert::Status::R, SEM::Hilbert::Status::B}
    }};
    return child_statuses[parent_status];
}
