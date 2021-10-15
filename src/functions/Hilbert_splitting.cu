#include "functions/Hilbert_splitting.cuh"

__host__ __device__
auto SEM::Device::Hilbert::deduct_first_element_status(size_t outgoing_side) -> SEM::Device::Hilbert::Status {
    constexpr std::array<SEM::Device::Hilbert::Status, 4> first_element_status {SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R};
    return first_element_status[outgoing_side];
}

__host__ __device__
auto SEM::Device::Hilbert::deduct_last_element_status(size_t incoming_side) -> SEM::Device::Hilbert::Status {
    constexpr std::array<SEM::Device::Hilbert::Status, 4> last_element_status {SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H};
    return last_element_status[incoming_side];
}

__host__ __device__
auto SEM::Device::Hilbert::deduct_element_status(size_t incoming_side, size_t outgoing_side) -> SEM::Device::Hilbert::Status {
    constexpr std::array<std::array<SEM::Device::Hilbert::Status, 4>, 4> element_status {{
        {SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::A},
        {SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::R},
        {SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::R},
        {SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::A}
    }};
    return element_status[incoming_side][outgoing_side];
}

__host__ __device__
auto SEM::Device::Hilbert::child_order(SEM::Device::Hilbert::Status parent_status, int rotation) -> std::array<size_t, 4> {
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

__host__ __device__
auto SEM::Device::Hilbert::child_statuses(SEM::Device::Hilbert::Status parent_status, int rotation) -> std::array<SEM::Device::Hilbert::Status, 4> {
    constexpr std::array<std::array<std::array<SEM::Device::Hilbert::Status, 4>, 4>, 4> child_statuses_array {{
        {{
            {SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::H},
            {SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R},
            {SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::A},
            {SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::B}
        }},
        {{
            {SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A},
            {SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::H},
            {SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R},
            {SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::B}
        }},
        {{
            {SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::B},
            {SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A},
            {SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::R},
            {SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H}
        }},
        {{
            {SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H},
            {SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::A},
            {SEM::Device::Hilbert::Status::A, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::R, SEM::Device::Hilbert::Status::B},
            {SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::B, SEM::Device::Hilbert::Status::H, SEM::Device::Hilbert::Status::R}
        }}
    }};
    return child_statuses_array[rotation][parent_status];
}
