#ifndef NDG_UTILITIES_H
#define NDG_UTILITIES_H

#include <string>

namespace SEM {
    auto is_power_of_two(int x) -> bool;

    auto to_lower(std::string& data) -> void;
}

#endif
