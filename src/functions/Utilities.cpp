#include "functions/Utilities.h"

#include <algorithm>

auto SEM::is_power_of_two(int x) -> bool {
    /* First x in the below expression is for the case when x is 0 */
    return x && (!(x&(x-1)));
}

auto SEM::to_lower(std::string& data) -> void {
    std::transform(data.begin(), data.end(), data.begin(),
            [](auto c){ return std::tolower(c); });
}