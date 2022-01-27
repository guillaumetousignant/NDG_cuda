#define NOMINMAX
#include "helpers/termcolor.hpp"

template<unsigned int r, unsigned int g, unsigned int b>
auto SEM::Helpers::ProgressBar_t::set_colour<r, g, b>() -> void {
    std::unique_lock lock{mutex_};
    colour_ = termcolor::color<r, g, b>; 
}