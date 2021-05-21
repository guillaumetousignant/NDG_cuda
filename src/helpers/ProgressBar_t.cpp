#include "helpers/ProgressBar_t.h"
#include "helpers/termcolor.hpp"
#include <algorithm>

SEM::Helpers::ProgressBar_t::ProgressBar_t() : 
        progress_{0.0},
        bar_width_{60},
        fill_{"#"},
        remainder_{" "},
        status_text_{""},
        status_width_{0} {}

auto SEM::Helpers::ProgressBar_t::set_progress(hostFloat value) -> void {
    std::unique_lock lock{mutex_};
    progress_ = value;
}

auto SEM::Helpers::ProgressBar_t::set_bar_width(size_t width) -> void {
    std::unique_lock lock{mutex_};
    bar_width_ = width;    
}

auto SEM::Helpers::ProgressBar_t::fill_bar_progress_with(const std::string& chars) -> void {
    std::unique_lock lock{mutex_};
    fill_ = chars;    
}

auto SEM::Helpers::ProgressBar_t::fill_bar_remainder_with(const std::string& chars) -> void {
    std::unique_lock lock{mutex_};
    remainder_ = chars;    
}

auto SEM::Helpers::ProgressBar_t::set_status_text(const std::string& status) -> void {
    std::unique_lock lock{mutex_};
    status_text_ = status;    
}

auto SEM::Helpers::ProgressBar_t::update(hostFloat value, std::ostream &os /* = std::cout */) -> void {
    set_progress(value);
    write_progress(os);
}

auto SEM::Helpers::ProgressBar_t::write_progress(std::ostream &os /* = std::cout */) -> void {
    std::unique_lock lock{mutex_};

    // Cap progress to 100%
    if (progress_ > 1.0) progress_ = 1.0;

    // Print in bold yellow
    os << termcolor::bold << termcolor::yellow;

    // Move cursor to the first position on the same line and flush 
    os << "\r" << std::flush;

    // Start bar
    os << "[";

    const auto completed = static_cast<size_t>(progress_ * static_cast<hostFloat>(bar_width_));
    for (size_t i = 0; i < bar_width_; ++i) {
        if (i <= completed) 
            os << fill_;
        else 
            os << remainder_;
    }

    // End bar
    os << "]";

    // Write progress percentage
    os << " " << min(static_cast<size_t>(progress_ * 100), size_t(100)) << "%"; 

    // Write status text
    os << " " << status_text_;
    if (status_width_ > status_text_.size()) {
        for (size_t i = 0; i < status_width_ - status_text_.size(); ++i) {
            os << " ";
        }
    }
    status_width_ = status_text_.size();

    // Reset the color
    os << termcolor::reset;
}
