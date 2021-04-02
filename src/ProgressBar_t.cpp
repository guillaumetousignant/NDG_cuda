#include "ProgressBar_t.h"
#include "termcolor.hpp"
#include <algorithm>

SEM::ProgressBar_t::ProgressBar_t() : 
        progress_{0.0},
        bar_width_{60},
        fill_{"#"},
        remainder_{" "},
        status_text_{""} {}

void SEM::ProgressBar_t::set_progress(hostFloat value) {
    std::unique_lock lock{mutex_};
    progress_ = value;
}

void SEM::ProgressBar_t::set_bar_width(size_t width) {
    std::unique_lock lock{mutex_};
    bar_width_ = width;    
}

void SEM::ProgressBar_t::fill_bar_progress_with(const std::string& chars) {
    std::unique_lock lock{mutex_};
    fill_ = chars;    
}

void SEM::ProgressBar_t::fill_bar_remainder_with(const std::string& chars) {
    std::unique_lock lock{mutex_};
    remainder_ = chars;    
}

void SEM::ProgressBar_t::set_status_text(const std::string& status) {
    std::unique_lock lock{mutex_};
    status_text_ = status;    
}

void SEM::ProgressBar_t::update(hostFloat value, std::ostream &os /* = std::cout */) {
    set_progress(value);
    write_progress(os);
}

void SEM::ProgressBar_t::write_progress(std::ostream &os /* = std::cout */) {
    std::unique_lock lock{mutex_};

    // No need to write once progress is 100%
    if (progress_ > 100.0) return;

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
    os << " " << std::min(static_cast<size_t>(progress_ * 100), size_t(100)) << "%"; 

    // Write status text
    os << " " << status_text_;

    // Reset the color
    os << termcolor::reset;
}