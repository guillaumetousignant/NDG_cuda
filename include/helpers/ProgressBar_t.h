#ifndef NDG_PROGRESSBAR_T_H
#define NDG_PROGRESSBAR_T_H

#include "helpers/float_types.h"
#include <atomic>
#include <mutex>
#include <iostream>

namespace SEM { namespace Helpers {
    class ProgressBar_t {
        public: 
            ProgressBar_t();

            auto set_progress(hostFloat value) -> void;

            auto set_bar_width(size_t width) -> void;

            auto fill_bar_progress_with(const std::string& chars) -> void;

            auto fill_bar_remainder_with(const std::string& chars) -> void;

            auto set_status_text(const std::string& status) -> void;

            auto update(hostFloat value, std::ostream &os = std::cout) -> void;

            auto write_progress(std::ostream &os = std::cout) -> void;

        private:
            std::mutex mutex_;
            hostFloat progress_;
            size_t bar_width_;
            std::string fill_;
            std::string remainder_;
            std::string status_text_;  
    };
}}

#endif
