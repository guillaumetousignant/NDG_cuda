#include "helpers/float_types.h"
#include <atomic>
#include <mutex>
#include <iostream>

namespace SEM { namespace Helpers {
    class ProgressBar_t {
        public: 
            ProgressBar_t();

            void set_progress(hostFloat value);

            void set_bar_width(size_t width);

            void fill_bar_progress_with(const std::string& chars);

            void fill_bar_remainder_with(const std::string& chars);

            void set_status_text(const std::string& status);

            void update(hostFloat value, std::ostream &os = std::cout);

            void write_progress(std::ostream &os = std::cout);

        private:
            std::mutex mutex_;
            hostFloat progress_;
            size_t bar_width_;
            std::string fill_;
            std::string remainder_;
            std::string status_text_;  
    };
}}
