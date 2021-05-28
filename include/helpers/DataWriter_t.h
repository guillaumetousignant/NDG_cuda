#ifndef NDG_DATAWRITER_T_H
#define NDG_DATAWRITER_T_H

#include "helpers/float_types.h"
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace SEM { namespace Helpers {
    class DataWriter_t {
        public:
            DataWriter_t(fs::path output_filename);

            auto write_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& x, const std::vector<deviceFloat>& y, const std::vector<deviceFloat>& p, const std::vector<deviceFloat>& u, const std::vector<deviceFloat>& v, const std::vector<int>& N, const std::vector<deviceFloat>& dp_dt, const std::vector<deviceFloat>& du_dt, const std::vector<deviceFloat>& dv_dt) const -> void;

        private:
            fs::path filename_;
            fs::path series_filename_;

            auto create_time_series_file() const -> void;
            auto add_time_series_to_file(std::string filename, deviceFloat time) const -> void;
    };
}}

#endif
