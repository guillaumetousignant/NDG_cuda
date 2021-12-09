#ifndef NDG_HELPERS_DATAWRITER_T_H
#define NDG_HELPERS_DATAWRITER_T_H

#include "helpers/float_types.h"
#include <filesystem>
#include <string>
#include <vector>
#include <vtkNew.h>
#include <vtkMPIController.h>

namespace fs = std::filesystem;

namespace SEM { namespace Helpers {
    class DataWriter_t {
        public:
            DataWriter_t(fs::path output_filename);

            auto write_data(size_t n_interpolation_points, size_t N_elements, deviceFloat time, const std::vector<deviceFloat>& x, const std::vector<deviceFloat>& y, const std::vector<deviceFloat>& p, const std::vector<deviceFloat>& u, const std::vector<deviceFloat>& v) const -> void;

            auto write_complete_data(size_t n_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& x, const std::vector<deviceFloat>& y, const std::vector<deviceFloat>& p, const std::vector<deviceFloat>& u, const std::vector<deviceFloat>& v, const std::vector<int>& N, const std::vector<deviceFloat>& dp_dt, const std::vector<deviceFloat>& du_dt, const std::vector<deviceFloat>& dv_dt, const std::vector<deviceFloat>& p_error, const std::vector<deviceFloat>& u_error, const std::vector<deviceFloat>& v_error, const std::vector<deviceFloat>& p_sigma, const std::vector<deviceFloat>& u_sigma, const std::vector<deviceFloat>& v_sigma, const std::vector<int>& refine, const std::vector<int>& coarsen, const std::vector<int>& split_level, const std::vector<deviceFloat>& p_analytical_error, const std::vector<deviceFloat>& u_analytical_error, const std::vector<deviceFloat>& v_analytical_error, const std::vector<int>& status, const std::vector<int>& rotation) const -> void;

        private:
            fs::path directory_;
            std::string filename_;
            std::string extension_;
            fs::path series_filename_;
            vtkNew<vtkMPIController> mpi_controller_;

            auto create_time_series_file() const -> void;
            auto add_time_series_to_file(std::string filename, deviceFloat time) const -> void;
    };
}}

#endif
