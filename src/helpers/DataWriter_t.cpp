#include "helpers/DataWriter_t.h"
#include <vtkNew.h>
#include <vtkXMLPUnstructuredGridWriter.h>

namespace fs = std::filesystem;

SEM::Helpers::DataWriter_t::DataWriter_t (fs::path output_filename) : filename_(output_filename) {}

auto SEM::Helpers::DataWriter_t::write_data(size_t N_interpolation_points, 
                                            size_t N_elements, 
                                            deviceFloat time, 
                                            int rank, 
                                            const std::vector<deviceFloat>& x, 
                                            const std::vector<deviceFloat>& y, 
                                            const std::vector<deviceFloat>& p, 
                                            const std::vector<deviceFloat>& u, 
                                            const std::vector<deviceFloat>& v, 
                                            const std::vector<int>& N) const -> void {
    vtkNew<vtkXMLPUnstructuredGridWriter> writer;
}
