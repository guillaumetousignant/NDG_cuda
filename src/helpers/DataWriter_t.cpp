#include "helpers/DataWriter_t.h"
#include <mpi.h>
#include <vtkNew.h>
#include <vtkPoints2D.h>
#include <vtkUnstructuredGrid.h>
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
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    int global_size;
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    // Creating points
    vtkNew<vtkPoints2D> points;
    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                points->InsertPoint(offset + i * N_interpolation_points + j, x[offset + i * N_interpolation_points + j], y[offset + i * N_interpolation_points + j]);
            }
        }
    }

    // Creating cells, currently as points (It seems like a bad idea)
    vtkNew<vtkUnstructuredGrid> grid;
    grid->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                const vtkIdType index = offset + i * N_interpolation_points + j;
                grid->InsertNextCell(VTK_VERTEX, 1, &index);
            }
        }
    }

    grid->SetPoints(points);

    vtkNew<vtkXMLPUnstructuredGridWriter> writer;
    writer->SetInputConnection(grid->GetOutputPort());
    writer->SetFileName(filename_.string().c_str());
    writer->SetNumberOfPieces(global_size);
    writer->SetStartPiece(0);
    writer->SetEndPiece(global_size - 1);
    writer->Update();
}
