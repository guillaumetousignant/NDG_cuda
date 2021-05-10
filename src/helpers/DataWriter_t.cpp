#include "helpers/DataWriter_t.h"
#include <array>
#include <mpi.h>
#include <vtkNew.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
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
    vtkNew<vtkPoints> points; // Should bt vtkPoints2D, but unstructured meshes can't take 2D points.
    points->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                points->InsertPoint(offset + i * N_interpolation_points + j, x[offset + i * N_interpolation_points + j], y[offset + i * N_interpolation_points + j], 0);
            }
        }
    }

    // Creating cells, currently as points (It seems like a bad idea)
    vtkNew<vtkUnstructuredGrid> grid;
    grid->AllocateExact(N_elements * N_interpolation_points * N_interpolation_points, 4);
    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points - 1; ++i) {
            for (size_t j = 0; j < N_interpolation_points - 1; ++j) {
                const std::array<vtkIdType, 4> index {static_cast<vtkIdType>(offset + (i + 1) * N_interpolation_points + j),
                                                      static_cast<vtkIdType>(offset + (i + 1) * N_interpolation_points + j + 1),
                                                      static_cast<vtkIdType>(offset + i * N_interpolation_points + j + 1),
                                                      static_cast<vtkIdType>(offset + i * N_interpolation_points + j)};
                grid->InsertNextCell(VTK_QUAD, 4, index.data());
            }
        }
    }

    grid->SetPoints(points);

    // Add pressure to each point
    vtkNew<vtkDoubleArray> pressure;
    pressure->SetNumberOfComponents(1);
    pressure->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    pressure->SetName("Pressure");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                pressure->InsertNextValue(p[offset + i * N_interpolation_points + j]);
            }
        }
    }

    grid->GetPointData()->AddArray(pressure);

    // Add velocity to each point
    vtkNew<vtkDoubleArray> velocity;
    velocity->SetNumberOfComponents(2);
    velocity->Allocate(N_elements * N_interpolation_points * N_interpolation_points * 2);
    velocity->SetName("Velocity");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                velocity->InsertNextValue(u[offset + i * N_interpolation_points + j]);
                velocity->InsertNextValue(v[offset + i * N_interpolation_points + j]);
            }
        }
    }

    grid->GetPointData()->AddArray(velocity);

    // Add N to each point
    vtkNew<vtkDoubleArray> N_output;
    N_output->SetNumberOfComponents(1);
    N_output->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    N_output->SetName("N");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                N_output->InsertNextValue(N[element_index]);
            }
        }
    }

    grid->GetPointData()->AddArray(N_output);

    // Add index to each point
    vtkNew<vtkDoubleArray> index;
    index->SetNumberOfComponents(1);
    index->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    index->SetName("index");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                index->InsertNextValue(element_index);
            }
        }
    }

    grid->GetPointData()->AddArray(index);

    std::stringstream ss;
    ss << "_t" << std::setprecision(9) << std::fixed << time << "s";
    std::string time_string = ss.str();
    std::replace(time_string.begin(), time_string.end(), '.', '_');
    std::stringstream ss2;
    ss2 << filename_.stem().string() << time_string << filename_.extension().string();
    const fs::path output_filename = filename_.parent_path() / ss2.str(); // It would be better to store all timesteps in the same file

    vtkNew<vtkXMLPUnstructuredGridWriter> writer;
    writer->SetInputData(grid);
    writer->SetFileName(output_filename.string().c_str());
    writer->SetNumberOfPieces(global_size);
    writer->SetStartPiece(0);
    writer->SetEndPiece(global_size - 1);
    writer->Update();
}
