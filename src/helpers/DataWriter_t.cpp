#include "helpers/DataWriter_t.h"
#include "helpers/json.hpp"
#include <array>
#include <fstream>
#include <mpi.h>
#include <vtkNew.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPUnstructuredGridWriter.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

SEM::Helpers::DataWriter_t::DataWriter_t (fs::path output_filename) : filename_(output_filename) {
    series_filename_ = filename_;
    series_filename_+= ".series";
    create_time_series_file();
}

auto SEM::Helpers::DataWriter_t::write_data(size_t N_interpolation_points, 
                                            size_t N_elements, 
                                            deviceFloat time, 
                                            int rank, 
                                            const std::vector<deviceFloat>& x, 
                                            const std::vector<deviceFloat>& y, 
                                            const std::vector<deviceFloat>& p, 
                                            const std::vector<deviceFloat>& u, 
                                            const std::vector<deviceFloat>& v, 
                                            const std::vector<int>& N,
                                            const std::vector<deviceFloat>& dp_dt, 
                                            const std::vector<deviceFloat>& du_dt, 
                                            const std::vector<deviceFloat>& dv_dt) const -> void {
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

    // Add pressure derivative to each point
    vtkNew<vtkDoubleArray> pressure_derivative;
    pressure_derivative->SetNumberOfComponents(1);
    pressure_derivative->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    pressure_derivative->SetName("PressureDerivative");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                pressure_derivative->InsertNextValue(dp_dt[offset + i * N_interpolation_points + j]);
            }
        }
    }

    grid->GetPointData()->AddArray(pressure_derivative);

    // Add velocity derivative to each point
    vtkNew<vtkDoubleArray> velocity_derivative;
    velocity_derivative->SetNumberOfComponents(2);
    velocity_derivative->Allocate(N_elements * N_interpolation_points * N_interpolation_points * 2);
    velocity_derivative->SetName("VelocityDerivative");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                velocity_derivative->InsertNextValue(du_dt[offset + i * N_interpolation_points + j]);
                velocity_derivative->InsertNextValue(dv_dt[offset + i * N_interpolation_points + j]);
            }
        }
    }

    grid->GetPointData()->AddArray(velocity_derivative);

    // Filename
    std::stringstream ss;
    ss << "_t" << std::setprecision(9) << std::fixed << time << "s";
    std::string time_string = ss.str();
    std::replace(time_string.begin(), time_string.end(), '.', '_');
    std::stringstream ss2;
    ss2 << filename_.stem().string() << time_string << filename_.extension().string();
    const fs::path output_filename = filename_.parent_path() / ss2.str(); // It would be better to store all timesteps in the same file

    // Writing to the file
    vtkNew<vtkXMLPUnstructuredGridWriter> writer;
    writer->SetInputData(grid);
    writer->SetFileName(output_filename.string().c_str());
    writer->SetNumberOfPieces(global_size);
    writer->SetStartPiece(0);
    writer->SetEndPiece(global_size - 1);
    writer->Update();

    add_time_series_to_file(ss2.str(), time);
}

auto SEM::Helpers::DataWriter_t::create_time_series_file() const -> void {
    json j;
    j["file-series-version"] = "1.0";
    j["files"] = json::array();

    std::ofstream o(series_filename_);
    o << std::setw(2) << j << std::endl;
}

auto SEM::Helpers::DataWriter_t::add_time_series_to_file(std::string filename, deviceFloat time) const -> void {
    std::fstream series_file(series_filename_, ios::in | ios::out);
    json j;
    series_file >> j;
    j["files"].push_back({{"name", filename}, {"time", time}});

    series_file.clear();
    series_file.seekp(std::ios_base::beg);
    series_file << std::setw(2) << j << std::endl;
}
