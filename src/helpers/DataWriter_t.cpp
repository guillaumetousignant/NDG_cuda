#include "helpers/DataWriter_t.h"
#include "helpers/json.hpp"
#include <array>
#include <fstream>
#include <mpi.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPUnstructuredGridWriter.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

SEM::Helpers::DataWriter_t::DataWriter_t(fs::path output_filename) : 
        directory_(output_filename.parent_path()),
        filename_(output_filename.stem().string()),
        extension_(output_filename.extension().string()) {

    mpi_controller_->Initialize(nullptr, nullptr, 1);

    series_filename_ = output_filename;
    series_filename_ += ".series";

    if (mpi_controller_->GetLocalProcessId() == 0) {
        create_time_series_file();
    }
}

auto SEM::Helpers::DataWriter_t::write_data(size_t N_interpolation_points, 
                                            size_t N_elements, 
                                            deviceFloat time,
                                            const std::vector<deviceFloat>& x, 
                                            const std::vector<deviceFloat>& y, 
                                            const std::vector<deviceFloat>& p, 
                                            const std::vector<deviceFloat>& u, 
                                            const std::vector<deviceFloat>& v, 
                                            const std::vector<int>& N,
                                            const std::vector<deviceFloat>& dp_dt, 
                                            const std::vector<deviceFloat>& du_dt, 
                                            const std::vector<deviceFloat>& dv_dt, 
                                            const std::vector<deviceFloat>& p_error, 
                                            const std::vector<deviceFloat>& u_error, 
                                            const std::vector<deviceFloat>& v_error, 
                                            const std::vector<int>& refine, 
                                            const std::vector<int>& coarsen) const -> void {

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

    // Add p_error to each point
    vtkNew<vtkDoubleArray> p_error_output;
    p_error_output->SetNumberOfComponents(1);
    p_error_output->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    p_error_output->SetName("PressureError");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                p_error_output->InsertNextValue(p_error[element_index]);
            }
        }
    }

    grid->GetPointData()->AddArray(p_error_output);

    // Add velocity error to each point
    vtkNew<vtkDoubleArray> velocity_error;
    velocity_error->SetNumberOfComponents(2);
    velocity_error->Allocate(N_elements * N_interpolation_points * N_interpolation_points * 2);
    velocity_error->SetName("VelocityError");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                velocity_error->InsertNextValue(u_error[element_index]);
                velocity_error->InsertNextValue(v_error[element_index]);
            }
        }
    }

    grid->GetPointData()->AddArray(velocity_error);

    // Add refine to each point
    vtkNew<vtkDoubleArray> refine_output;
    refine_output->SetNumberOfComponents(1);
    refine_output->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    refine_output->SetName("Refine");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                refine_output->InsertNextValue(refine[element_index]);
            }
        }
    }

    grid->GetPointData()->AddArray(refine_output);

    // Add coarsen to each point
    vtkNew<vtkDoubleArray> coarsen_output;
    coarsen_output->SetNumberOfComponents(1);
    coarsen_output->Allocate(N_elements * N_interpolation_points * N_interpolation_points);
    coarsen_output->SetName("Coarsen");

    for (size_t element_index = 0; element_index < N_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                coarsen_output->InsertNextValue(coarsen[element_index]);
            }
        }
    }

    grid->GetPointData()->AddArray(coarsen_output);

    // Filename
    std::stringstream ss;
    ss << "_t" << std::setprecision(9) << std::fixed << time << "s";
    std::string time_string = ss.str();
    std::replace(time_string.begin(), time_string.end(), '.', '_');
    std::stringstream ss2;
    ss2 << filename_ << time_string << extension_;
    const fs::path output_filename = directory_ / ss2.str();

    // Writing to the file
    vtkNew<vtkXMLPUnstructuredGridWriter> writer;
    writer->SetController(mpi_controller_);
    writer->SetInputData(grid);
    writer->SetFileName(output_filename.string().c_str());
    writer->SetNumberOfPieces(mpi_controller_->GetNumberOfProcesses());
    writer->SetStartPiece(mpi_controller_->GetLocalProcessId());
    writer->SetEndPiece(mpi_controller_->GetLocalProcessId());
    //writer->SetCompressorTypeToNone();
    writer->Update();

    if (mpi_controller_->GetLocalProcessId() == 0) {
        add_time_series_to_file(ss2.str(), time);
    }
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
