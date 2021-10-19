#include "helpers/DataWriter_t.h"
#include "helpers/json.hpp"
#include <array>
#include <fstream>
#include <mpi.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkUnsignedLongLongArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
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

    series_element_filename_ = directory_ / filename_;
    series_element_filename_ += "_element" + extension_ + ".series";

    if (mpi_controller_->GetLocalProcessId() == 0) {
        create_time_series_file();
    }
}

auto SEM::Helpers::DataWriter_t::write_data(size_t N_interpolation_points, 
                                            size_t n_elements, 
                                            deviceFloat time,
                                            const std::vector<deviceFloat>& x, 
                                            const std::vector<deviceFloat>& y, 
                                            const std::vector<deviceFloat>& p, 
                                            const std::vector<deviceFloat>& u, 
                                            const std::vector<deviceFloat>& v) const -> void {

    // Creating points
    vtkNew<vtkPoints> points; // Should bt vtkPoints2D, but unstructured meshes can't take 2D points.
    points->Allocate(n_elements * N_interpolation_points * N_interpolation_points);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                points->InsertPoint(offset + i * N_interpolation_points + j, x[offset + i * N_interpolation_points + j], y[offset + i * N_interpolation_points + j], 0);
            }
        }
    }

    // Creating cells, currently as points (It seems like a bad idea)
    vtkNew<vtkUnstructuredGrid> grid;
    grid->AllocateExact(n_elements * N_interpolation_points * N_interpolation_points, 4);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
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
    pressure->Allocate(n_elements * N_interpolation_points * N_interpolation_points);
    pressure->SetName("Pressure");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
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
    velocity->Allocate(n_elements * N_interpolation_points * N_interpolation_points * 2);
    velocity->SetName("Velocity");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                velocity->InsertNextValue(u[offset + i * N_interpolation_points + j]);
                velocity->InsertNextValue(v[offset + i * N_interpolation_points + j]);
            }
        }
    }

    grid->GetPointData()->AddArray(velocity);

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

auto SEM::Helpers::DataWriter_t::write_complete_data(size_t N_interpolation_points, 
                                                     size_t n_elements, 
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
                                                     const std::vector<deviceFloat>& p_sigma, 
                                                     const std::vector<deviceFloat>& u_sigma, 
                                                     const std::vector<deviceFloat>& v_sigma, 
                                                     const std::vector<int>& refine, 
                                                     const std::vector<int>& coarsen,
                                                     const std::vector<int>& split_level,
                                                     const std::vector<deviceFloat>& p_analytical_error,
                                                     const std::vector<deviceFloat>& u_analytical_error,
                                                     const std::vector<deviceFloat>& v_analytical_error,
                                                     const std::vector<int>& status) const -> void {

    // Creating points
    vtkNew<vtkPoints> points; // Should bt vtkPoints2D, but unstructured meshes can't take 2D points.
    points->Allocate(n_elements * N_interpolation_points * N_interpolation_points);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                points->InsertPoint(offset + i * N_interpolation_points + j, x[offset + i * N_interpolation_points + j], y[offset + i * N_interpolation_points + j], 0);
            }
        }
    }

    vtkNew<vtkPoints> points_elements; // Should bt vtkPoints2D, but unstructured meshes can't take 2D points.
    points_elements->Allocate(n_elements * 4);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        points_elements->InsertPoint(4 * element_index,     x[offset], y[offset], 0);
        points_elements->InsertPoint(4 * element_index + 1, x[offset + (N_interpolation_points - 1) * N_interpolation_points], y[offset + (N_interpolation_points - 1) * N_interpolation_points], 0);
        points_elements->InsertPoint(4 * element_index + 2, x[offset + (N_interpolation_points - 1) * N_interpolation_points + (N_interpolation_points - 1)], y[offset + (N_interpolation_points - 1) * N_interpolation_points + (N_interpolation_points - 1)], 0);
        points_elements->InsertPoint(4 * element_index + 3, x[offset + (N_interpolation_points - 1)], y[offset + (N_interpolation_points - 1)], 0);
    }

    // Creating cells, currently as points (It seems like a bad idea)
    vtkNew<vtkUnstructuredGrid> grid;
    grid->AllocateExact(n_elements * N_interpolation_points * N_interpolation_points, 4);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
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

    vtkNew<vtkUnstructuredGrid> grid_elements;
    grid_elements->AllocateExact(n_elements, 4);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const std::array<vtkIdType, 4> index {static_cast<vtkIdType>(4 * element_index),
                                              static_cast<vtkIdType>(4 * element_index + 1),
                                              static_cast<vtkIdType>(4 * element_index + 2),
                                              static_cast<vtkIdType>(4 * element_index + 3)};
        grid_elements->InsertNextCell(VTK_QUAD, 4, index.data());
    }

    grid_elements->SetPoints(points_elements);

    // Add pressure to each point
    vtkNew<vtkDoubleArray> pressure;
    pressure->SetNumberOfComponents(1);
    pressure->Allocate(n_elements * N_interpolation_points * N_interpolation_points);
    pressure->SetName("Pressure");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
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
    velocity->Allocate(n_elements * N_interpolation_points * N_interpolation_points * 2);
    velocity->SetName("Velocity");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
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
    vtkNew<vtkIntArray> N_output;
    N_output->SetNumberOfComponents(1);
    N_output->Allocate(n_elements);
    N_output->SetName("N");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        N_output->InsertNextValue(N[element_index]);
    }

    grid_elements->GetCellData()->AddArray(N_output);

    // Add index to each point
    vtkNew<vtkUnsignedLongLongArray> index;
    index->SetNumberOfComponents(1);
    index->Allocate(n_elements);
    index->SetName("Index");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        index->InsertNextValue(element_index);
    }

    grid_elements->GetCellData()->AddArray(index);

    // Add status to each point
    vtkNew<vtkIntArray> status_output;
    status_output->SetNumberOfComponents(1);
    status_output->Allocate(n_elements);
    status_output->SetName("Status");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        status_output->InsertNextValue(status[element_index]);
    }

    grid_elements->GetCellData()->AddArray(status_output);

    // Add pressure derivative to each point
    vtkNew<vtkDoubleArray> pressure_derivative;
    pressure_derivative->SetNumberOfComponents(1);
    pressure_derivative->Allocate(n_elements * N_interpolation_points * N_interpolation_points);
    pressure_derivative->SetName("PressureDerivative");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
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
    velocity_derivative->Allocate(n_elements * N_interpolation_points * N_interpolation_points * 2);
    velocity_derivative->SetName("VelocityDerivative");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
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
    p_error_output->Allocate(n_elements);
    p_error_output->SetName("PressureError");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        p_error_output->InsertNextValue(p_error[element_index]);
    }

    grid_elements->GetCellData()->AddArray(p_error_output);

    // Add velocity error to each point
    vtkNew<vtkDoubleArray> velocity_error;
    velocity_error->SetNumberOfComponents(2);
    velocity_error->Allocate(n_elements);
    velocity_error->SetName("VelocityError");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        velocity_error->InsertNextValue(u_error[element_index]);
        velocity_error->InsertNextValue(v_error[element_index]);
    }

    grid_elements->GetCellData()->AddArray(velocity_error);

    // Add p_sigma to each point
    vtkNew<vtkDoubleArray> p_sigma_output;
    p_sigma_output->SetNumberOfComponents(1);
    p_sigma_output->Allocate(n_elements);
    p_sigma_output->SetName("PressureSigma");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        p_sigma_output->InsertNextValue(p_sigma[element_index]);
    }

    grid_elements->GetCellData()->AddArray(p_sigma_output);

    // Add velocity sigma to each point
    vtkNew<vtkDoubleArray> velocity_sigma;
    velocity_sigma->SetNumberOfComponents(2);
    velocity_sigma->Allocate(n_elements);
    velocity_sigma->SetName("VelocitySigma");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        velocity_sigma->InsertNextValue(u_sigma[element_index]);
        velocity_sigma->InsertNextValue(v_sigma[element_index]);
    }

    grid_elements->GetCellData()->AddArray(velocity_sigma);

    // Add analytical solution pressure error to each point
    vtkNew<vtkDoubleArray> p_analytical_error_output;
    p_analytical_error_output->SetNumberOfComponents(1);
    p_analytical_error_output->Allocate(n_elements * N_interpolation_points * N_interpolation_points);
    p_analytical_error_output->SetName("PressureAnalyticalError");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                p_analytical_error_output->InsertNextValue(p_analytical_error[offset + i * N_interpolation_points + j]);
            }
        }
    }

    grid->GetPointData()->AddArray(p_analytical_error_output);

    // Add analytical solution velocity error to each point
    vtkNew<vtkDoubleArray> velocity_analytical_error;
    velocity_analytical_error->SetNumberOfComponents(2);
    velocity_analytical_error->Allocate(n_elements * N_interpolation_points * N_interpolation_points * 2);
    velocity_analytical_error->SetName("VelocityAnalyticalError");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset = element_index * N_interpolation_points * N_interpolation_points;
        for (size_t i = 0; i < N_interpolation_points; ++i) {
            for (size_t j = 0; j < N_interpolation_points; ++j) {
                velocity_analytical_error->InsertNextValue(u_analytical_error[offset + i * N_interpolation_points + j]);
                velocity_analytical_error->InsertNextValue(v_analytical_error[offset + i * N_interpolation_points + j]);
            }
        }
    }

    grid->GetPointData()->AddArray(velocity_analytical_error);

    // Add refine to each point
    vtkNew<vtkIntArray> refine_output;
    refine_output->SetNumberOfComponents(1);
    refine_output->Allocate(n_elements);
    refine_output->SetName("Refine");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        refine_output->InsertNextValue(refine[element_index]);
    }

    grid_elements->GetCellData()->AddArray(refine_output);

    // Add coarsen to each point
    vtkNew<vtkIntArray> coarsen_output;
    coarsen_output->SetNumberOfComponents(1);
    coarsen_output->Allocate(n_elements);
    coarsen_output->SetName("Coarsen");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        coarsen_output->InsertNextValue(coarsen[element_index]);
    }

   grid_elements->GetCellData()->AddArray(coarsen_output);

    // Add split level to each point
    vtkNew<vtkIntArray> split_level_output;
    split_level_output->SetNumberOfComponents(1);
    split_level_output->Allocate(n_elements);
    split_level_output->SetName("SplitLevel");

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        split_level_output->InsertNextValue(split_level[element_index]);
    }

    grid_elements->GetCellData()->AddArray(split_level_output);

    // Filename
    std::stringstream ss;
    ss << "_t" << std::setprecision(9) << std::fixed << time << "s";
    std::string time_string = ss.str();
    std::replace(time_string.begin(), time_string.end(), '.', '_');
    std::stringstream ss2;
    ss2 << filename_ << time_string << extension_;
    const fs::path output_filename = directory_ / ss2.str();

    std::stringstream ss3;
    ss3 << filename_ << "_element" << time_string << extension_;
    const fs::path output_filename_element = directory_ / ss3.str();

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

    vtkNew<vtkXMLPUnstructuredGridWriter> writer_element;
    writer_element->SetController(mpi_controller_);
    writer_element->SetInputData(grid_elements);
    writer_element->SetFileName(output_filename_element.string().c_str());
    writer_element->SetNumberOfPieces(mpi_controller_->GetNumberOfProcesses());
    writer_element->SetStartPiece(mpi_controller_->GetLocalProcessId());
    writer_element->SetEndPiece(mpi_controller_->GetLocalProcessId());
    //writer_element->SetCompressorTypeToNone();
    writer_element->Update();

    if (mpi_controller_->GetLocalProcessId() == 0) {
        add_time_series_to_file(ss2.str(), ss3.str(), time);
    }
}

auto SEM::Helpers::DataWriter_t::create_time_series_file() const -> void {
    json j;
    j["file-series-version"] = "1.0";
    j["files"] = json::array();

    std::ofstream o(series_filename_);
    o << std::setw(2) << j << std::endl;

    std::ofstream o_element(series_element_filename_);
    o_element << std::setw(2) << j << std::endl;
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

auto SEM::Helpers::DataWriter_t::add_time_series_to_file(std::string filename, std::string filename_element, deviceFloat time) const -> void {
    std::fstream series_file(series_filename_, ios::in | ios::out);
    json j;
    series_file >> j;
    j["files"].push_back({{"name", filename}, {"time", time}});

    series_file.clear();
    series_file.seekp(std::ios_base::beg);
    series_file << std::setw(2) << j << std::endl;

    std::fstream series_element_file(series_element_filename_, ios::in | ios::out);
    json j2;
    series_element_file >> j2;
    j2["files"].push_back({{"name", filename_element}, {"time", time}});

    series_element_file.clear();
    series_element_file.seekp(std::ios_base::beg);
    series_element_file << std::setw(2) << j2 << std::endl;
}
