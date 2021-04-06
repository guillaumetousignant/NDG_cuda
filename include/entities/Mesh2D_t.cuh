#ifndef NDG_MESH2D_T_H
#define NDG_MESH2D_T_H

#include "Element_t.cuh"
#include "Face_t.cuh"
#include "NDG_t.cuh"
#include "entities/device_vector.cuh"
#include "entities/Vec2.cuh"
#include "float_types.h"
#include <vector>
#include <limits>
#include <mpi.h>
#include <array>
#include <filesystem>

namespace SEM { namespace Entities {
    class Mesh2D_t {
        public:
            Mesh2D_t(std::filesystem::path filename, int initial_N, cudaStream_t &stream);

            constexpr static int elements_blockSize_ = 32;
            constexpr static int faces_blockSize_ = 32; // Same number of faces as elements for periodic BC
            constexpr static int boundaries_blockSize_ = 32;
            int elements_numBlocks_;
            int faces_numBlocks_;
            int boundaries_numBlocks_;
            
            size_t N_elements_global_;
            size_t N_elements_;
            size_t N_faces_;
            size_t N_local_boundaries_;
            size_t N_MPI_boundaries_;
            size_t global_element_offset_;
            size_t N_elements_per_process_;
            int initial_N_;
            deviceFloat delta_x_min_;
            int adaptivity_interval_;
            Element_t* elements_;
            Face_t* faces_;
            size_t* local_boundary_to_element_;
            size_t* MPI_boundary_to_element_;
            size_t* MPI_boundary_from_element_;

            void read_su2(std::filesystem::path filename);
            void set_initial_conditions(const deviceFloat* nodes);
            void boundary_conditions();
            void print();
            void write_data(deviceFloat time, size_t N_interpolation_points, const deviceFloat* interpolation_matrices);
            deviceFloat get_delta_t(const deviceFloat CFL);
            
            template<typename Polynomial>
            void solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const NDG_t<Polynomial> &NDG, deviceFloat viscosity);

        private:
            deviceFloat* device_delta_t_array_;
            std::vector<deviceFloat> host_delta_t_array_;
            unsigned long* device_refine_array_;
            std::vector<unsigned long> host_refine_array_;
            deviceFloat* device_boundary_phi_L_;
            std::vector<deviceFloat> host_boundary_phi_L_;
            deviceFloat* device_boundary_phi_R_;
            std::vector<deviceFloat> host_boundary_phi_R_;
            deviceFloat* device_boundary_phi_prime_L_;
            std::vector<deviceFloat> host_boundary_phi_prime_L_;
            deviceFloat* device_boundary_phi_prime_R_;
            std::vector<deviceFloat> host_boundary_phi_prime_R_;
            std::vector<size_t> host_MPI_boundary_to_element_;
            std::vector<size_t> host_MPI_boundary_from_element_;
            cudaStream_t &stream_;

            std::vector<std::array<double, 4>> send_buffers_;
            std::vector<std::array<double, 4>> receive_buffers_;
            std::vector<MPI_Request> requests_;
            std::vector<MPI_Status> statuses_;

            void write_file_data(size_t N_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& coordinates, const std::vector<deviceFloat>& velocity, const std::vector<deviceFloat>& du_dx, const std::vector<deviceFloat>& intermediate, const std::vector<deviceFloat>& x_L, const std::vector<deviceFloat>& x_R, const std::vector<int>& N, const std::vector<deviceFloat>& sigma, const bool* refine, const bool* coarsen, const std::vector<deviceFloat>& error);
            void adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights);
    };
}}

#endif
