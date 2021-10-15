#ifndef NDG_MESHES_MESH_T_CUH
#define NDG_MESHES_MESH_T_CUH

#include "entities/Element_t.cuh"
#include "entities/Face_t.cuh"
#include "entities/NDG_t.cuh"
#include "helpers/float_types.h"
#include <vector>
#include <limits>
#include <mpi.h>
#include <array>

namespace SEM { namespace Device { namespace Meshes {
    class Mesh_t {
        public:
            Mesh_t(size_t N_elements, int initial_N, deviceFloat delta_x_min, deviceFloat x_min, deviceFloat x_max, int adaptivity_interval, cudaStream_t &stream);
            ~Mesh_t();

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
            SEM::Device::Entities::Element_t* elements_;
            SEM::Device::Entities::Face_t* faces_;
            size_t* local_boundary_to_element_;
            size_t* MPI_boundary_to_element_;
            size_t* MPI_boundary_from_element_;

            void set_initial_conditions(const deviceFloat* nodes);
            void boundary_conditions();
            void print();
            void write_data(deviceFloat time, size_t n_interpolation_points, const deviceFloat* interpolation_matrices);
            deviceFloat get_delta_t(const deviceFloat CFL);
            
            template<typename Polynomial>
            void solve(const deviceFloat CFL, const std::vector<deviceFloat> output_times, const SEM::Device::Entities::NDG_t<Polynomial> &NDG, deviceFloat viscosity);

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

            void write_file_data(size_t n_interpolation_points, size_t N_elements, deviceFloat time, int rank, const std::vector<deviceFloat>& coordinates, const std::vector<deviceFloat>& velocity, const std::vector<deviceFloat>& du_dx, const std::vector<deviceFloat>& intermediate, const std::vector<deviceFloat>& x_L, const std::vector<deviceFloat>& x_R, const std::vector<int>& N, const std::vector<deviceFloat>& sigma, const bool* refine, const bool* coarsen, const std::vector<deviceFloat>& error, const std::vector<deviceFloat>& delta_x);
            void adapt(int N_max, const deviceFloat* nodes, const deviceFloat* barycentric_weights);
    };

    __global__
    void rk3_first_step(size_t N_elements, SEM::Device::Entities::Element_t* elements, deviceFloat delta_t, deviceFloat g);

    __global__
    void rk3_step(size_t N_elements, SEM::Device::Entities::Element_t* elements, deviceFloat delta_t, deviceFloat a, deviceFloat g);

    __global__
    void calculate_fluxes(size_t N_faces, SEM::Device::Entities::Face_t* faces, const SEM::Device::Entities::Element_t* elements);

    __global__
    void calculate_q_fluxes(size_t N_faces, SEM::Device::Entities::Face_t* faces, const SEM::Device::Entities::Element_t* elements);

    __device__
    void matrix_vector_multiply(int N, const deviceFloat* matrix, const deviceFloat* vector, deviceFloat* result);

    // Algorithm 19
    __device__
    void matrix_vector_derivative(int N, const deviceFloat* derivative_matrices_hat, const deviceFloat* phi, deviceFloat* phi_prime);

    // Algorithm 60 (not really anymore)
    __global__
    void compute_dg_derivative(size_t N_elements, SEM::Device::Entities::Element_t* elements, const SEM::Device::Entities::Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    // Algorithm 60 (not really anymore)
    __global__
    void compute_dg_derivative2(deviceFloat viscosity, size_t N_elements, SEM::Device::Entities::Element_t* elements, const SEM::Device::Entities::Face_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    void warp_reduce_delta_t(volatile deviceFloat *sdata, unsigned int tid);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    void reduce_delta_t(deviceFloat CFL, size_t N_elements, const SEM::Device::Entities::Element_t* elements, deviceFloat *g_odata);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    void warp_reduce_refine(volatile unsigned long *sdata, unsigned int tid);

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    void reduce_refine(size_t N_elements, deviceFloat delta_x_min, const SEM::Device::Entities::Element_t* elements, unsigned long *g_odata);
}}}

#include "meshes/Mesh_t.tcu"

#endif