#ifndef NDG_SOLVERS_SOLVER2D_T_CUH
#define NDG_SOLVERS_SOLVER2D_T_CUH

#include "helpers/float_types.h"
#include "entities/NDG_t.cuh"
#include "helpers/DataWriter_t.h"
#include "meshes/Mesh2D_t.cuh"
#include "entities/Face2D_t.cuh"
#include "entities/Element2D_t.cuh"
#include <vector>
#include <array>

namespace SEM { namespace Device { namespace Solvers {
    class Solver2D_t {
        public:
            Solver2D_t(deviceFloat CFL, std::vector<deviceFloat> output_times, deviceFloat viscosity);

            deviceFloat CFL_;
            deviceFloat viscosity_;
            std::vector<deviceFloat> output_times_;

            template<typename Polynomial>
            auto solve(const SEM::Device::Entities::NDG_t<Polynomial> &NDG, SEM::Device::Meshes::Mesh2D_t& mesh, const SEM::Helpers::DataWriter_t& data_writer) const -> void;
            
            template<typename Polynomial>
            auto pre_condition(const SEM::Device::Entities::NDG_t<Polynomial> &NDG, SEM::Device::Meshes::Mesh2D_t& mesh, size_t n_adaptivity_steps) const -> void;

            auto get_delta_t(SEM::Device::Meshes::Mesh2D_t& mesh) const -> deviceFloat;

            __host__ __device__
            static auto x_flux(deviceFloat p, deviceFloat u, deviceFloat v) -> std::array<deviceFloat, 3>;

            __host__ __device__
            static auto y_flux(deviceFloat p, deviceFloat u, deviceFloat v) -> std::array<deviceFloat, 3>;

            // Algorithm 19
            __device__
            static auto matrix_vector_multiply(int N, const deviceFloat* matrix, const deviceFloat* vector, deviceFloat* result) -> void;
    };

    __global__
    auto calculate_wave_fluxes(size_t N_faces, SEM::Device::Entities::Face2D_t* faces) -> void;

    // Algorithm 114
    __global__
    auto compute_dg_wave_derivative(size_t N_elements, SEM::Device::Entities::Element2D_t* elements, const SEM::Device::Entities::Face2D_t* faces, const deviceFloat* weights, const deviceFloat* derivative_matrices_hat, const deviceFloat* lagrange_interpolant_left, const deviceFloat* lagrange_interpolant_right) -> void;

    __global__
    auto rk3_first_step(size_t N_elements, SEM::Device::Entities::Element2D_t* elements, deviceFloat delta_t, deviceFloat g) -> void;

    __global__
    auto rk3_step(size_t N_elements, SEM::Device::Entities::Element2D_t* elements, deviceFloat delta_t, deviceFloat a, deviceFloat g) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __device__ 
    auto warp_reduce_delta_t_2D(volatile deviceFloat *sdata, unsigned int tid) -> void;

    // From https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <unsigned int blockSize>
    __global__ 
    auto reduce_wave_delta_t(deviceFloat CFL, size_t N_elements, const SEM::Device::Entities::Element2D_t* elements, deviceFloat *g_odata) -> void;
}}}

#include "solvers/Solver2D_t.tcu"

#endif
