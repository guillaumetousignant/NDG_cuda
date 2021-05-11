#include <catch2/catch.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <vector>
#include "helpers/float_types.h"
#include "functions/quad_map.cuh"
#include "entities/NDG_t.cuh"
#include "entities/Element2D_t.cuh"
#include "entities/device_vector.cuh"
#include "meshes/Mesh2D_t.cuh"
#include "polynomials/ChebyshevPolynomial_t.cuh"

using SEM::Entities::Vec2;
using SEM::Entities::device_vector;

__device__ constexpr std::array<Vec2<deviceFloat>, 4> points {Vec2<deviceFloat>{1, -1},
                                                              Vec2<deviceFloat>{1, 1},
                                                              Vec2<deviceFloat>{-1, 1},
                                                              Vec2<deviceFloat>{-1, -1}};

__global__
auto elements_init(size_t n_elements, SEM::Entities::Element2D_t* elements, const deviceFloat* polynomial_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        SEM::Entities::Element2D_t& element = elements[i];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

        const int N = element.N_;
        element.p_ = SEM::Entities::cuda_vector<deviceFloat>((N + 1) * (N + 1));
        element.u_ = SEM::Entities::cuda_vector<deviceFloat>((N + 1) * (N + 1));
        element.v_ = SEM::Entities::cuda_vector<deviceFloat>((N + 1) * (N + 1));

        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
                const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points);

                element.p_[i * (element.N_ + 1) + j] = std::sin(global_coordinates[0]) * std::cos(global_coordinates[1]);
                element.u_[i * (element.N_ + 1) + j] = global_coordinates[0];
                element.v_[i * (element.N_ + 1) + j] = global_coordinates[1];
            }
        }
    }
}

__global__
auto get_boundary_solution(size_t n_elements, const SEM::Entities::Element2D_t* elements, std::array<deviceFloat*, 4> p, std::array<deviceFloat*, 4> u, std::array<deviceFloat*, 4> v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        for (size_t k = 0; k < elements[i].p_extrapolated_.size(); ++k){
            for (int j = 0; j <= elements[i].N_; ++j) {
                p[k][j] = elements[i].p_extrapolated_[k][j];
                u[k][j] = elements[i].u_extrapolated_[k][j];
                v[k][j] = elements[i].v_extrapolated_[k][j];
            }
        }
    }
}

TEST_CASE("2D boundary interpolation test", "Checks the interpolated value of the solution at the element's edges.") {   
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = std::pow(N_max, 2);
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::Entities::NDG_t<SEM::Polynomials::ChebyshevPolynomial_t> NDG(N_max, N_interpolation_points, stream);

    std::vector<SEM::Entities::Element2D_t> host_elements(1);
    host_elements[0].N_ = N_test;

    device_vector<SEM::Entities::Element2D_t> device_elements;
    device_elements = host_elements;

    elements_init<<<1, 1, 0, stream>>>(1, device_elements.data(), NDG.nodes_.data());

    SEM::Meshes::interpolate_to_boundaries<<<1, 1, 0, stream>>>(1, device_elements.data(), NDG.lagrange_interpolant_left_.data(), NDG.lagrange_interpolant_right_.data());

    std::array<device_vector<deviceFloat>, 4> p {N_test, N_test, N_test, N_test};
    std::array<device_vector<deviceFloat>, 4> u {N_test, N_test, N_test, N_test};
    std::array<device_vector<deviceFloat>, 4> v {N_test, N_test, N_test, N_test};
    get_boundary_solution<<<1, 1, 0, stream>>>(1, device_elements.data(), {p[0].data(), p[1].data(), p[2].data(), p[3].data()}, 
                                                                          {u[0].data(), u[1].data(), u[2].data(), u[3].data()}, 
                                                                          {v[0].data(), v[1].data(), v[2].data(), v[3].data()});

    std::array<std::vector<deviceFloat>, 4> p_host {std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test)};
    std::array<std::vector<deviceFloat>, 4> u_host {std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test)};
    std::array<std::vector<deviceFloat>, 4> v_host {std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test), std::vector<deviceFloat>(N_test)};

    for (size_t k = 0; k < p_host.size(); ++k) {
        p[k].copy_to(p_host[k]);
        u[k].copy_to(u_host[k]);
        v[k].copy_to(v_host[k]);
    }

    for (int i = 0; i <= N_test; ++i) {
        const std::array<Vec2<deviceFloat>, 4> coordinates {Vec2<deviceFloat>{static_cast<deviceFloat>(i)/static_cast<deviceFloat>(N_interpolation_points - 1) * 2 - 1, -1},
                                                            Vec2<deviceFloat>{1, static_cast<deviceFloat>(i)/static_cast<deviceFloat>(N_interpolation_points - 1) * 2 - 1},
                                                            Vec2<deviceFloat>{static_cast<deviceFloat>(i)/static_cast<deviceFloat>(N_interpolation_points - 1) * 2 - 1, 1},
                                                            Vec2<deviceFloat>{-1, static_cast<deviceFloat>(i)/static_cast<deviceFloat>(N_interpolation_points - 1) * 2 - 1}};
        const std::array<Vec2<deviceFloat>, 4> global_coordinates {SEM::quad_map(coordinates[0], points),
                                                                   SEM::quad_map(coordinates[1], points),
                                                                   SEM::quad_map(coordinates[2], points),
                                                                   SEM::quad_map(coordinates[3], points)};
            
        for (size_t k = 0; k < coordinates.size(); ++k) {
            const deviceFloat p_target = std::sin(global_coordinates[k][0]) * std::cos(global_coordinates[k][1]);
            const deviceFloat u_target = global_coordinates[k][0];
            const deviceFloat v_target = global_coordinates[k][1];

            REQUIRE(std::abs(p_target - p_host[k][i]) < max_error);
            REQUIRE(std::abs(u_target - u_host[k][i]) < max_error);
            REQUIRE(std::abs(v_target - v_host[k][i]) < max_error);
        }
    }

    cudaStreamDestroy(stream);
}