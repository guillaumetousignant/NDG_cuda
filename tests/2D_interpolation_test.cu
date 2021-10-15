#include <catch2/catch.hpp>
#include <cmath>
#include <array>
#include <vector>
#include "helpers/float_types.h"
#include "functions/quad_map.cuh"
#include "entities/NDG_t.cuh"
#include "entities/Element2D_t.cuh"
#include "entities/device_vector.cuh"
#include "polynomials/LegendrePolynomial_t.cuh"

using SEM::Device::Entities::Vec2;

__device__ const std::array<Vec2<deviceFloat>, 4> points {Vec2<deviceFloat>{1, -1},
                                                              Vec2<deviceFloat>{1, 1},
                                                              Vec2<deviceFloat>{-1, 1},
                                                              Vec2<deviceFloat>{-1, -1}};

__global__
auto elements_init(size_t n_elements, size_t n_interpolation_points, SEM::Device::Entities::Element2D_t* elements, const deviceFloat* NDG_nodes, const deviceFloat* interpolation_matrices, deviceFloat* x, deviceFloat* y, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        SEM::Device::Entities::Element2D_t& element = elements[i];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;
        const size_t offset_interp_2D = i * n_interpolation_points * n_interpolation_points;
        const size_t offset_interp = element.N_ * (element.N_ + 1) * n_interpolation_points/2;

        const int N = element.N_;
        element.p_ = SEM::Device::Entities::cuda_vector<deviceFloat>((N + 1) * (N + 1));
        element.u_ = SEM::Device::Entities::cuda_vector<deviceFloat>((N + 1) * (N + 1));
        element.v_ = SEM::Device::Entities::cuda_vector<deviceFloat>((N + 1) * (N + 1));

        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<deviceFloat> coordinates {NDG_nodes[offset_1D + i], NDG_nodes[offset_1D + j]};
                const Vec2<deviceFloat> global_coordinates = SEM::Device::quad_map(coordinates, points);

                element.p_[i * (element.N_ + 1) + j] = std::sin(global_coordinates[0]) * std::cos(global_coordinates[1]);
                element.u_[i * (element.N_ + 1) + j] = global_coordinates[0];
                element.v_[i * (element.N_ + 1) + j] = global_coordinates[1];
            }
        }

        element.interpolate_solution(n_interpolation_points, points, interpolation_matrices + offset_interp, x + offset_interp_2D, y + offset_interp_2D, p + offset_interp_2D, u + offset_interp_2D, v + offset_interp_2D);
    }
}

TEST_CASE("2D interpolation test", "Checks the interpolated value of the solution at the output interpolation points.") {   
    const int N_max = 16;
    const int N_test = 16;
    const size_t n_interpolation_points = std::pow(N_max, 2);
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 
    
    SEM::Device::Entities::NDG_t<SEM::Device::Polynomials::LegendrePolynomial_t> NDG(N_max, n_interpolation_points, stream);

    std::vector<SEM::Device::Entities::Element2D_t> host_elements(1);
    host_elements[0].N_ = N_test;

    SEM::Device::Entities::device_vector<SEM::Device::Entities::Element2D_t> device_elements(host_elements, stream);

    SEM::Device::Entities::device_vector<deviceFloat> x(n_interpolation_points * n_interpolation_points, stream);
    SEM::Device::Entities::device_vector<deviceFloat> y(n_interpolation_points * n_interpolation_points, stream);
    SEM::Device::Entities::device_vector<deviceFloat> p(n_interpolation_points * n_interpolation_points, stream);
    SEM::Device::Entities::device_vector<deviceFloat> u(n_interpolation_points * n_interpolation_points, stream);
    SEM::Device::Entities::device_vector<deviceFloat> v(n_interpolation_points * n_interpolation_points, stream);

    elements_init<<<1, 1, 0, stream>>>(1, n_interpolation_points, device_elements.data(), NDG.nodes_.data(), NDG.interpolation_matrices_.data(), x.data(), y.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> x_host(n_interpolation_points * n_interpolation_points);
    std::vector<deviceFloat> y_host(n_interpolation_points * n_interpolation_points);
    std::vector<deviceFloat> p_host(n_interpolation_points * n_interpolation_points);
    std::vector<deviceFloat> u_host(n_interpolation_points * n_interpolation_points);
    std::vector<deviceFloat> v_host(n_interpolation_points * n_interpolation_points);

    x.copy_to(x_host, stream);
    y.copy_to(y_host, stream);
    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < n_interpolation_points; ++i) {
        for (size_t j = 0; j < n_interpolation_points; ++j) {
            const Vec2<deviceFloat> coordinates {static_cast<deviceFloat>(i)/static_cast<deviceFloat>(n_interpolation_points - 1) * 2 - 1, static_cast<deviceFloat>(j)/static_cast<deviceFloat>(n_interpolation_points - 1) * 2 - 1};
            const Vec2<deviceFloat> global_coordinates = SEM::Device::quad_map(coordinates, points);

            const deviceFloat p = std::sin(global_coordinates[0]) * std::cos(global_coordinates[1]);
            const deviceFloat u = global_coordinates[0];
            const deviceFloat v = global_coordinates[1];
            
            REQUIRE(std::abs(global_coordinates.x() - x_host[i * n_interpolation_points + j]) < max_error);
            REQUIRE(std::abs(global_coordinates.y() - y_host[i * n_interpolation_points + j]) < max_error);
            REQUIRE(std::abs(p - p_host[i * n_interpolation_points + j]) < max_error);
            REQUIRE(std::abs(u - u_host[i * n_interpolation_points + j]) < max_error);
            REQUIRE(std::abs(v - v_host[i * n_interpolation_points + j]) < max_error);
        }
    }

    device_elements.clear(stream);
    x.clear(stream);
    y.clear(stream);
    p.clear(stream);
    u.clear(stream);
    v.clear(stream);

    cudaStreamDestroy(stream);
}