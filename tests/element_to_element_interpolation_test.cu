#include <catch2/catch.hpp>
#include <cmath>
#include <array>
#include <vector>
#include "helpers/float_types.h"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "entities/NDG_t.cuh"
#include "entities/Element2D_t.cuh"
#include "entities/device_vector.cuh"
#include "functions/quad_map.cuh"

using SEM::Entities::Vec2;

__device__ const std::array<std::array<Vec2<deviceFloat>, 4>, 1> points {{Vec2<deviceFloat>{1, -1},
                                                                          Vec2<deviceFloat>{1, 1},
                                                                          Vec2<deviceFloat>{-1, 1},
                                                                          Vec2<deviceFloat>{-1, -1}}};

__device__ const std::array<std::array<Vec2<deviceFloat>, 4>, 1> points_small {{Vec2<deviceFloat>{1, -1},
                                                                                Vec2<deviceFloat>{1, 0},
                                                                                Vec2<deviceFloat>{0, 0},
                                                                                Vec2<deviceFloat>{0, -1}}};

__global__
auto element_to_element_projection_init(int N, size_t n_elements, SEM::Entities::Element2D_t* elements, SEM::Entities::Element2D_t* elements_small, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        std::array<SEM::Entities::cuda_vector<size_t>, 4> element_faces {SEM::Entities::cuda_vector<size_t>(1),
            SEM::Entities::cuda_vector<size_t>(1),
            SEM::Entities::cuda_vector<size_t>(1),
            SEM::Entities::cuda_vector<size_t>(1)};

        element_faces[0][0] = element_index;
        element_faces[1][0] = element_index;
        element_faces[2][0] = element_index;
        element_faces[3][0] = element_index;

        elements[element_index] = SEM::Entities::Element2D_t(N, 0, element_faces, std::array<size_t, 4>{4 * element_index, 4 * element_index + 1, 4 * element_index + 2, 4 * element_index + 3});
        elements_small[element_index] = SEM::Entities::Element2D_t(N, 1, element_faces, std::array<size_t, 4>{4 * element_index, 4 * element_index + 1, 4 * element_index + 2, 4 * element_index + 3});
        
        SEM::Entities::Element2D_t& element = elements[element_index];
        SEM::Entities::Element2D_t& element_small = elements_small[element_index];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
                const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points[element_index]);

                element.p_[i * (element.N_ + 1) + j] = std::sin(global_coordinates[0]) * std::cos(global_coordinates[1]);
                element.u_[i * (element.N_ + 1) + j] = global_coordinates[0];
                element.v_[i * (element.N_ + 1) + j] = global_coordinates[1];
            }
        }

        element_small.interpolate_from(points_small[element_index], points[element_index], element, polynomial_nodes, barycentric_weights);

        const size_t offset_results = element_index * (N + 1) * (N + 1);
        for (int i = 0; i < (N + 1) * (N + 1); ++i) {
            p[offset_results + i] = element_small.p_[i];
            u[offset_results + i] = element_small.u_[i];
            v[offset_results + i] = element_small.v_[i];
        }
    }
}

__global__
auto element_to_element_projection_init_2(int N, int N_high, size_t n_elements, SEM::Entities::Element2D_t* elements, SEM::Entities::Element2D_t* elements_high, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t element_index = index; element_index < n_elements; element_index += stride) {
        std::array<SEM::Entities::cuda_vector<size_t>, 4> element_faces {SEM::Entities::cuda_vector<size_t>(1),
            SEM::Entities::cuda_vector<size_t>(1),
            SEM::Entities::cuda_vector<size_t>(1),
            SEM::Entities::cuda_vector<size_t>(1)};

        element_faces[0][0] = element_index;
        element_faces[1][0] = element_index;
        element_faces[2][0] = element_index;
        element_faces[3][0] = element_index;

        elements[element_index] = SEM::Entities::Element2D_t(N, 0, element_faces, std::array<size_t, 4>{4 * element_index, 4 * element_index + 1, 4 * element_index + 2, 4 * element_index + 3});
        elements_high[element_index] = SEM::Entities::Element2D_t(N_high, 1, element_faces, std::array<size_t, 4>{4 * element_index, 4 * element_index + 1, 4 * element_index + 2, 4 * element_index + 3});
        
        SEM::Entities::Element2D_t& element = elements[element_index];
        SEM::Entities::Element2D_t& element_high = elements_high[element_index];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

        for (int i = 0; i <= element.N_; ++i) {
            for (int j = 0; j <= element.N_; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes[offset_1D + i], polynomial_nodes[offset_1D + j]};
                const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points[element_index]);

                element.p_[i * (element.N_ + 1) + j] = std::sin(global_coordinates[0]) * std::cos(global_coordinates[1]);
                element.u_[i * (element.N_ + 1) + j] = global_coordinates[0];
                element.v_[i * (element.N_ + 1) + j] = global_coordinates[1];
            }
        }

        element_high.interpolate_from(element, polynomial_nodes, barycentric_weights);

        const size_t offset_results = element_index * (N_high + 1) * (N_high + 1);
        for (int i = 0; i < (N_high + 1) * (N_high + 1); ++i) {
            p[offset_results + i] = element_high.p_[i];
            u[offset_results + i] = element_high.u_[i];
            v[offset_results + i] = element_high.v_[i];
        }
    }
}

TEST_CASE("Element to smaller element projection test", "Projects the solution from one element to a smaller one, as in h-adaptivity and checks the values match.") {   
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max;
    constexpr size_t n_elements = 1;
    const double max_error = 1e-9;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements(n_elements, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements_small(n_elements, stream);

    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;

    SEM::Entities::device_vector<deviceFloat> p(n_elements * (N_test + 1) * (N_test + 1));
    SEM::Entities::device_vector<deviceFloat> u(n_elements * (N_test + 1) * (N_test + 1));
    SEM::Entities::device_vector<deviceFloat> v(n_elements * (N_test + 1) * (N_test + 1));

    element_to_element_projection_init<<<elements_blockSize, elements_numBlocks, 0, stream>>>(N_test, n_elements, elements.data(), elements_small.data(), NDG.nodes_.data(), NDG.barycentric_weights_.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> polynomial_nodes_host(NDG.nodes_.size());
    NDG.nodes_.copy_to(polynomial_nodes_host, stream);

    std::vector<deviceFloat> p_host(n_elements * (N_test + 1) * (N_test + 1));
    std::vector<deviceFloat> u_host(n_elements * (N_test + 1) * (N_test + 1));
    std::vector<deviceFloat> v_host(n_elements * (N_test + 1) * (N_test + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);

    std::vector<std::vector<deviceFloat>> p_expected(n_elements);
    std::vector<std::vector<deviceFloat>> u_expected(n_elements);
    std::vector<std::vector<deviceFloat>> v_expected(n_elements);

    cudaStreamSynchronize(stream);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        p_expected[element_index] = std::vector<deviceFloat>((N_test + 1) * (N_test + 1));
        u_expected[element_index] = std::vector<deviceFloat>((N_test + 1) * (N_test + 1));
        v_expected[element_index] = std::vector<deviceFloat>((N_test + 1) * (N_test + 1));
        const size_t offset_1D = N_test * (N_test + 1) /2;

        for (size_t i = 0; i <= N_test; ++i) {
            for (size_t j = 0; j <= N_test; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes_host[offset_1D + i], polynomial_nodes_host[offset_1D + j]};
                const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points_small[element_index]);

                p_expected[element_index][i * (N_test + 1) + j] = std::sin(global_coordinates[0]) * std::cos(global_coordinates[1]);
                u_expected[element_index][i * (N_test + 1) + j] = global_coordinates[0];
                v_expected[element_index][i * (N_test + 1) + j] = global_coordinates[1];
            }
        }
    }

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset_results = element_index * (N_test + 1) * (N_test + 1);

        for (size_t i = 0; i <= N_test; ++i) {
            for (size_t j = 0; j <= N_test; ++j) {
                REQUIRE(std::abs(p_expected[element_index][i * (N_test + 1) + j] - p_host[offset_results + i * (N_test + 1) + j]) < max_error);
                REQUIRE(std::abs(u_expected[element_index][i * (N_test + 1) + j] - u_host[offset_results + i * (N_test + 1) + j]) < max_error);
                REQUIRE(std::abs(v_expected[element_index][i * (N_test + 1) + j] - v_host[offset_results + i * (N_test + 1) + j]) < max_error);
            }
        }
    }

    elements.clear(stream);
    elements_small.clear(stream);
    p.clear(stream);
    u.clear(stream);
    v.clear(stream);

    cudaStreamDestroy(stream);
}

TEST_CASE("Element to higher order element projection test", "Projects the solution from one element to a higher order one, as in p-adaptivity and checks the values match.") {   
    const int N_max = 16;
    const int N_test = 16;
    const int N_test_low = N_test - 2;
    const size_t N_interpolation_points = N_max;
    constexpr size_t n_elements = 1;
    const double max_error = 1e-9;

    REQUIRE(N_test <= N_max);
    REQUIRE(N_test_low <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements(n_elements, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements_high(n_elements, stream);

    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;

    SEM::Entities::device_vector<deviceFloat> p(n_elements * (N_test + 1) * (N_test + 1));
    SEM::Entities::device_vector<deviceFloat> u(n_elements * (N_test + 1) * (N_test + 1));
    SEM::Entities::device_vector<deviceFloat> v(n_elements * (N_test + 1) * (N_test + 1));

    element_to_element_projection_init_2<<<elements_blockSize, elements_numBlocks, 0, stream>>>(N_test_low, N_test, n_elements, elements.data(), elements_high.data(), NDG.nodes_.data(), NDG.barycentric_weights_.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> polynomial_nodes_host(NDG.nodes_.size());
    NDG.nodes_.copy_to(polynomial_nodes_host, stream);

    std::vector<deviceFloat> p_host(n_elements * (N_test + 1) * (N_test + 1));
    std::vector<deviceFloat> u_host(n_elements * (N_test + 1) * (N_test + 1));
    std::vector<deviceFloat> v_host(n_elements * (N_test + 1) * (N_test + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);

    std::vector<std::vector<deviceFloat>> p_expected(n_elements);
    std::vector<std::vector<deviceFloat>> u_expected(n_elements);
    std::vector<std::vector<deviceFloat>> v_expected(n_elements);

    cudaStreamSynchronize(stream);
    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        p_expected[element_index] = std::vector<deviceFloat>((N_test + 1) * (N_test + 1));
        u_expected[element_index] = std::vector<deviceFloat>((N_test + 1) * (N_test + 1));
        v_expected[element_index] = std::vector<deviceFloat>((N_test + 1) * (N_test + 1));
        const size_t offset_1D = N_test * (N_test + 1) /2;

        for (size_t i = 0; i <= N_test; ++i) {
            for (size_t j = 0; j <= N_test; ++j) {
                const Vec2<deviceFloat> coordinates {polynomial_nodes_host[offset_1D + i], polynomial_nodes_host[offset_1D + j]};
                const Vec2<deviceFloat> global_coordinates = SEM::quad_map(coordinates, points[element_index]);

                p_expected[element_index][i * (N_test + 1) + j] = std::sin(global_coordinates[0]) * std::cos(global_coordinates[1]);
                u_expected[element_index][i * (N_test + 1) + j] = global_coordinates[0];
                v_expected[element_index][i * (N_test + 1) + j] = global_coordinates[1];
            }
        }
    }

    for (size_t element_index = 0; element_index < n_elements; ++element_index) {
        const size_t offset_results = element_index * (N_test + 1) * (N_test + 1);

        for (size_t i = 0; i <= N_test; ++i) {
            for (size_t j = 0; j <= N_test; ++j) {
                REQUIRE(std::abs(p_expected[element_index][i * (N_test + 1) + j] - p_host[offset_results + i * (N_test + 1) + j]) < max_error);
                REQUIRE(std::abs(u_expected[element_index][i * (N_test + 1) + j] - u_host[offset_results + i * (N_test + 1) + j]) < max_error);
                REQUIRE(std::abs(v_expected[element_index][i * (N_test + 1) + j] - v_host[offset_results + i * (N_test + 1) + j]) < max_error);
            }
        }
    }

    elements.clear(stream);
    elements_high.clear(stream);
    p.clear(stream);
    u.clear(stream);
    v.clear(stream);

    cudaStreamDestroy(stream);
}
