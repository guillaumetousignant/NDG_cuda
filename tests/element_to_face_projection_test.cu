#include <catch2/catch.hpp>
#include <cmath>
#include <array>
#include <vector>
#include "helpers/float_types.h"
#include "polynomials/LegendrePolynomial_t.cuh"
#include "entities/NDG_t.cuh"
#include "entities/Face2D_t.cuh"
#include "entities/Element2D_t.cuh"
#include "meshes/Mesh2D_t.cuh"
#include "entities/device_vector.cuh"

using SEM::Entities::Vec2;

__device__ const std::array<Vec2<deviceFloat>, 4> points {Vec2<deviceFloat>{1, -1},
                                                          Vec2<deviceFloat>{1, 1},
                                                          Vec2<deviceFloat>{-1, 1},
                                                          Vec2<deviceFloat>{-1, -1}};

__global__
auto elements_to_face_projection_init(int N, size_t n_elements, SEM::Entities::Element2D_t* elements, SEM::Entities::Face2D_t* faces, const deviceFloat* NDG_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        faces[4 * i] = SEM::Entities::Face2D_t(N, {points[4 * i], points[4 * i + 1]}, {i, i}, {0, 2});
        faces[4 * i + 1] = SEM::Entities::Face2D_t(N, {points[4 * i + 1], points[4 * i + 2]}, {i, i}, {1, 3});
        faces[4 * i + 2] = SEM::Entities::Face2D_t(N, {points[4 * i + 3], points[4 * i +  + 2]}, {i, i}, {0, 2});
        faces[4 * i + 3] = SEM::Entities::Face2D_t(N, {points[4 * i], points[4 * i + 3]}, {3, 1}, 1, 3});

        elements[i] = SEM::Entities::Element2D_t({4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3}, {points[4 * i], points[4 * i + 1], points[4 * i + 2], points[4 * i + 3]});

        SEM::Entities::Element2D_t& element = elements[i];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

        for (int i = 0; i <= element.N_; ++i) {
            const std::array<Vec2<deviceFloat>, 4> coordinates {Vec2<deviceFloat>{NDG_nodes[offset_1D + i], -1},
                                                                Vec2<deviceFloat>{1, NDG_nodes[offset_1D + i]},
                                                                Vec2<deviceFloat>{NDG_nodes[offset_1D + element.N_ - i], 1},
                                                                Vec2<deviceFloat>{-1, NDG_nodes[offset_1D + element.N_ - i]}};

            const td::array<Vec2<deviceFloat>, 4> global_coordinates = {SEM::quad_map(coordinates[0], points),
                                                                        SEM::quad_map(coordinates[1], points),
                                                                        SEM::quad_map(coordinates[2], points),
                                                                        SEM::quad_map(coordinates[3], points)};

            element.p_extrapolated_[0][i] = std::sin(global_coordinates[0].x()) * std::cos(global_coordinates[0].y());
            element.u_extrapolated_[0][i] = global_coordinates[0].x();
            element.v_extrapolated_[0][i] = global_coordinates[0].y();

            element.p_extrapolated_[1][i] = std::sin(global_coordinates[1].x()) * std::cos(global_coordinates[1].y());
            element.u_extrapolated_[1][i] = global_coordinates[1].x();
            element.v_extrapolated_[1][i] = global_coordinates[1].y();

            element.p_extrapolated_[2][i] = std::sin(global_coordinates[2].x()) * std::cos(global_coordinates[2].y());
            element.u_extrapolated_[2][i] = global_coordinates[2].x();
            element.v_extrapolated_[2][i] = global_coordinates[2].y();

            element.p_extrapolated_[3][i] = std::sin(global_coordinates[3].x()) * std::cos(global_coordinates[3].y());
            element.u_extrapolated_[3][i] = global_coordinates[3].x();
            element.v_extrapolated_[3][i] = global_coordinates[3].y();
        }
    }
}

__global__
auto retrieve_face_projected_solution(int N, size_t n_elements, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        const size_t offset_results = 4 * i * (N + 1);
        const SEM::Entities::Element2D_t& element = elements[i];
        const SEM::Entities::Face2D_t& face_bottom = faces[element.faces_[0]];
        const SEM::Entities::Face2D_t& face_right = faces[element.faces_[1]];
        const SEM::Entities::Face2D_t& face_top = faces[element.faces_[2]];
        const SEM::Entities::Face2D_t& face_left = faces[element.faces_[3]];
        
        for (int j = 0; j <= N; ++j) {
            p[offset_results + j] = face_bottom.p_[0][j];
            p[offset_results + (N + 1) + j] = face_right.p_[0][j];
            p[offset_results + 2 * (N + 1) + j] = face_top.p_[1][j];
            p[offset_results + 3 * (N + 1) + j] = face_left.p_[1][j];

            u[offset_results + j] = face_bottom.u_[0][j];
            u[offset_results + (N + 1) + j] = face_right.u_[0][j];
            u[offset_results + 2 * (N + 1) + j] = face_top.u_[1][j];
            u[offset_results + 3 * (N + 1) + j] = face_left.u_[1][j];

            v[offset_results + j] = face_bottom.v_[0][j];
            v[offset_results + (N + 1) + j] = face_right.v_[0][j];
            v[offset_results + 2 * (N + 1) + j] = face_top.v_[1][j];
            v[offset_results + 3 * (N + 1) + j] = face_left.v_[1][j];
        }
    }
}

TEST_CASE("Element to face projects test", "Projects the edge interpolated solution of an element to a face and checks the values match.") {   
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max;
    constexpr size_t n_elements = 1;
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements(n_elements, stream);
    SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces(4 * n_elements, stream);

    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;
    elements_to_face_projection_init<<<elements_numBlocks, elements_blockSize, 0, stream>>>(N_test, n_elements, elements.data(), faces.data(), NDG.nodes_.data()) -> void {

    constexpr int faces_blockSize = 32;
    constexpr int faces_numBlocks = (n_elements * 4 + faces_blockSize - 1) / faces_blockSize;
    SEM::Meshes::project_to_faces<<<faces_numBlocks, faces_blockSize, 0, stream>>>(n_elements * 4, Face2D_t* faces, const Element2D_t* elements, const deviceFloat* polynomial_nodes, const deviceFloat* barycentric_weights) -> void {
    
    std::vector<deviceFloat> polynomial_nodes_host(NDG.nodes_.size());

    NDG.nodes_.copy_to(polynomial_nodes_host, stream);

    std::vector<std::array<std::vector<deviceFloat>, 4>> p_expected(n_elements);
    std::vector<std::array<std::vector<deviceFloat>, 4>> u_expected(n_elements);
    std::vector<std::array<std::vector<deviceFloat>, 4>> v_expected(n_elements);
    constexpr size_t offset_1D = N_test * (N_test + 1) /2;
    for (size_t i = 0; i < n_elements; ++i) {
        p_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
        u_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
        v_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
    
        for (int j = 0; j <= N_test; ++j) {
            const deviceFloat interp = (polynomial_nodes_host[offset_1D + j] + 1)/2;
            const td::array<Vec2<deviceFloat>, 4> global_coordinates = {points[4 * i + 1] * interp + points[4 * i] * (1 - interp),
                                                                        points[4 * i + 2] * interp + points[4 * i + 1] * (1 - interp),
                                                                        points[4 * i + 3] * interp + points[4 * i + 2] * (1 - interp),
                                                                        points[4 * i] * interp + points[4 * i + 3] * (1 - interp)};

            p_expected[i][0][j] = std::sin(global_coordinates[0].x()) * std::cos(global_coordinates[0].y());
            u_expected[i][0][j] = global_coordinates[0].x();
            v_expected[i][0][j] = global_coordinates[0].y();

            p_expected[i][1][j] = std::sin(global_coordinates[1].x()) * std::cos(global_coordinates[1].y());
            u_expected[i][1][j] = global_coordinates[1].x();
            v_expected[i][1][j] = global_coordinates[1].y();

            p_expected[i][2][j] = std::sin(global_coordinates[2].x()) * std::cos(global_coordinates[2].y());
            u_expected[i][2][j] = global_coordinates[2].x();
            v_expected[i][2][j] = global_coordinates[2].y();

            p_expected[i][3][j] = std::sin(global_coordinates[3].x()) * std::cos(global_coordinates[3].y());
            u_expected[i][3][j] = global_coordinates[3].x();
            v_expected[i][3][j] = global_coordinates[3].y();
        }
    }

    SEM::Entities::device_vector<deviceFloat> p(n_elements * 4 * (N_test + 1), stream);
    SEM::Entities::device_vector<deviceFloat> u(n_elements * 4 * (N_test + 1), stream);
    SEM::Entities::device_vector<deviceFloat> v(n_elements * 4 * (N_test + 1), stream);

    retrieve_face_projected_solution<<<elements_numBlocks, elements_blockSize, 0, stream>>>(iN_test, n_elements, elements.data(), faces.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> p_host(n_elements * 4 * (N_test + 1));
    std::vector<deviceFloat> u_host(n_elements * 4 * (N_test + 1));
    std::vector<deviceFloat> v_host(n_elements * 4 * (N_test + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);

    for (size_t i = 0; i < n_elements; ++i) {
        const size_t offset_results = 4 * i * (N + 1);

        for (int j = 0; j <= N_test; ++j) {
            REQUIRE(std::abs(p_expected[i][0][j] - p_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][0][j] - u_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][0][j] - v_host[offset_results + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][1][j] - p_host[offset_results + N + 1 + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][1][j] - u_host[offset_results + N + 1 + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][1][j] - v_host[offset_results + N + 1 + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][2][j] - p_host[offset_results + 2 * (N + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][2][j] - u_host[offset_results + 2 * (N + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][2][j] - v_host[offset_results + 2 * (N + 1) + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][3][j] - p_host[offset_results + 3 * (N + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][3][j] - u_host[offset_results + 3 * (N + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][3][j] - v_host[offset_results + 3 * (N + 1) + j]) < max_error);

        }
    }

    cudaStreamDestroy(stream);
}