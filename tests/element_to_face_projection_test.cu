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
#include "functions/quad_map.cuh"

using SEM::Entities::Vec2;

__device__ const std::array<Vec2<deviceFloat>, 4> points {Vec2<deviceFloat>{1, -1},
                                                          Vec2<deviceFloat>{1, 1},
                                                          Vec2<deviceFloat>{-1, 1},
                                                          Vec2<deviceFloat>{-1, -1}};

__device__ const std::array<Vec2<deviceFloat>, 8> faces_points {Vec2<deviceFloat>{1, -1},
                                                                Vec2<deviceFloat>{1, 0},
                                                                Vec2<deviceFloat>{1, 1},
                                                                Vec2<deviceFloat>{0, 1},
                                                                Vec2<deviceFloat>{-1, 1},
                                                                Vec2<deviceFloat>{-1, 0},
                                                                Vec2<deviceFloat>{-1, -1},
                                                                Vec2<deviceFloat>{0, -1}};

__global__
auto element_to_face_projection_init(int N, size_t n_elements, SEM::Entities::Element2D_t* elements, SEM::Entities::Face2D_t* faces, const deviceFloat* NDG_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        faces[4 * i] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{4 * i, 4 * i + 1}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{0, 2});
        faces[4 * i + 1] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{4 * i + 1, 4 * i + 2}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{1, 3});
        faces[4 * i + 2] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{4 * i + 3, 4 * i + 2}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{0, 2});
        faces[4 * i + 3] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{4 * i, 4 * i + 3}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{1, 3});

        faces[4 * i].offset_     = {0.0, 0.0};
        faces[4 * i].scale_      = {1.0, 1.0};
        faces[4 * i + 1].offset_ = {0.0, 0.0};
        faces[4 * i + 1].scale_  = {1.0, 1.0};
        faces[4 * i + 2].offset_ = {0.0, 0.0};
        faces[4 * i + 2].scale_  = {1.0, 1.0};
        faces[4 * i + 3].offset_ = {0.0, 0.0};
        faces[4 * i + 3].scale_  = {1.0, 1.0};

        std::array<SEM::Entities::cuda_vector<size_t>, 4> element_faces {SEM::Entities::cuda_vector<size_t>(1),
                                                                         SEM::Entities::cuda_vector<size_t>(1),
                                                                         SEM::Entities::cuda_vector<size_t>(1),
                                                                         SEM::Entities::cuda_vector<size_t>(1)};
        element_faces[0][0] = 4 * i;
        element_faces[1][0] = 4 * i + 1;
        element_faces[2][0] = 4 * i + 2;
        element_faces[3][0] = 4 * i + 3;

        elements[i] = SEM::Entities::Element2D_t(N, element_faces, std::array<size_t, 4>{4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3});

        SEM::Entities::Element2D_t& element = elements[i];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

        for (int i = 0; i <= element.N_; ++i) {
            const std::array<Vec2<deviceFloat>, 4> coordinates {Vec2<deviceFloat>{NDG_nodes[offset_1D + i], -1},
                                                                Vec2<deviceFloat>{1, NDG_nodes[offset_1D + i]},
                                                                Vec2<deviceFloat>{NDG_nodes[offset_1D + element.N_ - i], 1},
                                                                Vec2<deviceFloat>{-1, NDG_nodes[offset_1D + element.N_ - i]}};

            const std::array<Vec2<deviceFloat>, 4> element_points = {points[element.nodes_[0]],
                                                                     points[element.nodes_[1]],
                                                                     points[element.nodes_[2]],
                                                                     points[element.nodes_[3]]};

            const std::array<Vec2<deviceFloat>, 4> global_coordinates = {SEM::quad_map(coordinates[0], element_points),
                                                                         SEM::quad_map(coordinates[1], element_points),
                                                                         SEM::quad_map(coordinates[2], element_points),
                                                                         SEM::quad_map(coordinates[3], element_points)};

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
auto element_to_faces_projection_init(int N, size_t n_elements, SEM::Entities::Element2D_t* elements, SEM::Entities::Face2D_t* faces, const deviceFloat* NDG_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        faces[8 * i]     = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i, 8 * i + 1},     std::array<size_t, 2>{i, i}, std::array<size_t, 2>{0, 2});
        faces[8 * i + 1] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i + 1, 8 * i + 2}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{0, 2});
        faces[8 * i + 2] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i + 2, 8 * i + 3}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{1, 3});
        faces[8 * i + 3] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i + 3, 8 * i + 4}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{1, 3});
        faces[8 * i + 4] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i + 6, 8 * i + 5}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{0, 2});
        faces[8 * i + 5] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i + 5, 8 * i + 4}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{0, 2});
        faces[8 * i + 6] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i, 8 * i + 7},     std::array<size_t, 2>{i, i}, std::array<size_t, 2>{1, 3});
        faces[8 * i + 7] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i + 7, 8 * i + 6}, std::array<size_t, 2>{i, i}, std::array<size_t, 2>{1, 3});

        faces[8 * i].offset_     = {0.0, 0.5};
        faces[8 * i].scale_      = {0.5, 0.5};
        faces[8 * i + 1].offset_ = {0.5, 0.0};
        faces[8 * i + 1].scale_  = {0.5, 0.5};
        faces[8 * i + 2].offset_ = {0.0, 0.5};
        faces[8 * i + 2].scale_  = {0.5, 0.5};
        faces[8 * i + 3].offset_ = {0.5, 0.0};
        faces[8 * i + 3].scale_  = {0.5, 0.5};
        faces[8 * i + 4].offset_ = {0.0, 0.5};
        faces[8 * i + 4].scale_  = {0.5, 0.5};
        faces[8 * i + 5].offset_ = {0.5, 0.0};
        faces[8 * i + 5].scale_  = {0.5, 0.5};
        faces[8 * i + 6].offset_ = {0.0, 0.5};
        faces[8 * i + 6].scale_  = {0.5, 0.5};
        faces[8 * i + 7].offset_ = {0.5, 0.0};
        faces[8 * i + 7].scale_  = {0.5, 0.5};

        std::array<SEM::Entities::cuda_vector<size_t>, 4> element_faces {SEM::Entities::cuda_vector<size_t>(2),
                                                                         SEM::Entities::cuda_vector<size_t>(2),
                                                                         SEM::Entities::cuda_vector<size_t>(2),
                                                                         SEM::Entities::cuda_vector<size_t>(2)};
        element_faces[0][0] = 8 * i;
        element_faces[0][1] = 8 * i + 1;
        element_faces[1][0] = 8 * i + 2;
        element_faces[1][1] = 8 * i + 3;
        element_faces[2][0] = 8 * i + 4;
        element_faces[2][1] = 8 * i + 5;
        element_faces[3][0] = 8 * i + 6;
        element_faces[3][1] = 8 * i + 7;

        elements[i] = SEM::Entities::Element2D_t(N, element_faces, std::array<size_t, 4>{8 * i, 8 * i + 2, 8 * i + 4, 8 * i + 6});

        SEM::Entities::Element2D_t& element = elements[i];
        const size_t offset_1D = element.N_ * (element.N_ + 1) /2;

        for (int i = 0; i <= element.N_; ++i) {
            const std::array<Vec2<deviceFloat>, 4> coordinates {Vec2<deviceFloat>{NDG_nodes[offset_1D + i], -1},
                                                                Vec2<deviceFloat>{1, NDG_nodes[offset_1D + i]},
                                                                Vec2<deviceFloat>{NDG_nodes[offset_1D + element.N_ - i], 1},
                                                                Vec2<deviceFloat>{-1, NDG_nodes[offset_1D + element.N_ - i]}};

            const std::array<Vec2<deviceFloat>, 4> element_points = {faces_points[element.nodes_[0]],
                                                                     faces_points[element.nodes_[1]],
                                                                     faces_points[element.nodes_[2]],
                                                                     faces_points[element.nodes_[3]]};

            const std::array<Vec2<deviceFloat>, 4> global_coordinates = {SEM::quad_map(coordinates[0], element_points),
                                                                         SEM::quad_map(coordinates[1], element_points),
                                                                         SEM::quad_map(coordinates[2], element_points),
                                                                         SEM::quad_map(coordinates[3], element_points)};

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
        const SEM::Entities::Face2D_t& face_bottom = faces[element.faces_[0][0]];
        const SEM::Entities::Face2D_t& face_right = faces[element.faces_[1][0]];
        const SEM::Entities::Face2D_t& face_top = faces[element.faces_[2][0]];
        const SEM::Entities::Face2D_t& face_left = faces[element.faces_[3][0]];
        
        for (int j = 0; j <= N; ++j) {
            p[offset_results + j]               = face_bottom.p_[0][j];
            p[offset_results + (N + 1) + j]     = face_right.p_[0][j];
            p[offset_results + 2 * (N + 1) + j] = face_top.p_[1][j];
            p[offset_results + 3 * (N + 1) + j] = face_left.p_[1][j];

            u[offset_results + j]               = face_bottom.u_[0][j];
            u[offset_results + (N + 1) + j]     = face_right.u_[0][j];
            u[offset_results + 2 * (N + 1) + j] = face_top.u_[1][j];
            u[offset_results + 3 * (N + 1) + j] = face_left.u_[1][j];

            v[offset_results + j]               = face_bottom.v_[0][j];
            v[offset_results + (N + 1) + j]     = face_right.v_[0][j];
            v[offset_results + 2 * (N + 1) + j] = face_top.v_[1][j];
            v[offset_results + 3 * (N + 1) + j] = face_left.v_[1][j];
        }
    }
}

__global__
auto retrieve_faces_projected_solution(int N, size_t n_elements, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_elements; i += stride) {
        const size_t offset_results = 8 * i * (N + 1);
        const SEM::Entities::Element2D_t& element = elements[i];
        const SEM::Entities::Face2D_t& face_bottom   = faces[element.faces_[0][0]];
        const SEM::Entities::Face2D_t& face_bottom_1 = faces[element.faces_[0][1]];
        const SEM::Entities::Face2D_t& face_right    = faces[element.faces_[1][0]];
        const SEM::Entities::Face2D_t& face_right_1  = faces[element.faces_[1][1]];
        const SEM::Entities::Face2D_t& face_top      = faces[element.faces_[2][0]];
        const SEM::Entities::Face2D_t& face_top_1    = faces[element.faces_[2][1]];
        const SEM::Entities::Face2D_t& face_left     = faces[element.faces_[3][0]];
        const SEM::Entities::Face2D_t& face_left_1   = faces[element.faces_[3][1]];
        
        for (int j = 0; j <= N; ++j) {
            p[offset_results + j]               = face_bottom.p_[0][j];
            p[offset_results + (N + 1) + j]     = face_bottom_1.p_[0][j];
            p[offset_results + 2 * (N + 1) + j] = face_right.p_[0][j];
            p[offset_results + 3 * (N + 1) + j] = face_right_1.p_[0][j];
            p[offset_results + 4 * (N + 1) + j] = face_top.p_[1][j];
            p[offset_results + 5 * (N + 1) + j] = face_top_1.p_[1][j];
            p[offset_results + 6 * (N + 1) + j] = face_left.p_[1][j];
            p[offset_results + 7 * (N + 1) + j] = face_left_1.p_[1][j];

            u[offset_results + j]               = face_bottom.u_[0][j];
            u[offset_results + (N + 1) + j]     = face_bottom_1.u_[0][j];
            u[offset_results + 2 * (N + 1) + j] = face_right.u_[0][j];
            u[offset_results + 3 * (N + 1) + j] = face_right_1.u_[0][j];
            u[offset_results + 4 * (N + 1) + j] = face_top.u_[1][j];
            u[offset_results + 5 * (N + 1) + j] = face_top_1.u_[1][j];
            u[offset_results + 6 * (N + 1) + j] = face_left.u_[1][j];
            u[offset_results + 7 * (N + 1) + j] = face_left_1.u_[1][j];

            v[offset_results + j]               = face_bottom.v_[0][j];
            v[offset_results + (N + 1) + j]     = face_bottom_1.v_[0][j];
            v[offset_results + 2 * (N + 1) + j] = face_right.v_[0][j];
            v[offset_results + 3 * (N + 1) + j] = face_right_1.v_[0][j];
            v[offset_results + 4 * (N + 1) + j] = face_top.v_[1][j];
            v[offset_results + 5 * (N + 1) + j] = face_top_1.v_[1][j];
            v[offset_results + 6 * (N + 1) + j] = face_left.v_[1][j];
            v[offset_results + 7 * (N + 1) + j] = face_left_1.v_[1][j];
        }
    }
}

TEST_CASE("Element to face projection test", "Projects the edge interpolated solution of an element to a face and checks the values match.") {   
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
    SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces(4 * n_elements, stream);

    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;
    element_to_face_projection_init<<<elements_numBlocks, elements_blockSize, 0, stream>>>(N_test, n_elements, elements.data(), faces.data(), NDG.nodes_.data());

    constexpr int faces_blockSize = 32;
    constexpr int faces_numBlocks = (n_elements * 4 + faces_blockSize - 1) / faces_blockSize;
    SEM::Meshes::project_to_faces<<<faces_numBlocks, faces_blockSize, 0, stream>>>(n_elements * 4, faces.data(), elements.data(), NDG.nodes_.data(), NDG.barycentric_weights_.data());
    
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
            const std::array<Vec2<deviceFloat>, 4> global_coordinates = {points[4 * i + 1] * interp + points[4 * i] * (1 - interp),
                                                                         points[4 * i + 2] * interp + points[4 * i + 1] * (1 - interp),
                                                                         points[4 * i + 2] * interp + points[4 * i + 3] * (1 - interp),
                                                                         points[4 * i + 3] * interp + points[4 * i] * (1 - interp)}; // The last two faces are backwards, as if the element is the face's second element. 

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

    retrieve_face_projected_solution<<<elements_numBlocks, elements_blockSize, 0, stream>>>(N_test, n_elements, elements.data(), faces.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> p_host(n_elements * 4 * (N_test + 1));
    std::vector<deviceFloat> u_host(n_elements * 4 * (N_test + 1));
    std::vector<deviceFloat> v_host(n_elements * 4 * (N_test + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);

    for (size_t i = 0; i < n_elements; ++i) {
        const size_t offset_results = 4 * i * (N_test + 1);

        for (int j = 0; j <= N_test; ++j) {
            REQUIRE(std::abs(p_expected[i][0][j] - p_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][0][j] - u_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][0][j] - v_host[offset_results + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][1][j] - p_host[offset_results + N_test + 1 + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][1][j] - u_host[offset_results + N_test + 1 + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][1][j] - v_host[offset_results + N_test + 1 + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][2][j] - p_host[offset_results + 2 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][2][j] - u_host[offset_results + 2 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][2][j] - v_host[offset_results + 2 * (N_test + 1) + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][3][j] - p_host[offset_results + 3 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][3][j] - u_host[offset_results + 3 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][3][j] - v_host[offset_results + 3 * (N_test + 1) + j]) < max_error);
        }
    }

    cudaStreamDestroy(stream);
}

TEST_CASE("Element to two faces projection test", "Projects the edge interpolated solution of an element to two faces per side and checks the values match.") {   
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
    SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces(8 * n_elements, stream);

    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;
    element_to_faces_projection_init<<<elements_numBlocks, elements_blockSize, 0, stream>>>(N_test, n_elements, elements.data(), faces.data(), NDG.nodes_.data());

    constexpr int faces_blockSize = 32;
    constexpr int faces_numBlocks = (n_elements * 8 + faces_blockSize - 1) / faces_blockSize;
    SEM::Meshes::project_to_faces<<<faces_numBlocks, faces_blockSize, 0, stream>>>(n_elements * 8, faces.data(), elements.data(), NDG.nodes_.data(), NDG.barycentric_weights_.data());
    
    std::vector<deviceFloat> polynomial_nodes_host(NDG.nodes_.size());

    NDG.nodes_.copy_to(polynomial_nodes_host, stream);

    std::vector<std::array<std::vector<deviceFloat>, 8>> p_expected(n_elements);
    std::vector<std::array<std::vector<deviceFloat>, 8>> u_expected(n_elements);
    std::vector<std::array<std::vector<deviceFloat>, 8>> v_expected(n_elements);
    constexpr size_t offset_1D = N_test * (N_test + 1) /2;
    for (size_t i = 0; i < n_elements; ++i) {
        p_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
        u_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
        v_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
    
        for (int j = 0; j <= N_test; ++j) {
            const deviceFloat interp = (polynomial_nodes_host[offset_1D + j] + 1)/2;
            const std::array<Vec2<deviceFloat>, 8> global_coordinates = {faces_points[8 * i + 1] * interp + faces_points[8 * i]     * (1 - interp),
                                                                         faces_points[8 * i + 2] * interp + faces_points[8 * i + 1] * (1 - interp),
                                                                         faces_points[8 * i + 3] * interp + faces_points[8 * i + 2] * (1 - interp),
                                                                         faces_points[8 * i + 4] * interp + faces_points[8 * i + 3] * (1 - interp),
                                                                         faces_points[8 * i + 5] * interp + faces_points[8 * i + 6] * (1 - interp),
                                                                         faces_points[8 * i + 4] * interp + faces_points[8 * i + 5] * (1 - interp),
                                                                         faces_points[8 * i + 7] * interp + faces_points[8 * i] * (1 - interp),
                                                                         faces_points[8 * i + 6] * interp + faces_points[8 * i + 7] * (1 - interp)}; // The last four faces are backwards, as if the element is the face's second element. 

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

            p_expected[i][4][j] = std::sin(global_coordinates[4].x()) * std::cos(global_coordinates[4].y());
            u_expected[i][4][j] = global_coordinates[4].x();
            v_expected[i][4][j] = global_coordinates[4].y();

            p_expected[i][5][j] = std::sin(global_coordinates[5].x()) * std::cos(global_coordinates[5].y());
            u_expected[i][5][j] = global_coordinates[5].x();
            v_expected[i][5][j] = global_coordinates[5].y();

            p_expected[i][6][j] = std::sin(global_coordinates[6].x()) * std::cos(global_coordinates[6].y());
            u_expected[i][6][j] = global_coordinates[6].x();
            v_expected[i][6][j] = global_coordinates[6].y();

            p_expected[i][7][j] = std::sin(global_coordinates[7].x()) * std::cos(global_coordinates[7].y());
            u_expected[i][7][j] = global_coordinates[7].x();
            v_expected[i][7][j] = global_coordinates[7].y();
        }
    }

    SEM::Entities::device_vector<deviceFloat> p(n_elements * 8 * (N_test + 1), stream);
    SEM::Entities::device_vector<deviceFloat> u(n_elements * 8 * (N_test + 1), stream);
    SEM::Entities::device_vector<deviceFloat> v(n_elements * 8 * (N_test + 1), stream);

    retrieve_faces_projected_solution<<<elements_numBlocks, elements_blockSize, 0, stream>>>(N_test, n_elements, elements.data(), faces.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> p_host(n_elements * 8 * (N_test + 1));
    std::vector<deviceFloat> u_host(n_elements * 8 * (N_test + 1));
    std::vector<deviceFloat> v_host(n_elements * 8 * (N_test + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);

    for (size_t i = 0; i < n_elements; ++i) {
        const size_t offset_results = 8 * i * (N_test + 1);

        for (int j = 0; j <= N_test; ++j) {
            REQUIRE(std::abs(p_expected[i][0][j] - p_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][0][j] - u_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][0][j] - v_host[offset_results + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][1][j] - p_host[offset_results + N_test + 1 + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][1][j] - u_host[offset_results + N_test + 1 + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][1][j] - v_host[offset_results + N_test + 1 + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][2][j] - p_host[offset_results + 2 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][2][j] - u_host[offset_results + 2 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][2][j] - v_host[offset_results + 2 * (N_test + 1) + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][3][j] - p_host[offset_results + 3 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][3][j] - u_host[offset_results + 3 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][3][j] - v_host[offset_results + 3 * (N_test + 1) + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][4][j] - p_host[offset_results + 4 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][4][j] - u_host[offset_results + 4 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][4][j] - v_host[offset_results + 4 * (N_test + 1) + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][5][j] - p_host[offset_results + 5 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][5][j] - u_host[offset_results + 5 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][5][j] - v_host[offset_results + 5 * (N_test + 1) + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][6][j] - p_host[offset_results + 6 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][6][j] - u_host[offset_results + 6 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][6][j] - v_host[offset_results + 6 * (N_test + 1) + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][7][j] - p_host[offset_results + 7 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][7][j] - u_host[offset_results + 7 * (N_test + 1) + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][7][j] - v_host[offset_results + 7 * (N_test + 1) + j]) < max_error);
        }
    }

    cudaStreamDestroy(stream);
}