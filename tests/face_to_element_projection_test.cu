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

__device__ const std::array<Vec2<deviceFloat>, 6> points {Vec2<deviceFloat>{0, -1},
                                                          Vec2<deviceFloat>{0, 1},
                                                          Vec2<deviceFloat>{-1, 1},
                                                          Vec2<deviceFloat>{-1, -1},
                                                          Vec2<deviceFloat>{1, -1},
                                                          Vec2<deviceFloat>{1, 1}};

__global__
auto face_to_element_projection_init(int N, size_t n_faces, SEM::Entities::Element2D_t* elements, SEM::Entities::Face2D_t* faces, const deviceFloat* NDG_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_faces; i += stride) {
        std::array<SEM::Entities::cuda_vector<size_t>, 4> element_faces {SEM::Entities::cuda_vector<size_t>(1),
                                                                         SEM::Entities::cuda_vector<size_t>(1),
                                                                         SEM::Entities::cuda_vector<size_t>(1),
                                                                         SEM::Entities::cuda_vector<size_t>(1)};

        element_faces[0][0] = i;
        element_faces[1][0] = i;
        element_faces[2][0] = i;
        element_faces[3][0] = i;

        elements[2 * i]     = SEM::Entities::Element2D_t(N, element_faces, std::array<size_t, 4>{6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3});
        elements[2 * i + 1] = SEM::Entities::Element2D_t(N, element_faces, std::array<size_t, 4>{6 * i + 4, 6 * i + 5, 6 * i + 1, 6 * i});

        faces[i] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{6 * i, 6 * i + 1}, std::array<size_t, 2>{2 * i, 2 * i + 1}, std::array<size_t, 2>{0, 2});

        faces[i].offset_     = {0.0, 0.0};
        faces[i].scale_      = {1.0, 1.0};

        SEM::Entities::Face2D_t& face = faces[i];
        const size_t offset_1D = face.N_ * (face.N_ + 1) /2;

        for (int i = 0; i <= face.N_; ++i) {
            const deviceFloat interp = (polynomial_nodes_host[offset_1D + i] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = points[face.nodes_[1]] * interp + face.nodes_[0] * (1 - interp);

            face.p_flux_[i] = std::sin(global_coordinates.x()) * std::cos(global_coordinates.y());
            face.u_flux_[i] = global_coordinates.x();
            face.v_flux_[i] = global_coordinates.y();
        }
    }
}

__global__
auto retrieve_element_projected_solution(int N, size_t n_elements, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {

}

TEST_CASE("Face to element projection test", "Projects the face flux solution of a face to its elements and checks the values match.") {   
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max;
    constexpr size_t n_faces = 1;
    constexpr size_t n_elements = 2 * n_faces;
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements(n_elements, stream);
    SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces(n_faces, stream);

    constexpr int faces_blockSize = 32;
    constexpr int faces_numBlocks = (n_faces + faces_blockSize - 1) / faces_blockSize;
    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;

    face_to_element_projection_init<<<faces_blockSize, faces_numBlocks, 0, stream>>>(N_test, n_faces, elements.data(), faces.data(), NDG.nodes_.data());

    compute_element_geometry<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, elements, &points, NDG.nodes_.data());

    SEM::Meshes::project_to_elements<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, faces, elements, NDG.nodes_.data(), NDG.barycentric_weights_.data());



    
    
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