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

__global__
auto face_to_element_projection_init(int N_test_elements, int N_test_faces, size_t n_faces, SEM::Entities::Element2D_t* elements, SEM::Entities::Face2D_t* faces, const Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void {
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

        elements[2 * i]     = SEM::Entities::Element2D_t(N_test_elements, 0, element_faces, std::array<size_t, 4>{6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3});
        elements[2 * i + 1] = SEM::Entities::Element2D_t(N_test_elements, 0, element_faces, std::array<size_t, 4>{6 * i + 4, 6 * i + 5, 6 * i + 1, 6 * i});

        faces[i] = SEM::Entities::Face2D_t(N_test_faces, std::array<size_t, 2>{6 * i, 6 * i + 1}, std::array<size_t, 2>{2 * i, 2 * i + 1}, std::array<size_t, 2>{0, 2});

        faces[i].offset_     = {0.0, 0.0};
        faces[i].scale_      = {1.0, 1.0};

        SEM::Entities::Face2D_t& face = faces[i];
        const size_t offset_1D = face.N_ * (face.N_ + 1) /2;

        for (int i = 0; i <= face.N_; ++i) {
            const deviceFloat interp = (polynomial_nodes[offset_1D + i] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = nodes[face.nodes_[1]] * interp + nodes[face.nodes_[0]] * (1 - interp);

            face.p_flux_[i] = std::cos(global_coordinates.y());
            face.u_flux_[i] = global_coordinates.y() * global_coordinates.y();
            face.v_flux_[i] = global_coordinates.y();
        }
    }
}

__global__
auto faces_to_element_projection_init(int N, size_t n_faces, SEM::Entities::Element2D_t* elements, SEM::Entities::Face2D_t* faces, const Vec2<deviceFloat>* nodes, const deviceFloat* polynomial_nodes) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_faces/2; i += stride) {
        std::array<SEM::Entities::cuda_vector<size_t>, 4> element_faces {SEM::Entities::cuda_vector<size_t>(2),
                                                                         SEM::Entities::cuda_vector<size_t>(2),
                                                                         SEM::Entities::cuda_vector<size_t>(2),
                                                                         SEM::Entities::cuda_vector<size_t>(2)};

        element_faces[0][0] = 2 * i;
        element_faces[0][1] = 2 * i + 1;
        element_faces[1][0] = 2 * i;
        element_faces[1][1] = 2 * i + 1;
        element_faces[2][0] = 2 * i;
        element_faces[2][1] = 2 * i + 1;
        element_faces[3][0] = 2 * i;
        element_faces[3][1] = 2 * i + 1;

        elements[2 * i]     = SEM::Entities::Element2D_t(N, 0, element_faces, std::array<size_t, 4>{6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3});
        elements[2 * i + 1] = SEM::Entities::Element2D_t(N, 0, element_faces, std::array<size_t, 4>{6 * i + 4, 6 * i + 5, 6 * i + 1, 6 * i});

        faces[2 * i]     = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i, 8 * i + 1},     std::array<size_t, 2>{2 * i, 2 * i + 1}, std::array<size_t, 2>{0, 2});
        faces[2 * i + 1] = SEM::Entities::Face2D_t(N, std::array<size_t, 2>{8 * i + 1, 8 * i + 2}, std::array<size_t, 2>{2 * i, 2 * i + 1}, std::array<size_t, 2>{0, 2});

        faces[2 * i].offset_     = {0.0, 0.5};
        faces[2 * i].scale_      = {0.5, 0.5};
        faces[2 * i + 1].offset_ = {0.5, 0.0};
        faces[2 * i + 1].scale_  = {0.5, 0.5};

        SEM::Entities::Face2D_t& face_0 = faces[2 * i];
        SEM::Entities::Face2D_t& face_1 = faces[2 * i + 1];
        const size_t offset_1D_0 = face_0.N_ * (face_0.N_ + 1) /2;
        const size_t offset_1D_1 = face_1.N_ * (face_1.N_ + 1) /2;

        for (int i = 0; i <= face_0.N_; ++i) {
            const deviceFloat interp = (polynomial_nodes[offset_1D_0 + i] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = nodes[face_0.nodes_[1]] * interp + nodes[face_0.nodes_[0]] * (1 - interp);

            face_0.p_flux_[i] = std::cos(global_coordinates.y());
            face_0.u_flux_[i] = global_coordinates.y() * global_coordinates.y();
            face_0.v_flux_[i] = global_coordinates.y();
        }
        for (int i = 0; i <= face_1.N_; ++i) {
            const deviceFloat interp = (polynomial_nodes[offset_1D_1 + i] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = nodes[face_1.nodes_[1]] * interp + nodes[face_1.nodes_[0]] * (1 - interp);

            face_1.p_flux_[i] = std::cos(global_coordinates.y());
            face_1.u_flux_[i] = global_coordinates.y() * global_coordinates.y();
            face_1.v_flux_[i] = global_coordinates.y();
        }
    }
}

__global__
auto retrieve_element_projected_solution(int N, size_t n_faces, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_faces; i += stride) {
        const size_t offset_results = 2 * i * (N + 1);
        const SEM::Entities::Face2D_t& face = faces[i];
        const SEM::Entities::Element2D_t& element_left = elements[face.elements_[0]];
        const SEM::Entities::Element2D_t& element_right = elements[face.elements_[1]];
        const size_t element_side_left = face.elements_side_[0];
        const size_t element_side_right = face.elements_side_[1];
        
        for (int j = 0; j <= N; ++j) {
            p[offset_results + j]               = element_left.p_flux_extrapolated_[element_side_left][j];
            p[offset_results + (N + 1) + j]     = element_right.p_flux_extrapolated_[element_side_right][j];

            u[offset_results + j]               = element_left.u_flux_extrapolated_[element_side_left][j];
            u[offset_results + (N + 1) + j]     = element_right.u_flux_extrapolated_[element_side_right][j];

            v[offset_results + j]               = element_left.v_flux_extrapolated_[element_side_left][j];
            v[offset_results + (N + 1) + j]     = element_right.v_flux_extrapolated_[element_side_right][j];
        }
    }
}

__global__
auto retrieve_element_projected_solution_2(int N, size_t n_faces, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_faces/2; i += stride) {
        const size_t offset_results = 2 * i * (N + 1);
        const SEM::Entities::Face2D_t& face = faces[2 * i];
        const SEM::Entities::Element2D_t& element_left = elements[face.elements_[0]];
        const SEM::Entities::Element2D_t& element_right = elements[face.elements_[1]];
        const size_t element_side_left = face.elements_side_[0];
        const size_t element_side_right = face.elements_side_[1];
        
        for (int j = 0; j <= N; ++j) {
            p[offset_results + j]               = element_left.p_flux_extrapolated_[element_side_left][j];
            p[offset_results + (N + 1) + j]     = element_right.p_flux_extrapolated_[element_side_right][j];

            u[offset_results + j]               = element_left.u_flux_extrapolated_[element_side_left][j];
            u[offset_results + (N + 1) + j]     = element_right.u_flux_extrapolated_[element_side_right][j];

            v[offset_results + j]               = element_left.v_flux_extrapolated_[element_side_left][j];
            v[offset_results + (N + 1) + j]     = element_right.v_flux_extrapolated_[element_side_right][j];
        }
    }
}

TEST_CASE("Face to element projection test", "Projects the face flux solution of a face to its elements and checks the values match.") {   
    const int N_max = 16;
    const int N_test_faces = 16;
    const int N_test_elements = N_test_faces;
    const size_t N_interpolation_points = N_max;
    constexpr size_t n_faces = 1;
    constexpr size_t n_elements = 2 * n_faces;
    const double max_error = 1e-9;

    REQUIRE(N_test_elements <= N_max);
    REQUIRE(N_test_faces <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    const std::vector<Vec2<deviceFloat>> host_nodes {Vec2<deviceFloat>{0, -1},
                                                     Vec2<deviceFloat>{0, 1},
                                                     Vec2<deviceFloat>{-1, 1},
                                                     Vec2<deviceFloat>{-1, -1},
                                                     Vec2<deviceFloat>{1, -1},
                                                     Vec2<deviceFloat>{1, 1}};

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements(n_elements, stream);
    SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces(n_faces, stream);
    SEM::Entities::device_vector<Vec2<deviceFloat>> nodes(host_nodes, stream);

    constexpr int faces_blockSize = 32;
    constexpr int faces_numBlocks = (n_faces + faces_blockSize - 1) / faces_blockSize;
    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;

    face_to_element_projection_init<<<faces_blockSize, faces_numBlocks, 0, stream>>>(N_test_elements, N_test_faces, n_faces, elements.data(), faces.data(), nodes.data(), NDG.nodes_.data());

    SEM::Meshes::compute_element_geometry<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, elements.data(), nodes.data(), NDG.nodes_.data());

    SEM::Meshes::project_to_elements<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, faces.data(), elements.data(), NDG.nodes_.data(), NDG.weights_.data(), NDG.barycentric_weights_.data());
    
    std::vector<deviceFloat> polynomial_nodes_host(NDG.nodes_.size());

    NDG.nodes_.copy_to(polynomial_nodes_host, stream);

    std::vector<std::array<std::vector<deviceFloat>, 2>> p_expected(n_faces);
    std::vector<std::array<std::vector<deviceFloat>, 2>> u_expected(n_faces);
    std::vector<std::array<std::vector<deviceFloat>, 2>> v_expected(n_faces);
    
    cudaStreamSynchronize(stream);
    constexpr size_t offset_1D = N_test_elements * (N_test_elements + 1) /2;
    for (size_t i = 0; i < n_faces; ++i) {
        p_expected[i] = {std::vector<deviceFloat>(N_test_elements + 1), std::vector<deviceFloat>(N_test_elements + 1)};
        u_expected[i] = {std::vector<deviceFloat>(N_test_elements + 1), std::vector<deviceFloat>(N_test_elements + 1)};
        v_expected[i] = {std::vector<deviceFloat>(N_test_elements + 1), std::vector<deviceFloat>(N_test_elements + 1)};
    
        for (int j = 0; j <= N_test_elements; ++j) {
            const deviceFloat interp = (polynomial_nodes_host[offset_1D + j] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = host_nodes[6 * i + 1] * interp + host_nodes[6 * i] * (1 - interp);
            
            p_expected[i][0][j] = std::cos(global_coordinates.y());
            u_expected[i][0][j] = global_coordinates.y() * global_coordinates.y();
            v_expected[i][0][j] = global_coordinates.y();

            p_expected[i][1][N_test_elements - j] = -std::cos(global_coordinates.y());
            u_expected[i][1][N_test_elements - j] = -global_coordinates.y() * global_coordinates.y();
            v_expected[i][1][N_test_elements - j] = -global_coordinates.y();
        }
    }

    SEM::Entities::device_vector<deviceFloat> p(n_elements * (N_test_elements + 1), stream);
    SEM::Entities::device_vector<deviceFloat> u(n_elements * (N_test_elements + 1), stream);
    SEM::Entities::device_vector<deviceFloat> v(n_elements * (N_test_elements + 1), stream);

    retrieve_element_projected_solution<<<elements_numBlocks, elements_blockSize, 0, stream>>>(N_test_elements, n_faces, elements.data(), faces.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> p_host(n_elements * (N_test_elements + 1));
    std::vector<deviceFloat> u_host(n_elements * (N_test_elements + 1));
    std::vector<deviceFloat> v_host(n_elements * (N_test_elements + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < n_faces; ++i) {
        const size_t offset_results = 2 * i * (N_test_elements + 1);

        for (int j = 0; j <= N_test_elements; ++j) {
            REQUIRE(std::abs(p_expected[i][0][j] - p_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][0][j] - u_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][0][j] - v_host[offset_results + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][1][j] - p_host[offset_results + N_test_elements + 1 + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][1][j] - u_host[offset_results + N_test_elements + 1 + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][1][j] - v_host[offset_results + N_test_elements + 1 + j]) < max_error);
        }
    }

    elements.clear(stream);
    faces.clear(stream);
    nodes.clear(stream);
    p.clear(stream);
    u.clear(stream);
    v.clear(stream);

    cudaStreamDestroy(stream);
}

TEST_CASE("Face to lower order element projection test", "Projects the face flux solution of a face to its lower order elements and checks the values match.") {   
    const int N_max = 16;
    const int N_test_faces = 16;
    const int N_test_elements = N_test_faces - 2;
    const size_t N_interpolation_points = N_max;
    constexpr size_t n_faces = 1;
    constexpr size_t n_elements = 2 * n_faces;
    const double max_error = 1e-9;

    REQUIRE(N_test_elements <= N_max);
    REQUIRE(N_test_faces <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    const std::vector<Vec2<deviceFloat>> host_nodes {Vec2<deviceFloat>{0, -1},
                                                     Vec2<deviceFloat>{0, 1},
                                                     Vec2<deviceFloat>{-1, 1},
                                                     Vec2<deviceFloat>{-1, -1},
                                                     Vec2<deviceFloat>{1, -1},
                                                     Vec2<deviceFloat>{1, 1}};

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements(n_elements, stream);
    SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces(n_faces, stream);
    SEM::Entities::device_vector<Vec2<deviceFloat>> nodes(host_nodes, stream);

    constexpr int faces_blockSize = 32;
    constexpr int faces_numBlocks = (n_faces + faces_blockSize - 1) / faces_blockSize;
    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;

    face_to_element_projection_init<<<faces_blockSize, faces_numBlocks, 0, stream>>>(N_test_elements, N_test_faces, n_faces, elements.data(), faces.data(), nodes.data(), NDG.nodes_.data());

    SEM::Meshes::compute_element_geometry<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, elements.data(), nodes.data(), NDG.nodes_.data());

    SEM::Meshes::project_to_elements<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, faces.data(), elements.data(), NDG.nodes_.data(), NDG.weights_.data(), NDG.barycentric_weights_.data());
    
    std::vector<deviceFloat> polynomial_nodes_host(NDG.nodes_.size());

    NDG.nodes_.copy_to(polynomial_nodes_host, stream);

    std::vector<std::array<std::vector<deviceFloat>, 2>> p_expected(n_faces);
    std::vector<std::array<std::vector<deviceFloat>, 2>> u_expected(n_faces);
    std::vector<std::array<std::vector<deviceFloat>, 2>> v_expected(n_faces);
    
    cudaStreamSynchronize(stream);
    constexpr size_t offset_1D = N_test_elements * (N_test_elements + 1) /2;
    for (size_t i = 0; i < n_faces; ++i) {
        p_expected[i] = {std::vector<deviceFloat>(N_test_elements + 1), std::vector<deviceFloat>(N_test_elements + 1)};
        u_expected[i] = {std::vector<deviceFloat>(N_test_elements + 1), std::vector<deviceFloat>(N_test_elements + 1)};
        v_expected[i] = {std::vector<deviceFloat>(N_test_elements + 1), std::vector<deviceFloat>(N_test_elements + 1)};
    
        for (int j = 0; j <= N_test_elements; ++j) {
            const deviceFloat interp = (polynomial_nodes_host[offset_1D + j] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = host_nodes[6 * i + 1] * interp + host_nodes[6 * i] * (1 - interp);
            
            p_expected[i][0][j] = std::cos(global_coordinates.y());
            u_expected[i][0][j] = global_coordinates.y() * global_coordinates.y();
            v_expected[i][0][j] = global_coordinates.y();

            p_expected[i][1][N_test_elements - j] = -std::cos(global_coordinates.y());
            u_expected[i][1][N_test_elements - j] = -global_coordinates.y() * global_coordinates.y();
            v_expected[i][1][N_test_elements - j] = -global_coordinates.y();
        }
    }

    SEM::Entities::device_vector<deviceFloat> p(n_elements * (N_test_elements + 1), stream);
    SEM::Entities::device_vector<deviceFloat> u(n_elements * (N_test_elements + 1), stream);
    SEM::Entities::device_vector<deviceFloat> v(n_elements * (N_test_elements + 1), stream);

    retrieve_element_projected_solution<<<elements_numBlocks, elements_blockSize, 0, stream>>>(N_test_elements, n_faces, elements.data(), faces.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> p_host(n_elements * (N_test_elements + 1));
    std::vector<deviceFloat> u_host(n_elements * (N_test_elements + 1));
    std::vector<deviceFloat> v_host(n_elements * (N_test_elements + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < n_faces; ++i) {
        const size_t offset_results = 2 * i * (N_test_elements + 1);

        for (int j = 0; j <= N_test_elements; ++j) {
            REQUIRE(std::abs(p_expected[i][0][j] - p_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][0][j] - u_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][0][j] - v_host[offset_results + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][1][j] - p_host[offset_results + N_test_elements + 1 + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][1][j] - u_host[offset_results + N_test_elements + 1 + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][1][j] - v_host[offset_results + N_test_elements + 1 + j]) < max_error);
        }
    }

    elements.clear(stream);
    faces.clear(stream);
    nodes.clear(stream);
    p.clear(stream);
    u.clear(stream);
    v.clear(stream);

    cudaStreamDestroy(stream);
}

TEST_CASE("Faces to element projection test", "Projects the face flux solution of two faces to their elements and checks the values match.") {   
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = N_max;
    constexpr size_t n_faces = 2;
    constexpr size_t n_elements = n_faces;
    const double max_error = 1e-9;

    REQUIRE(N_test <= N_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream); 

    const std::vector<Vec2<deviceFloat>> host_nodes {Vec2<deviceFloat>{0, -1},
                                                     Vec2<deviceFloat>{0, 0},
                                                     Vec2<deviceFloat>{0, 1},
                                                     Vec2<deviceFloat>{-1, 1},
                                                     Vec2<deviceFloat>{-1, -1},
                                                     Vec2<deviceFloat>{1, -1},
                                                     Vec2<deviceFloat>{1, 1}};

    SEM::Entities::NDG_t<SEM::Polynomials::LegendrePolynomial_t> NDG(N_max, N_interpolation_points, stream);
    SEM::Entities::device_vector<SEM::Entities::Element2D_t> elements(n_elements, stream);
    SEM::Entities::device_vector<SEM::Entities::Face2D_t> faces(n_faces, stream);
    SEM::Entities::device_vector<Vec2<deviceFloat>> nodes(host_nodes, stream);

    constexpr int faces_blockSize = 32;
    constexpr int faces_numBlocks = (n_faces + faces_blockSize - 1) / faces_blockSize;
    constexpr int elements_blockSize = 32;
    constexpr int elements_numBlocks = (n_elements + elements_blockSize - 1) / elements_blockSize;

    faces_to_element_projection_init<<<faces_blockSize, (faces_numBlocks + 1)/2, 0, stream>>>(N_test, n_faces, elements.data(), faces.data(), nodes.data(), NDG.nodes_.data());

    SEM::Meshes::compute_element_geometry<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, elements.data(), nodes.data(), NDG.nodes_.data());

    SEM::Meshes::project_to_elements<<<elements_blockSize, elements_numBlocks, 0, stream>>>(n_elements, faces.data(), elements.data(), NDG.nodes_.data(), NDG.weights_.data(), NDG.barycentric_weights_.data());
    
    std::vector<deviceFloat> polynomial_nodes_host(NDG.nodes_.size());

    NDG.nodes_.copy_to(polynomial_nodes_host, stream);

    std::vector<std::array<std::vector<deviceFloat>, 2>> p_expected(n_faces/2);
    std::vector<std::array<std::vector<deviceFloat>, 2>> u_expected(n_faces/2);
    std::vector<std::array<std::vector<deviceFloat>, 2>> v_expected(n_faces/2);
    
    cudaStreamSynchronize(stream);
    constexpr size_t offset_1D = N_test * (N_test + 1) /2;
    for (size_t i = 0; i < n_faces/2; ++i) {
        p_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
        u_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
        v_expected[i] = {std::vector<deviceFloat>(N_test + 1), std::vector<deviceFloat>(N_test + 1)};
    
        for (int j = 0; j <= N_test; ++j) {
            const deviceFloat interp = (polynomial_nodes_host[offset_1D + j] + 1)/2;
            const Vec2<deviceFloat> global_coordinates = host_nodes[8 * i + 2] * interp + host_nodes[8 * i] * (1 - interp);
            
            p_expected[i][0][j] = std::cos(global_coordinates.y());
            u_expected[i][0][j] = global_coordinates.y() * global_coordinates.y();
            v_expected[i][0][j] = global_coordinates.y();

            p_expected[i][1][N_test - j] = -std::cos(global_coordinates.y());
            u_expected[i][1][N_test - j] = -global_coordinates.y() * global_coordinates.y();
            v_expected[i][1][N_test - j] = -global_coordinates.y();
        }
    }

    SEM::Entities::device_vector<deviceFloat> p(n_faces * (N_test + 1), stream);
    SEM::Entities::device_vector<deviceFloat> u(n_faces * (N_test + 1), stream);
    SEM::Entities::device_vector<deviceFloat> v(n_faces * (N_test + 1), stream);

    retrieve_element_projected_solution_2<<<faces_numBlocks, (faces_blockSize + 1)/2, 0, stream>>>(N_test, n_faces, elements.data(), faces.data(), p.data(), u.data(), v.data());

    std::vector<deviceFloat> p_host(n_faces * (N_test + 1));
    std::vector<deviceFloat> u_host(n_faces * (N_test + 1));
    std::vector<deviceFloat> v_host(n_faces * (N_test + 1));

    p.copy_to(p_host, stream);
    u.copy_to(u_host, stream);
    v.copy_to(v_host, stream);
    cudaStreamSynchronize(stream);
   
    for (size_t i = 0; i < n_faces/2; ++i) {
        const size_t offset_results = 2 * i * (N_test + 1);

        for (int j = 0; j <= N_test; ++j) {
            REQUIRE(std::abs(p_expected[i][0][j] - p_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][0][j] - u_host[offset_results + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][0][j] - v_host[offset_results + j]) < max_error);

            REQUIRE(std::abs(p_expected[i][1][j] - p_host[offset_results + N_test + 1 + j]) < max_error);
            REQUIRE(std::abs(u_expected[i][1][j] - u_host[offset_results + N_test + 1 + j]) < max_error);
            REQUIRE(std::abs(v_expected[i][1][j] - v_host[offset_results + N_test + 1 + j]) < max_error);
        }
    }

    elements.clear(stream);
    faces.clear(stream);
    nodes.clear(stream);
    p.clear(stream);
    u.clear(stream);
    v.clear(stream);

    cudaStreamDestroy(stream);
}