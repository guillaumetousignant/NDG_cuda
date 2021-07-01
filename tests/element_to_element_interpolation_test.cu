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
auto element_to_element_projection_init(int N, size_t n_elements, SEM::Entities::Element2D_t* elements, SEM::Entities::Face2D_t* faces, const deviceFloat* NDG_nodes) -> void {

}

__global__
auto retrieve_element_to_element_projected_solution(int N, size_t n_elements, const SEM::Entities::Element2D_t* elements, const SEM::Entities::Face2D_t* faces, deviceFloat* p, deviceFloat* u, deviceFloat* v) -> void {

}

TEST_CASE("Element to element projection test", "Projects the solution from one element to another and checks the values match.") {   

}