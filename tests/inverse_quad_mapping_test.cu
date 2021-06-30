#include <catch2/catch.hpp>
#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include "meshes/Mesh2D_t.cuh"
#include "functions/inverse_quad_map.cuh"

using SEM::Entities::Vec2;

TEST_CASE("Inverse quad mapping", "Checks the inverse quad mapping returns the right result.") {
    const double error = 1e-6;

    const std::array<Vec2<deviceFloat>, 4> points {Vec2<deviceFloat>{4.0, 2.0},
                                                   Vec2<deviceFloat>{4.0, 4.0},
                                                   Vec2<deviceFloat>{2.0, 4.0},
                                                   Vec2<deviceFloat>{2.0, 2.0}};

    const std::array<Vec2<deviceFloat>, 25> local_coordinates {Vec2<deviceFloat>{-1,   -1},
                                                               Vec2<deviceFloat>{-0.5, -1},
                                                               Vec2<deviceFloat>{0,    -1},
                                                               Vec2<deviceFloat>{0.5,  -1},
                                                               Vec2<deviceFloat>{1,    -1},
                                                               Vec2<deviceFloat>{-1,   -0.5},
                                                               Vec2<deviceFloat>{-0.5, -0.5},
                                                               Vec2<deviceFloat>{0,    -0.5},
                                                               Vec2<deviceFloat>{0.5,  -0.5},
                                                               Vec2<deviceFloat>{1,    -0.5},
                                                               Vec2<deviceFloat>{-1,   0},
                                                               Vec2<deviceFloat>{-0.5, 0},
                                                               Vec2<deviceFloat>{0,    0},
                                                               Vec2<deviceFloat>{0.5,  0},
                                                               Vec2<deviceFloat>{1,    0},
                                                               Vec2<deviceFloat>{-1,   0.5},
                                                               Vec2<deviceFloat>{-0.5, 0.5},
                                                               Vec2<deviceFloat>{0,    0.5},
                                                               Vec2<deviceFloat>{0.5,  0.5},
                                                               Vec2<deviceFloat>{1,    0.5},
                                                               Vec2<deviceFloat>{-1,   1},
                                                               Vec2<deviceFloat>{-0.5, 1},
                                                               Vec2<deviceFloat>{0,    1},
                                                               Vec2<deviceFloat>{0.5,  1},
                                                               Vec2<deviceFloat>{1,    1}};

    const std::array<Vec2<deviceFloat>, 25> global_coordinates {Vec2<deviceFloat>{4,   2},
                                                                Vec2<deviceFloat>{4,   2.5},
                                                                Vec2<deviceFloat>{4,   3},
                                                                Vec2<deviceFloat>{4,   3.5},
                                                                Vec2<deviceFloat>{4,   4},
                                                                Vec2<deviceFloat>{3.5, 2},
                                                                Vec2<deviceFloat>{3.5, 2.5},
                                                                Vec2<deviceFloat>{3.5, 3},
                                                                Vec2<deviceFloat>{3.5, 3.5},
                                                                Vec2<deviceFloat>{3.5, 4},
                                                                Vec2<deviceFloat>{3,   2},
                                                                Vec2<deviceFloat>{3,   2.5},
                                                                Vec2<deviceFloat>{3,   3},
                                                                Vec2<deviceFloat>{3,   3.5},
                                                                Vec2<deviceFloat>{3,   4},
                                                                Vec2<deviceFloat>{2.5, 2},
                                                                Vec2<deviceFloat>{2.5, 2.5},
                                                                Vec2<deviceFloat>{2.5, 3},
                                                                Vec2<deviceFloat>{2.5, 3.5},
                                                                Vec2<deviceFloat>{2.5, 4},
                                                                Vec2<deviceFloat>{2,   2},
                                                                Vec2<deviceFloat>{2,   2.5},
                                                                Vec2<deviceFloat>{2,   3},
                                                                Vec2<deviceFloat>{2,   3.5},
                                                                Vec2<deviceFloat>{2,   4}};

    std::array<Vec2<deviceFloat>, 25> local_coordinates_computed;

    for (int i = 0; i < local_coordinates_computed.size(); ++i) {
        local_coordinates_computed[i] = SEM::inverse_quad_map(global_coordinates[i], points);
    }

    for (int i = 0; i < local_coordinates_computed.size(); ++i) {
        REQUIRE(std::abs(local_coordinates_computed[i].x() - local_coordinates[i].x()) < error);
        REQUIRE(std::abs(local_coordinates_computed[i].y() - local_coordinates[i].y()) < error);
    }
}

TEST_CASE("Inverse quad mapping parallelogram", "Checks the inverse quad mapping returns the right result on a parallelogram.") {
    const double error = 1e-6;

    const std::array<Vec2<deviceFloat>, 4> points {Vec2<deviceFloat>{0.0, 0.0},
                                                   Vec2<deviceFloat>{1.0, 1.0},
                                                   Vec2<deviceFloat>{1.0, 2.0},
                                                   Vec2<deviceFloat>{0.0, 1.0}};

    const std::array<Vec2<deviceFloat>, 25> local_coordinates {Vec2<deviceFloat>{-1,   -1},
                                                               Vec2<deviceFloat>{-0.5, -1},
                                                               Vec2<deviceFloat>{0,    -1},
                                                               Vec2<deviceFloat>{0.5,  -1},
                                                               Vec2<deviceFloat>{1,    -1},
                                                               Vec2<deviceFloat>{-1,   -0.5},
                                                               Vec2<deviceFloat>{-0.5, -0.5},
                                                               Vec2<deviceFloat>{0,    -0.5},
                                                               Vec2<deviceFloat>{0.5,  -0.5},
                                                               Vec2<deviceFloat>{1,    -0.5},
                                                               Vec2<deviceFloat>{-1,   0},
                                                               Vec2<deviceFloat>{-0.5, 0},
                                                               Vec2<deviceFloat>{0,    0},
                                                               Vec2<deviceFloat>{0.5,  0},
                                                               Vec2<deviceFloat>{1,    0},
                                                               Vec2<deviceFloat>{-1,   0.5},
                                                               Vec2<deviceFloat>{-0.5, 0.5},
                                                               Vec2<deviceFloat>{0,    0.5},
                                                               Vec2<deviceFloat>{0.5,  0.5},
                                                               Vec2<deviceFloat>{1,    0.5},
                                                               Vec2<deviceFloat>{-1,   1},
                                                               Vec2<deviceFloat>{-0.5, 1},
                                                               Vec2<deviceFloat>{0,    1},
                                                               Vec2<deviceFloat>{0.5,  1},
                                                               Vec2<deviceFloat>{1,    1}};

    // Could get those with quad_map, but I'm not sure how to make tests depend on each other.
    const std::array<Vec2<deviceFloat>, 25> global_coordinates {Vec2<deviceFloat>{0,    0},
                                                                Vec2<deviceFloat>{0.25, 0.25},
                                                                Vec2<deviceFloat>{0.5,  0.5},
                                                                Vec2<deviceFloat>{0.75, 0.75},
                                                                Vec2<deviceFloat>{1,    1},
                                                                Vec2<deviceFloat>{0,    0.25},
                                                                Vec2<deviceFloat>{0.25, 0.5},
                                                                Vec2<deviceFloat>{0.5,  0.75},
                                                                Vec2<deviceFloat>{0.75, 1},
                                                                Vec2<deviceFloat>{1,    1.25},
                                                                Vec2<deviceFloat>{0,    0.5},
                                                                Vec2<deviceFloat>{0.25, 0.75},
                                                                Vec2<deviceFloat>{0.5,  1},
                                                                Vec2<deviceFloat>{0.75, 1.25},
                                                                Vec2<deviceFloat>{1,    1.5},
                                                                Vec2<deviceFloat>{0,    0.75},
                                                                Vec2<deviceFloat>{0.25, 1},
                                                                Vec2<deviceFloat>{0.5,  1.25},
                                                                Vec2<deviceFloat>{0.75, 1.5},
                                                                Vec2<deviceFloat>{1,    1.75},
                                                                Vec2<deviceFloat>{0,    1},
                                                                Vec2<deviceFloat>{0.25, 1.25},
                                                                Vec2<deviceFloat>{0.5,  1.5},
                                                                Vec2<deviceFloat>{0.75, 1.75},
                                                                Vec2<deviceFloat>{1,    2}};

    std::array<Vec2<deviceFloat>, 25> local_coordinates_computed;

    for (int i = 0; i < local_coordinates_computed.size(); ++i) {
        local_coordinates_computed[i] = SEM::inverse_quad_map(global_coordinates[i], points);
    }

    for (int i = 0; i < local_coordinates_computed.size(); ++i) {
        REQUIRE(std::abs(local_coordinates_computed[i].x() - local_coordinates[i].x()) < error);
        REQUIRE(std::abs(local_coordinates_computed[i].y() - local_coordinates[i].y()) < error);
    }
}

TEST_CASE("Inverse quad mapping trapezoid", "Checks the inverse quad mapping returns the right result on a trapezoid.") {
    const double error = 1e-6;

    const std::array<Vec2<deviceFloat>, 4> points {Vec2<deviceFloat>{1.0, 2.0},
                                                   Vec2<deviceFloat>{0.0, 1.0},
                                                   Vec2<deviceFloat>{0.0, 0.0},
                                                   Vec2<deviceFloat>{1.0, -1.0}};

    const std::array<Vec2<deviceFloat>, 25> local_coordinates {Vec2<deviceFloat>{-1,   -1},
                                                               Vec2<deviceFloat>{-0.5, -1},
                                                               Vec2<deviceFloat>{0,    -1},
                                                               Vec2<deviceFloat>{0.5,  -1},
                                                               Vec2<deviceFloat>{1,    -1},
                                                               Vec2<deviceFloat>{-1,   -0.5},
                                                               Vec2<deviceFloat>{-0.5, -0.5},
                                                               Vec2<deviceFloat>{0,    -0.5},
                                                               Vec2<deviceFloat>{0.5,  -0.5},
                                                               Vec2<deviceFloat>{1,    -0.5},
                                                               Vec2<deviceFloat>{-1,   0},
                                                               Vec2<deviceFloat>{-0.5, 0},
                                                               Vec2<deviceFloat>{0,    0},
                                                               Vec2<deviceFloat>{0.5,  0},
                                                               Vec2<deviceFloat>{1,    0},
                                                               Vec2<deviceFloat>{-1,   0.5},
                                                               Vec2<deviceFloat>{-0.5, 0.5},
                                                               Vec2<deviceFloat>{0,    0.5},
                                                               Vec2<deviceFloat>{0.5,  0.5},
                                                               Vec2<deviceFloat>{1,    0.5},
                                                               Vec2<deviceFloat>{-1,   1},
                                                               Vec2<deviceFloat>{-0.5, 1},
                                                               Vec2<deviceFloat>{0,    1},
                                                               Vec2<deviceFloat>{0.5,  1},
                                                               Vec2<deviceFloat>{1,    1}};

    // Could get those with quad_map, but I'm not sure how to make tests depend on each other.
    const std::array<Vec2<deviceFloat>, 25> global_coordinates {Vec2<deviceFloat>{1,    2},
                                                                Vec2<deviceFloat>{0.75, 1.75},
                                                                Vec2<deviceFloat>{0.5,  1.5},
                                                                Vec2<deviceFloat>{0.25, 1.25},
                                                                Vec2<deviceFloat>{0,    1},

                                                                Vec2<deviceFloat>{1,    1.25},
                                                                Vec2<deviceFloat>{0.75, 1.125},
                                                                Vec2<deviceFloat>{0.5,  1},
                                                                Vec2<deviceFloat>{0.25, 0.875},
                                                                Vec2<deviceFloat>{0,    0.75},

                                                                Vec2<deviceFloat>{1,    0.5},
                                                                Vec2<deviceFloat>{0.75, 0.5},
                                                                Vec2<deviceFloat>{0.5,  0.5},
                                                                Vec2<deviceFloat>{0.25, 0.5},
                                                                Vec2<deviceFloat>{0,    0.5},

                                                                Vec2<deviceFloat>{1,    -0.25},
                                                                Vec2<deviceFloat>{0.75, -0.125},
                                                                Vec2<deviceFloat>{0.5,  0},
                                                                Vec2<deviceFloat>{0.25, 0.125},
                                                                Vec2<deviceFloat>{0,    0.25},

                                                                Vec2<deviceFloat>{1,    -1},
                                                                Vec2<deviceFloat>{0.75, -0.75},
                                                                Vec2<deviceFloat>{0.5,  -0.5},
                                                                Vec2<deviceFloat>{0.25, -0.25},
                                                                Vec2<deviceFloat>{0,    0}};

    std::array<Vec2<deviceFloat>, 25> local_coordinates_computed;

    for (int i = 0; i < local_coordinates_computed.size(); ++i) {
        local_coordinates_computed[i] = SEM::inverse_quad_map(global_coordinates[i], points);
    }

    for (int i = 0; i < local_coordinates_computed.size(); ++i) {
        REQUIRE(std::abs(local_coordinates_computed[i].x() - local_coordinates[i].x()) < error);
        REQUIRE(std::abs(local_coordinates_computed[i].y() - local_coordinates[i].y()) < error);
    }
}