#include <catch2/catch.hpp>
#include "helpers/float_types.h"
#include "entities/Vec2.cuh"
#include "meshes/Mesh2D_t.cuh"

using SEM::Entities::Vec2;

TEST_CASE("Quad mapping", "Checks the quad mapping returns the right result.") {
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

    const std::array<Vec2<deviceFloat>, 25> global_coordinates {Vec2<deviceFloat>{2,   2},
                                                                Vec2<deviceFloat>{2.5, 2},
                                                                Vec2<deviceFloat>{3,   2},
                                                                Vec2<deviceFloat>{3.5, 2},
                                                                Vec2<deviceFloat>{4,   2},
                                                                Vec2<deviceFloat>{2,   2.5},
                                                                Vec2<deviceFloat>{2.5, 2.5},
                                                                Vec2<deviceFloat>{3,   2.5},
                                                                Vec2<deviceFloat>{3.5, 2.5},
                                                                Vec2<deviceFloat>{4,   2.5},
                                                                Vec2<deviceFloat>{2,   3},
                                                                Vec2<deviceFloat>{2.5, 3},
                                                                Vec2<deviceFloat>{3,   3},
                                                                Vec2<deviceFloat>{3.5, 3},
                                                                Vec2<deviceFloat>{4,   3},
                                                                Vec2<deviceFloat>{2,   3.5},
                                                                Vec2<deviceFloat>{2.5, 3.5},
                                                                Vec2<deviceFloat>{3,   3.5},
                                                                Vec2<deviceFloat>{3.5, 3.5},
                                                                Vec2<deviceFloat>{4,   3.5},
                                                                Vec2<deviceFloat>{2,   4},
                                                                Vec2<deviceFloat>{2.5, 4},
                                                                Vec2<deviceFloat>{3,   4},
                                                                Vec2<deviceFloat>{3.5, 4},
                                                                Vec2<deviceFloat>{4,   4}};

    std::array<Vec2<deviceFloat>, 25> global_coordinates_computed;

    for (int i = 0; i < global_coordinates_computed.size(); ++i) {
        global_coordinates_computed[i] = SEM::Meshes::Mesh2D_t::quad_map(local_coordinates[i], points);
        REQUIRE(std::abs(global_coordinates_computed[i].x() - global_coordinates[i].x()) < error);
        REQUIRE(std::abs(global_coordinates_computed[i].y() - global_coordinates[i].y()) < error);
    }
}