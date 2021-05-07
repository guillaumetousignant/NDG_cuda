#include <catch2/catch.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include "entities/NDG_t.cuh"
#include "helpers/float_types.h"

TEST_CASE("2D interpolation test", "Checks the interpolated value of the solution at the output interpolation points.") {   
    const int N_max = 16;
    const int N_test = 16;
    const size_t N_interpolation_points = std::pow(N_max, 2);
    const double max_error = 1e-6;

    REQUIRE(N_test <= N_max);
}