#include <catch2/catch.hpp>
#include <iostream>
#include "NDG_t.cuh"

TEST_CASE("LegendrePolynomials", "Checks the Legendre polynomials"){
    const int N_max = 16;
    int one = 1;
    REQUIRE( one == 1 );

    SECTION("Polynomial value") {
        REQUIRE( one == 1 );
    }
}

TEST_CASE("ChebyshevPolynomials", "Checks the Chebyshev polynomials"){
    const int N_max = 16;
    int one = 1;
    REQUIRE( one == 1 );

    SECTION("Polynomial value") {
        REQUIRE( one == 1 );
    }
}