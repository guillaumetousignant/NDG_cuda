#include <catch2/catch.hpp>
#include <iostream>

TEST_CASE( "stupid/1=1", "Prove that one equals 1" ){
    int one = 1;
    REQUIRE( one == 1 );
}