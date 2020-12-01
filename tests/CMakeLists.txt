include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.3
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

include(CTest)
include(Catch)

add_executable(unit_tests 
    test_main.cpp
    example_test.cpp)
target_link_libraries(unit_tests PRIVATE 
    Catch2::Catch2)
target_compile_features(unit_tests PRIVATE cxx_std_17)

catch_discover_tests(unit_tests)