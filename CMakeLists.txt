cmake_minimum_required(VERSION 3.18)
# 3.18 enables CUDA architecture flags

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source.")
endif()

project(NDG_cuda VERSION 1.0.0 LANGUAGES CUDA CXX)

#enable_language(CUDA) # Put this somewhere conditionally if CUDA is optional
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(NDG_cuda 
    src/main.cu)

set_target_properties(NDG_cuda PROPERTIES
                            CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET NDG_cuda PROPERTY CUDA_ARCHITECTURES 61) # sm_61 is GTX 10X0

target_include_directories(NDG_cuda 
    PRIVATE
        include/
)

target_compile_features(NDG_cuda PRIVATE cxx_std_17)

option(BUILD_TESTING "Build all tests." OFF)

if (BUILD_TESTING)
    include(CTest)
    find_package(Catch2 REQUIRED)
    include(Catch)

    add_executable(unit_tests 
        tests/unit_tests.cpp)
    target_compile_features(unit_tests PRIVATE cxx_std_17)
    target_link_libraries(unit_tests Catch2::Catch2)

    catch_discover_tests(unit_tests)
endif()

install(TARGETS NDG_cuda DESTINATION bin)