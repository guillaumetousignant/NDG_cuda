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

add_subdirectory(src)

option(BUILD_TESTING "Build all tests." OFF)
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
