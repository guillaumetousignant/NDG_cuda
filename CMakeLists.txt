cmake_minimum_required(VERSION 3.18)
# 3.18 enables CUDA architecture flags

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source.")
endif()

option(BUILD_GPU "Build GPU version of the targets." ON)
option(BUILD_CPU "Build CPU version of the targets." ON)

if (BUILD_GPU)
    project(NDG_cuda VERSION 1.0.0 LANGUAGES CUDA CXX C)
else ()
    project(NDG_cuda VERSION 1.0.0 LANGUAGES CXX C)
endif()

#enable_language(CUDA) # Put this somewhere conditionally if CUDA is optional
if (BUILD_GPU)
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_subdirectory(src)

option(BUILD_DOC "Build documentation." OFF)
if (BUILD_DOC)
    add_subdirectory(docs)
endif()

option(BUILD_EXTRAS "Build extras, such as meshing tools." OFF)
if (BUILD_EXTRAS)
    add_subdirectory(extras)
endif()

option(BUILD_TESTING "Build all tests." OFF)
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
