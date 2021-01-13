add_library(NDG_cuda
    NDG_t.cu
    Element_t.cu
    Face_t.cu
    Mesh_t.cu
    ChebyshevPolynomial_t.cu
    LegendrePolynomial_t.cu
    ChebyshevPolynomial_host_t.cpp
    LegendrePolynomial_host_t.cpp
    NDG_host_t.cpp
    Mesh_host_t.cpp
    Element_host_t.cpp
    Face_host_t.cpp)

target_include_directories(NDG_cuda PUBLIC 
    $<INSTALL_INTERFACE:include/NDG_cuda/>    
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>)

set_target_properties(NDG_cuda PROPERTIES
                            CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET NDG_cuda PROPERTY CUDA_ARCHITECTURES 61) # sm_61 is GTX 10X0
target_compile_features(NDG_cuda PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET NDG_cuda PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

install(TARGETS NDG_cuda 
    EXPORT NDG_cuda-export
    DESTINATION lib)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_INSTALL_PREFIX)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/NDG_cuda
        FILES_MATCHING PATTERN "*.cuh")
endif()

install(EXPORT NDG_cuda-export
  FILE
    NDG_cudaConfig.cmake
  NAMESPACE
    NDG_cuda::
  DESTINATION
    lib/cmake/NDG_cuda
)

add_executable(NDG 
    main.cu)

target_link_libraries(NDG PRIVATE NDG_cuda)
set_target_properties(NDG PROPERTIES
                            CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET NDG PROPERTY CUDA_ARCHITECTURES 61) # sm_61 is GTX 10X0
target_compile_features(NDG PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET NDG PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

install(TARGETS NDG DESTINATION bin)

add_executable(NDG_host 
    main.cpp)

target_link_libraries(NDG_host PRIVATE NDG_cuda)
set_target_properties(NDG_host PROPERTIES
                            CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET NDG_host PROPERTY CUDA_ARCHITECTURES 61) # sm_61 is GTX 10X0
target_compile_features(NDG_host PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET NDG_host PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

install(TARGETS NDG_host DESTINATION bin)
