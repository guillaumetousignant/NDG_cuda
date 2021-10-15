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
    2D_boundary_interpolation_test.cu
    2D_interpolation_test.cu
    polynomials_test.cu
    reduce_test.cu
    initial_conditions_test.cu
    cpu_gpu_test.cu
    quad_mapping_test.cu
    inverse_quad_mapping_test.cu
    element_to_element_interpolation_test.cu
    face_to_element_projection_test.cu
    element_to_face_projection_test.cu)
    
target_link_libraries(unit_tests PRIVATE 
    Catch2::Catch2
    SEM-CUDA
    SEM)

set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET unit_tests PROPERTY CUDA_ARCHITECTURES "60;61;70") # sm_61 is GTX 10X0
target_compile_features(unit_tests PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET unit_tests PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()
target_compile_options(unit_tests PUBLIC "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-lineinfo>")
target_compile_options(unit_tests PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

catch_discover_tests(unit_tests)