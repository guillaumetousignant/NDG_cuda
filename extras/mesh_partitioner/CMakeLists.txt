add_executable(mesh_partitioner main.cpp)

target_link_libraries(mesh_partitioner PRIVATE SEM)
target_compile_features(mesh_partitioner PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET mesh_partitioner PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()
if (OPTIMIZE_FOR_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(mesh_partitioner PUBLIC -march=native)
    endif()
endif()

install(TARGETS mesh_partitioner DESTINATION bin)