find_package(CGNS REQUIRED)

add_executable(mesh_generator main.cpp)

target_link_libraries(mesh_generator PRIVATE 
                        SEM
                        CGNS::CGNS)
target_compile_features(mesh_generator PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET mesh_generator PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()
if (OPTIMIZE_FOR_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(mesh_generator PUBLIC -march=native)
    endif()
endif()

install(TARGETS mesh_generator DESTINATION bin)