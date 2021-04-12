#set(HDF5_USE_STATIC_LIBRARIES ON)

find_package (HDF5 COMPONENTS C)
find_package(CGNS REQUIRED)

add_executable(mesh_generator main.cpp)

target_link_libraries(mesh_generator PRIVATE 
                        SEM
                        hdf5::hdf5 # Should be hdf5::hdf5, but it doesn't work for some reason
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