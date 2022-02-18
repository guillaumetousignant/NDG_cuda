if (BUILD_CPU)
    add_library(SEM "")
endif()
if (BUILD_GPU)
    add_library(SEM-CUDA "")
endif()

add_subdirectory(entities)
add_subdirectory(functions)
add_subdirectory(helpers)
add_subdirectory(meshes)
add_subdirectory(polynomials)
add_subdirectory(solvers)

# Dependencies
# MPI stuff
find_package(MPI REQUIRED CXX)

# VTK stuff
find_package(VTK COMPONENTS 
                    CommonCore
                    FiltersCore
                    FiltersSources
                    IOParallelXML
                    ParallelMPI)

# CGNS stuff
if (USE_CGNS)
    set(HDF5_USE_STATIC_LIBRARIES TRUE)
    set(HDF5_PREFER_PARALLEL TRUE)

    find_package(HDF5 REQUIRED COMPONENTS C)
    find_package(CGNS REQUIRED)

    set(HDF5_DEPENDENCY "find_dependency(HDF5 REQUIRED COMPONENTS C)")
    set(CGNS_DEPENDENCY "find_dependency(CGNS REQUIRED)")
endif()

# Targets
if (BUILD_CPU)
    target_include_directories(SEM PUBLIC 
        $<INSTALL_INTERFACE:include/SEM/>    
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>)

    target_compile_features(SEM PRIVATE cxx_std_17)
    if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
        set_property(TARGET SEM PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    endif()

    target_link_libraries(SEM PUBLIC 
                            MPI::MPI_CXX
                            ${VTK_LIBRARIES})
    
    if (USE_CGNS)
        target_link_libraries(SEM PUBLIC 
                                CGNS::CGNS
                                hdf5::hdf5-static)
        target_compile_definitions(SEM PUBLIC NDG_USE_CGNS)
    endif()

    vtk_module_autoinit(TARGETS SEM
                            MODULES ${VTK_LIBRARIES})

    # Installation stuff
    install(TARGETS SEM 
        EXPORT SEM-targets
        DESTINATION lib)

    install(EXPORT SEM-targets
        FILE
            SEMTargets.cmake
        NAMESPACE
            SEM::
        DESTINATION
            lib/cmake/SEM
    )

    #Create a ConfigVersion.cmake file
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/SEMConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/SEMConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/SEMConfig.cmake
        INSTALL_DESTINATION lib/cmake/SEM
    )

    #Install the config, configversion and custom find modules
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SEMConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/SEMConfigVersion.cmake
        DESTINATION lib/cmake/SEM
    )

    export(EXPORT SEM-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/SEMTargets.cmake
        NAMESPACE SEM::
    )

    #Register package in user's package registry
    export(PACKAGE SEM)

    add_executable(NDG_host 
        main.cpp)

    target_link_libraries(NDG_host PRIVATE SEM)
    target_compile_features(NDG_host PRIVATE cxx_std_17)
    if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
        set_property(TARGET NDG_host PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    endif()

    install(TARGETS NDG_host DESTINATION bin)
endif()

if (BUILD_GPU)
    target_include_directories(SEM-CUDA PUBLIC 
        $<INSTALL_INTERFACE:include/SEM/>    
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>)

    # Cuda stuff
    set_target_properties(SEM-CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_features(SEM-CUDA PRIVATE cxx_std_17)
    if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
        set_property(TARGET SEM PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
        set_property(TARGET SEM-CUDA PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    endif()
    target_compile_options(SEM-CUDA PUBLIC "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:DEBUG>,$<CONFIG:RelWithDebInfo>>>:-lineinfo>")
    target_compile_options(SEM-CUDA PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

    target_link_libraries(SEM-CUDA PUBLIC 
                            MPI::MPI_CXX
                            ${VTK_LIBRARIES})

    if (USE_CGNS)
        target_link_libraries(SEM-CUDA PUBLIC 
                                CGNS::CGNS
                                hdf5::hdf5-static)
        target_compile_definitions(SEM-CUDA PUBLIC NDG_USE_CGNS)
    endif()
                            
    vtk_module_autoinit(TARGETS SEM-CUDA
                            MODULES ${VTK_LIBRARIES})

    # Installation stuff
    install(TARGETS SEM-CUDA 
        EXPORT SEM-CUDA-targets
        DESTINATION lib)

    install(EXPORT SEM-CUDA-targets
        FILE
            SEM-CUDATargets.cmake
        NAMESPACE
            SEM-CUDA::
        DESTINATION
            lib/cmake/SEM
    )

    #Create a ConfigVersion.cmake file
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/SEM-CUDAConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/SEM-CUDAConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/SEM-CUDAConfig.cmake
        INSTALL_DESTINATION lib/cmake/SEM
    )

    #Install the config, configversion and custom find modules
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SEM-CUDAConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/SEM-CUDAConfigVersion.cmake
        DESTINATION lib/cmake/SEM
    )

    export(EXPORT SEM-CUDA-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/SEM-CUDATargets.cmake
        NAMESPACE SEM-CUDA::
    )

    #Register package in user's package registry
    export(PACKAGE SEM-CUDA)

    add_executable(NDG 
        main.cu)

    target_link_libraries(NDG PRIVATE SEM-CUDA)
    set_target_properties(NDG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_features(NDG PRIVATE cxx_std_17)
    if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
        set_property(TARGET NDG PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    endif()
    
    install(TARGETS NDG DESTINATION bin)
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_INSTALL_PREFIX)
    if (BUILD_GPU)
        install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
            DESTINATION include/SEM
            FILES_MATCHING PATTERN "*.cuh")
        install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
            DESTINATION include/SEM
            FILES_MATCHING PATTERN "*.tcu")
    endif()
    
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/SEM
        FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/SEM
        FILES_MATCHING PATTERN "*.tpp")
endif()

if(WIN32)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/
        DESTINATION bin
        FILES_MATCHING PATTERN "*.dll")
endif()
