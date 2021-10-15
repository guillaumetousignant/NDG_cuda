add_library(SEM "")
add_library(SEM-CUDA "")

add_subdirectory(entities)
add_subdirectory(functions)
add_subdirectory(helpers)
add_subdirectory(meshes)
add_subdirectory(polynomials)
add_subdirectory(solvers)

target_include_directories(SEM PUBLIC 
    $<INSTALL_INTERFACE:include/SEM/>    
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>)

target_include_directories(SEM-CUDA PUBLIC 
    $<INSTALL_INTERFACE:include/SEM/>    
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>)

# Cuda stuff
set_target_properties(SEM-CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(SEM-CUDA PROPERTIES CUDA_ARCHITECTURES "60;61;70") # sm_61 is GTX 10X0, sm_60 is Tesla P100, sm_70 is Tesla V100
target_compile_features(SEM PRIVATE cxx_std_17)
target_compile_features(SEM-CUDA PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET SEM PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    set_property(TARGET SEM-CUDA PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()
target_compile_options(SEM-CUDA PUBLIC "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-lineinfo>")
target_compile_options(SEM-CUDA PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# MPI stuff
find_package(MPI REQUIRED CXX)
target_link_libraries(SEM PUBLIC MPI::MPI_CXX)
target_link_libraries(SEM-CUDA PUBLIC MPI::MPI_CXX)

# CGNS stuff
set(HDF5_USE_STATIC_LIBRARIES TRUE)
set(HDF5_PREFER_PARALLEL TRUE)

find_package(HDF5 COMPONENTS C)
find_package(CGNS REQUIRED)

target_link_libraries(SEM PUBLIC 
                        hdf5::hdf5-static
                        CGNS::CGNS)
target_link_libraries(SEM-CUDA PUBLIC 
                        hdf5::hdf5-static
                        CGNS::CGNS)

# VTK stuff
find_package(VTK COMPONENTS 
                    CommonCore
                    FiltersCore
                    FiltersSources
                    IOParallelXML
                    ParallelMPI)

target_link_libraries(SEM PUBLIC 
                        ${VTK_LIBRARIES})
target_link_libraries(SEM-CUDA PUBLIC 
                        ${VTK_LIBRARIES})

vtk_module_autoinit(TARGETS SEM
                        MODULES ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS SEM-CUDA
                        MODULES ${VTK_LIBRARIES})

# Installation stuff
install(TARGETS SEM 
    EXPORT SEM-export
    DESTINATION lib)
install(TARGETS SEM-CUDA 
    EXPORT SEM-CUDA-export
    DESTINATION lib)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_INSTALL_PREFIX)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/SEM
        FILES_MATCHING PATTERN "*.cuh")
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/SEM
        FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/SEM
        FILES_MATCHING PATTERN "*.tcu")
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ 
        DESTINATION include/SEM
        FILES_MATCHING PATTERN "*.tpp")
endif()

install(EXPORT SEM-export
  FILE
    NDGConfig.cmake
  NAMESPACE
  SEM::
  DESTINATION
    lib/cmake/SEM
)
install(EXPORT SEM-CUDA-export
  FILE
    NDG_cudaConfig.cmake
  NAMESPACE
  SEM-CUDA::
  DESTINATION
    lib/cmake/SEM
)

add_executable(NDG 
    main.cu)

target_link_libraries(NDG PRIVATE SEM-CUDA)
set_target_properties(NDG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET NDG PROPERTY CUDA_ARCHITECTURES "60;61;70") # sm_61 is GTX 10X0
target_compile_features(NDG PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET NDG PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

install(TARGETS NDG DESTINATION bin)

add_executable(NDG_host 
    main.cpp)

target_link_libraries(NDG_host PRIVATE SEM)
set_property(TARGET NDG_host PROPERTY CUDA_ARCHITECTURES "60;61;70") # sm_61 is GTX 10X0
target_compile_features(NDG_host PRIVATE cxx_std_17)
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    set_property(TARGET NDG_host PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

install(TARGETS NDG_host DESTINATION bin)
