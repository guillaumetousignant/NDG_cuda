add_executable(NDG_cuda 
    main.cu
    NDG_t.cu
    Element_t.cu
    Face_t.cu
    Mesh_t.cu)

set_target_properties(NDG_cuda PROPERTIES
                            CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET NDG_cuda PROPERTY CUDA_ARCHITECTURES 61) # sm_61 is GTX 10X0

target_include_directories(NDG_cuda PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
)

target_compile_features(NDG_cuda PRIVATE cxx_std_17)

install(TARGETS NDG_cuda DESTINATION bin)
