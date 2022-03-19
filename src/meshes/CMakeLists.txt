target_sources(SEM
    PRIVATE
        Mesh_t.cpp
        Mesh2D_t.cpp
)
if (BUILD_GPU)
    target_sources(SEM-CUDA
        PRIVATE
            Mesh_t.cu
            Mesh2D_t.cu
    )
endif()