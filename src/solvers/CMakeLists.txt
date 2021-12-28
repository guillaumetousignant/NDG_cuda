if (BUILD_CPU)
    target_sources(SEM
        PRIVATE
            Solver2D_t.cpp
    )
endif()
if (BUILD_GPU)
    target_sources(SEM-CUDA
        PRIVATE
            Solver2D_t.cu
    )
endif()