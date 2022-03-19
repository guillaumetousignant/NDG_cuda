target_sources(SEM
    PRIVATE
        Element_t.cpp
        Element2D_t.cpp
        Face_t.cpp
        Face2D_t.cpp
)
if (BUILD_GPU)
    target_sources(SEM-CUDA
        PRIVATE
            Element_t.cu
            Element2D_t.cu
            Face_t.cu
            Face2D_t.cu
            NDG_t.cu
    )
endif()