if (BUILD_CPU)
    target_sources(SEM
        PRIVATE
            Hilbert.cpp
            Hilbert_splitting.cpp
            Utilities.cpp
    )
endif()
if (BUILD_GPU)
    target_sources(SEM-CUDA
        PRIVATE
            Hilbert.cpp
            Hilbert_splitting.cu
            Utilities.cpp
    )
endif()