target_sources(SEM
    PRIVATE
        DataWriter_t.cpp
        ProgressBar_t.cpp
        InputParser_t.cpp
)
if (BUILD_GPU)
    target_sources(SEM-CUDA
        PRIVATE
            DataWriter_t.cpp
            ProgressBar_t.cpp
            InputParser_t.cpp
    )
endif()