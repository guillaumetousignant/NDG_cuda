if (BUILD_CPU)
    target_sources(SEM
        PRIVATE
            ChebyshevPolynomial_t.cpp
            LegendrePolynomial_t.cpp
    )
endif()
if (BUILD_GPU)
    target_sources(SEM-CUDA
        PRIVATE
            ChebyshevPolynomial_t.cu
            LegendrePolynomial_t.cu
    )
endif()