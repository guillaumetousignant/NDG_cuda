#ifndef NDG_ENTITIES_CUDA_VECTOR_CUH
#define NDG_ENTITIES_CUDA_VECTOR_CUH

namespace SEM { namespace Device { namespace Entities {
    template<typename T>
    class cuda_vector { 
        public: 
            template<typename TI>
            __device__
            cuda_vector(TI size);

            __device__ __host__
            cuda_vector();

            __device__
            ~cuda_vector();

            __device__
            cuda_vector(const cuda_vector<T>& other); // copy constructor

            __device__
            cuda_vector(cuda_vector<T>&& other) noexcept; // move constructor

            __device__
            auto operator=(const cuda_vector<T>& other) -> cuda_vector&; // copy assignment

            __device__
            auto operator=(cuda_vector<T>&& other) noexcept -> cuda_vector&; // move assignment

            T* data_;
            size_t size_;

            template<typename TI>
            __device__
            auto operator[](TI index) -> T&;

            template<typename TI>
            __device__
            auto operator[](TI index) const -> const T&;

            __device__
            auto size() const -> size_t;

            __device__
            auto data() -> T*;

            __device__
            auto data() const -> const T*;

            __device__
            auto clear() -> void;

            __device__
            auto empty() const -> bool;
    };
}}}

#include "entities/cuda_vector.tcu"

#endif