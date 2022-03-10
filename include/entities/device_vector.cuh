#ifndef NDG_ENTITIES_DEVICE_VECTOR_CUH
#define NDG_ENTITIES_DEVICE_VECTOR_CUH

#include <vector>
#include <memory>
#include "entities/host_vector.cuh"
#include "entities/Element2D_t.cuh"
#include "entities/Face2D_t.cuh"

namespace SEM { namespace Device { namespace Entities {
    template<typename T>
    __global__
    auto empty_device_vector(size_t size, T* data) -> void;

    template __global__ auto empty_device_vector<SEM::Device::Entities::Element2D_t>(size_t size, SEM::Device::Entities::Element2D_t* data) -> void; // AAAAH why is this needed, for some reason it won't 
    template __global__ auto empty_device_vector<SEM::Device::Entities::Face2D_t>(size_t size, SEM::Device::Entities::Face2D_t* data) -> void;       // automatically instantiate this when instantiating device vectors??

    template<typename T>
    class device_vector { 
        public: 
            template<typename TI>
            __host__ __device__
            device_vector(TI size);

            template<typename TI>
            __host__
            device_vector(TI size, const cudaStream_t& stream);

            __host__ __device__
            device_vector();

           template < class U = T,
                    class std::enable_if<!std::is_trivially_destructible<U>::value, int>::type = 0>
            __host__ __device__
            auto destroy() -> void {
                #ifdef  __CUDA_ARCH__
                for (size_t i = 0; i < size_; ++i) {
                    data_[i].~T();
                }
                #else
                const int numBlocks = (size_ + blockSize_ - 1) / blockSize_;
                SEM::Device::Entities::empty_device_vector<T><<<numBlocks, blockSize_, 0>>>(size_, data_);
                #endif
            }

            template<class U = T,
                    class std::enable_if<std::is_trivially_destructible<U>::value, int>::type = 0>
            __host__ __device__
            auto destroy() -> void {}

            template < class U = T,
                    class std::enable_if<!std::is_trivially_destructible<U>::value, int>::type = 0>
            __host__ __device__
            auto destroy(const cudaStream_t& stream) -> void {
                #ifdef  __CUDA_ARCH__
                for (size_t i = 0; i < size_; ++i) {
                    data_[i].~T();
                }
                #else
                const int numBlocks = (size_ + blockSize_ - 1) / blockSize_;
                SEM::Device::Entities::empty_device_vector<T><<<numBlocks, blockSize_, 0, stream>>>(size_, data_);
                #endif
            }

            template<class U = T,
                    class std::enable_if<std::is_trivially_destructible<U>::value, int>::type = 0>
            __host__ __device__
            auto destroy(const cudaStream_t& stream) -> void {}

            __host__ __device__
            ~device_vector();

            __host__ __device__
            device_vector(const device_vector<T>& other); // copy constructor

            __host__
            device_vector(const device_vector<T>& other, const cudaStream_t& stream); // copy constructor
            
            __host__ 
            device_vector(const std::vector<T>& other); // copy constructor
            
            __host__ 
            device_vector(const std::vector<T>& other, const cudaStream_t& stream); // copy constructor

            __host__ 
            device_vector(const host_vector<T>& other); // copy constructor
            
            __host__ 
            device_vector(const host_vector<T>& other, const cudaStream_t& stream); // copy constructor

            __host__ __device__
            device_vector(device_vector<T>&& other) noexcept; // move constructor

            __host__ __device__
            auto operator=(const device_vector<T>& other) -> device_vector&; // copy assignment

            __host__
            auto operator=(const std::vector<T>& other) -> device_vector&; // copy assignment

            __host__
            auto operator=(const host_vector<T>& other) -> device_vector&; // copy assignment

            __host__ __device__
            auto operator=(device_vector<T>&& other) noexcept -> device_vector&; // move assignment

            T* data_;
            size_t size_;
            constexpr static int blockSize_ = 32;

            template<typename TI>
            __device__
            auto operator[](TI index) -> T&;

            template<typename TI>
            __device__
            auto operator[](TI index) const -> const T&;

            __host__ __device__
            auto size() const -> size_t;

            __host__ __device__
            auto data() -> T*;

            __host__ __device__
            auto data() const -> const T*;

            __host__
            auto copy_from(const std::vector<T>& host_vector) -> void;

            __host__
            auto copy_from(const std::vector<T>& host_vector, const cudaStream_t& stream) -> void;

            __host__
            auto copy_to(std::vector<T>& host_vector) const -> void;

            __host__
            auto copy_to(std::vector<T>& host_vector, const cudaStream_t& stream) const -> void;

            __host__
            auto copy_from(const host_vector<T>& host_vector) -> void;

            __host__
            auto copy_from(const host_vector<T>& host_vector, const cudaStream_t& stream) -> void;

            __host__
            auto copy_to(host_vector<T>& host_vector) const -> void;

            __host__
            auto copy_to(host_vector<T>& host_vector, const cudaStream_t& stream) const -> void;

            __host__
            auto copy_from(const device_vector<T>& device_vector) -> void;

            __host__
            auto copy_from(const device_vector<T>& device_vector, const cudaStream_t& stream) -> void;

            __host__
            auto copy_to(device_vector<T>& device_vector) const -> void;

            __host__
            auto copy_to(device_vector<T>& device_vector, const cudaStream_t& stream) const -> void;

            __host__
            auto copy_from(const std::unique_ptr<T[]>& smart_ptr) -> void;

            __host__
            auto copy_from(const std::unique_ptr<T[]>& smart_ptr, const cudaStream_t& stream) -> void;

            __host__
            auto copy_to(std::unique_ptr<T[]>& smart_ptr) const -> void;

            __host__
            auto copy_to(std::unique_ptr<T[]>& smart_ptr, const cudaStream_t& stream) const -> void;

            __host__ __device__
            auto clear() -> void;

            __host__
            auto clear(const cudaStream_t& stream) -> void;

            __host__ __device__
            auto empty() const -> bool;
    };
}}}

#include "entities/device_vector.tcu"

#endif