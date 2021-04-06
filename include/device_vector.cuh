#ifndef NDG_DEVICE_VECTOR_H
#define NDG_DEVICE_VECTOR_H

#include <vector>

namespace SEM {
    template<typename T>
    class device_vector { 
        public: 
            __host__ __device__
            device_vector(size_t size);

            __host__ __device__
            device_vector();

            __host__ __device__
            ~device_vector();

            __host__ __device__
            device_vector(const device_vector<T>& other); // copy constructor

            __host__ 
            device_vector(const std::vector<T>& other); // copy constructor

            __host__ __device__
            device_vector(device_vector<T>&& other) noexcept; // move constructor

            __host__ __device__
            auto operator=(const device_vector<T>& other) -> device_vector&; // copy assignment

            __host__ __device__
            auto operator=(device_vector<T>&& other) noexcept -> device_vector&; // move assignment

            T* data_;
            size_t size_;

            __device__
            auto operator[](size_t index) -> T&;

            __device__
            auto operator[](size_t index) const -> const T&;

            __host__ __device__
            auto size() const -> size_t;

            __host__ __device__
            auto data() -> T*;

            __host__ __device__
            auto data() const -> const T*;

            __host__
            auto copy_from(const std::vector<T>& host_vector) -> void;

            __host__
            auto copy_to(std::vector<T>& host_vector) const -> void;

            __host__
            auto copy_from(const device_vector<T>& device_vector) -> void;

            __host__
            auto copy_to(device_vector<T>& device_vector) const -> void;
    };
}

#include "device_vector.tcu"

#endif