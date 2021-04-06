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
            device_vector& operator=(const device_vector<T>& other); // copy assignment

            __host__ __device__
            device_vector& operator=(device_vector<T>&& other) noexcept; // move assignment

            T* data_;
            size_t size_;

            __device__
            T& operator[](size_t index);

            __device__
            const T& operator[](size_t index) const;

            __host__ __device__
            size_t size() const;

            __host__ __device__
            T* data();

            __host__
            void copy_from(const std::vector<T>& host_vector);

            __host__
            void copy_to(std::vector<T>& host_vector) const;

            __host__
            void copy_from(const device_vector<T>& device_vector);

            __host__
            void copy_to(device_vector<T>& device_vector) const;
    };
}

#include "device_vector.tcu"

#endif