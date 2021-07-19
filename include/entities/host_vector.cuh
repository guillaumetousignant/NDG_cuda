#ifndef NDG_HOST_VECTOR_H
#define NDG_HOST_VECTOR_H

namespace SEM { namespace Entities {
    template<typename T>
    class host_vector { 
        public: 
            template<typename TI>
            host_vector(TI size);

            host_vector();

           template < class U = T,
                    class std::enable_if<!std::is_trivially_destructible<U>::value, int>::type = 0>
            auto destroy() -> void {
                for (size_t i = 0; i < size_; ++i) {
                    data_[i].~T();
                }
            }

            template<class U = T,
                    class std::enable_if<std::is_trivially_destructible<U>::value, int>::type = 0>
            auto destroy() -> void {}

            ~host_vector();

            host_vector(const host_vector<T>& other); // copy constructor

            host_vector(host_vector<T>&& other) noexcept; // move constructor

            auto operator=(const host_vector<T>& other) -> host_vector&; // copy assignment

            auto operator=(host_vector<T>&& other) noexcept -> host_vector&; // move assignment

            T* data_;
            size_t size_;

            template<typename TI>
            auto operator[](TI index) -> T&;

            template<typename TI>
            auto operator[](TI index) const -> const T&;

            auto size() const -> size_t;

            auto data() -> T*;

            auto data() const -> const T*;

            auto clear() -> void;

            auto empty() const -> bool;
    };
}}

#include "entities/host_vector.tcu"

#endif