#ifndef SEM_VEC2_H
#define SEM_VEC2_H

#include <iostream>
#include <array>

namespace SEM { namespace Entities {

    /**
     * @brief The Vec2 class represents a 2-element vector.
     * 
     * This can be used for 2D coordinates, 2D directions, 2-component colours, etc.
     * Several arithmetical operations are defined for those vectors.
     */
    template<typename T>
    class Vec2 {
        public:
            std::array<T, 2> v_; /**< @brief Array of the 2 values in the vector.*/
        
            /**
             * @brief Construct a new Vec2 object with (0, 0).
             */
            __host__ __device__
            constexpr Vec2();

            /**
             * @brief Construct a new Vec2 object from 2 components.
             * 
             * @param x First component of the vector.
             * @param y Second component of the vector.
             */
            template <class T2>
            __host__ __device__
            constexpr Vec2(T x, T2 y); 

            /**
             * @brief Construct a new Vec2 object from one value.
             * 
             * @param x Value given to the two components of the vector.
             */
             __host__ __device__
             constexpr explicit Vec2(T x); 

            /**
             * @brief Accesses the selected component of the vector, returning a reference.
             * 
             * @param index Index of the component to access.
             * @return T& Reference to the selected component.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator[](T2 index) -> T&;

            /**
             * @brief Accesses the selected component of the vector, returning a const reference.
             * 
             * @param index Index of the component to access.
             * @return T& Reference to the selected component.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator[](T2 index) const -> const T&;

            /**
             * @brief Returns the selected component of the vector.
             * 
             * @param index Index of the component to return.
             * @return T Selected component.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator[](T2 index) const -> T; 

            /**
             * @brief Multiplies all components of the vector by a factor.
             * 
             * Returns (x1*a, y1*a).
             * 
             * @param scale Factor used to multiply all components of the vector.
             * @return Vec2 Resulting vector, (x1*a, y1*a).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator*(T2 scale) const -> Vec2<T>;

            /**
             * @brief Element-wise multiplication of two vectors.
             * 
             * Returns (x1*x2, y1*y2).
             * 
             * @param other Vector used to multiply.
             * @return Vec2 Resulting vector, (x1*x2, y1*y2).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator*(const Vec2<T2> &other) const -> Vec2<T>;

            /**
             * @brief Divides all components of the vector by a factor.
             * 
             * Returns (x1/a, y1/a).
             * 
             * @param scale Factor used to divide all components of the vector.
             * @return Vec2 Resulting vector, (x1/a, y1/a).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator/(T2 scale) const -> Vec2<T>;

            /**
             * @brief Elements-wise division by the provided vector.
             * 
             * Returns (x1/x2, y1/y2).
             * 
             * @param other Vector used to divide the components of this vector.
             * @return Vec2 Resulting vector, (x1/x2, y1/y2).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator/(const Vec2<T2> &other) const -> Vec2<T>;

            /**
             * @brief Adds two vectors.
             * 
             * Returns (x1+x2, y1+y2).
             * 
             * @param other Vector added to this vector.
             * @return Vec2 Resulting vector, (x1+x2, y1+y2).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator+(const Vec2<T2> &other) const -> Vec2<T>;

            /**
             * @brief Adds a factor to all components of the vector.
             * 
             * Returns (x1+a, y1+a).
             * 
             * @param factor Factor added to all components of the vector.
             * @return Vec2 Resulting vector, (x1+a, y1+a).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator+(T2 factor) const -> Vec2<T>;

            /**
             * @brief Subtracts a vector from this vector.
             * 
             * Returns (x1-x2, y1-y2).
             * 
             * @param other Vector to subtract from this vector.
             * @return Vec2 Resulting vector, (x1-x2, y1-y2).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator-(const Vec2<T2> &other) const -> Vec2<T>;

            /**
             * @brief Subtracts a factor from all components of the vector.
             * 
             * Returns (x1-a, y1-a).
             * 
             * @param factor Factor subtracted from all components of the vector.
             * @return Vec2 Resulting vector, (x1-a, y1-a).
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator-(T2 factor) const -> Vec2<T>;

            /**
             * @brief Returns the vector negated.
             * 
             * Returns (-x1, -y1).
             * 
             * @return Vec2 Resulting vector, (-x1, -y1).
             */
            __host__ __device__
            constexpr auto operator-() const -> Vec2<T>; 

            /**
             * @brief Tests equality between two vectors.
             * 
             * @param other Vector used to test equality.
             * @return true All two components of the vectors are equal.
             * @return false At least one component of the vectors is different.
             */
            __host__ __device__
            constexpr auto operator==(const Vec2<T> &other) const -> bool;

            /**
             * @brief In-place multiplies all components of the vector by a factor.
             * 
             * Becomes (x1*a, y1*a).
             * 
             * @param scale Factor used to multiply all components of the vector.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator*=(T2 scale) -> const Vec2<T>&;

            /**
             * @brief In-place element-wise multiplication of the vector by another vector.
             * 
             * Becomes (x1*x2, y1*y2).
             * 
             * @param other Vector used to multiply.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator*=(const Vec2<T2> &other) -> const Vec2<T>&;

            /**
             * @brief In-place divides all components of the vector by a factor.
             * 
             * Becomes (x1/a, y1/a).
             * 
             * @param scale Factor used to divide all components of the vector.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator/=(T2 scale) -> const Vec2<T>&;

            /**
             * @brief In-place elements-wise division by the provided vector.
             * 
             * Becomes (x1/x2, y1/y2).
             * 
             * @param other Vector used to divide the components of this vector.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator/=(const Vec2<T2> &other) -> const Vec2<T>&;

            /**
             * @brief In-place addition of another vector.
             * 
             * Becomes (x1+x2, y1+y2).
             * 
             * @param other Vector added to this vector.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator+=(const Vec2<T2> &other) -> const Vec2<T>&;

            /**
             * @brief In-place adds a factor to all components of the vector.
             * 
             * Becomes (x1+a, y1+a).
             * 
             * @param factor Factor added to all components of the vector.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator+=(T2 factor) -> const Vec2<T>&;

            /**
             * @brief In-place subtracts a vector from this vector.
             * 
             * Becomes (x1-x2, y1-y2).
             * 
             * @param other Vector to subtract from this vector.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator-=(const Vec2<T2> &other) -> const Vec2<T>&; 

            /**
             * @brief In-place subtracts a factor from all components of the vector.
             * 
             * Becomes (x1-a, y1-a).
             * 
             * @param factor Factor subtracted from all components of the vector.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto operator-=(T2 factor) -> const Vec2<T>&;

            /**
             * @brief Sets the components of the vector to the minimum of its components and the other's.
             * 
             * Becomes (min(x1, x2), min(y1, y2))
             * 
             * @param other Vector to calculate minimum components with.
             * @return Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto min(const Vec2<T2> &other) -> Vec2<T>&;

            /**
             * @brief Sets the components of the vector to the minimum of its components and the provided factor.
             * 
             * Becomes (min(x1, a), min(y1, a))
             * 
             * @param other Factor to calculate minimum with.
             * @return Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto min(T2 other) -> Vec2<T>&;

            /**
             * @brief Sets the components of the vector to the maximum of its components and the other's.
             * 
             * Becomes (max(x1, x2), max(y1, y2))
             * 
             * @param other Vector to calculate maximum components with.
             * @return Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto max(const Vec2<T2> &other) -> Vec2<T>&;

            /**
             * @brief Sets the components of the vector to the maximum of its components and the provided factor.
             * 
             * Becomes (max(x1, a), max(y1, a))
             * 
             * @param other Factor to calculate maximum with.
             * @return Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto max(T2 other) -> Vec2<T>&;

            /**
             * @brief Returns a vector with the minimum components of this vector and another.
             * 
             * Returns (min(x1, x2), min(y1, y2))
             * 
             * @param other Vector to calculate minimum components with.
             * @return Vec2 Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto getMin(const Vec2<T2> &other) const -> Vec2<T>;

            /**
             * @brief Returns a vector with the minimum components of this vector and a factor.
             * 
             * Returns (min(x1, a), min(y1, a))
             * 
             * @param other Factor to calculate minimum with.
             * @return Vec2 Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto getMin(T2 other) const -> Vec2<T>;

            /**
             * @brief Returns a vector with the maximum components of this vector and another.
             * 
             * Returns (max(x1, x2), max(y1, y2))
             * 
             * @param other Vector to calculate maximum components with.
             * @return Vec2 Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto getMax(const Vec2<T2> &other) const -> Vec2<T>;

            /**
             * @brief Returns a vector with the maximum components of this vector and a factor.
             * 
             * Returns (max(x1, a), max(y1, a))
             * 
             * @param other Factor to calculate maximum with.
             * @return Vec2 Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            constexpr auto getMax(T2 other) const -> Vec2<T>;

            /**
             * @brief Returns the magnitude of the vector.
             * 
             * Returns the L2 norm of the vector: sqrt(x^2 + y^2).
             * 
             * @return T Magnitude of the vector.
             */
            __host__ __device__
            auto magnitude() const -> T;

            /**
             * @brief Returns the squared magnitude of the vector.
             * 
             * Returns x^2 + y^2. Useful because it is much faster than the norm,
             * and can be used instead of it in some situations.
             * 
             * @return T Squared magnitude of the norm.
             */
            __host__ __device__
            constexpr auto magnitudeSquared() const -> T;

            /**
             * @brief Returns a the normalized vector.
             * 
             * Divides all components of the vector by its magnitude.
             * 
             * @return Vec2 Normalized vector.
             */
            __host__ __device__
            auto normalize() const -> Vec2<T>;

            /**
             * @brief Normalizes the vector in-place, dividing it by its norm.
             * 
             * Divides all components of the vector by its magnitude.
             * 
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            __host__ __device__
            auto normalize_inplace() -> const Vec2<T>&;

            /**
             * @brief Computes the dot product of this vector and another.
             * 
             * Returns v1.v2
             * 
             * @param other Vector to dot with this one.
             * @return double Dot product of the two vectors.
             */
            template <class T2>
            __host__ __device__
            constexpr auto dot(const Vec2<T2> &other) const -> T;

            /**
             * @brief Changes the vector in-place to polar coordinates.
             * 
             * Assumes the vector is in cartesian coordinates.
             * 
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            __host__ __device__
            auto to_sph() -> const Vec2<T>&;

            /**
             * @brief Changes the vector in-place to cartesian coordinates.
             * 
             * Assumes the vector is in polar coordinates.
             * 
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            __host__ __device__
            auto to_xy() -> const Vec2<T>&;

            /**
             * @brief Changes the vector in-place to cartesian coordinates with arbitrary axises.
             * 
             * Assumes the vector is in polar coordinates.
             * 
             * @param ref1 Axis used for x.
             * @param ref2 Axis used for y.
             * @return const Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2, class T3>
            __host__ __device__
            auto to_xy_offset(const Vec2<T2>& ref1, const Vec2<T3>& ref2) -> const Vec2<T>&;

            /**
             * @brief Returns the vector in polar coordinates.
             * 
             * Assumes the vector is in cartesian coordinates.
             * 
             * @return Vec2 polar coordinates of the vector.
             */
            __host__ __device__
            auto get_sph() const -> Vec2<T>;

            /**
             * @brief Returns the vector in cartesian coordinates.
             * 
             * Assumes the vector is in polar coordinates.
             * 
             * @return Vec2 Cartesian coordinates of the vector.
             */
            __host__ __device__
            auto get_xy() const -> Vec2<T>;

            /**
             * @brief Returns the vector in cartesian coordinates with arbitrary axises.
             * 
             * Assumes the vector is in polar coordinates.
             * 
             * @param ref1 Axis used for x.
             * @param ref2 Axis used for y.
             * @return Vec2 Cartesian coordinates of the vector.
             */
            template <class T2, class T3>
            __host__ __device__
            auto get_xy_offset(const Vec2<T2>& ref1, const Vec2<T3>& ref2) const -> Vec2<T>;

            /**
             * @brief Returns a vector of the natural logarithm of all components of the vector.
             * 
             * Returns (ln(x), ln(y))
             * 
             * @return Vec2 Vector made of the natural logarithm of all components of the vector.
             */
            __host__ __device__
            auto ln() const -> Vec2<T>;

            /**
             * @brief Returns a vector of the square root of all components of the vector.
             * 
             * Returns (sqrt(x), sqrt(y))
             * 
             * @return Vec2 Vector made of the square root of all components of the vector.
             */
            __host__ __device__
            auto sqrt() const -> Vec2<T>;

            /**
             * @brief Returns a vector of the exponential of all components of the vector.
             * 
             * Returns (e^x, e^y).
             * 
             * @return Vec2 Vector made of the exponential of all components of the vector.
             */
            __host__ __device__
            auto exp() const -> Vec2<T>;

            /**
             * @brief Returns a vector of the components of the vector to the specified power.
             * 
             * Returns (x^a, y^a).
             * 
             * @param exp Power to be applied to all components.
             * @return Vec2 Vector made of the components of the vector to the specified power.
             */
            template <class T2>
            __host__ __device__
            auto pow(T2 exp) const -> Vec2<T>;

            /**
             * @brief In-place raise the components of the vector to the specified power.
             * 
             * Becomes (x^a, y^a).
             * 
             * @param exp Power to be applied to all components.
             * @return Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2>
            __host__ __device__
            auto pow_inplace(T2 exp) -> Vec2<T>&;

            /**
             * @brief Returns a vector of the components of the vector rounded down.
             * 
             * Returns (floor(x), floor(y))
             * 
             * @return Vec2 Vector made of the components of the vector rounded down.
             */
            __host__ __device__
            auto floor() const -> Vec2<T>;

            /**
             * @brief Returns a vector of the components of the vector rounded up.
             * 
             * Returns (ceil(x), ceil(y))
             * 
             * @return Vec2 Vector made of the components of the vector rounded up.
             */
            __host__ __device__
            auto ceil() const -> Vec2<T>;

            /**
             * @brief In-place rounds the components to the nearest integer value.
             * 
             * Becomes (round(x), round(y))
             * 
             * @return Vec2& Reference to the vector, used to chain operations.
             */
            __host__ __device__
            auto round_inplace() -> Vec2<T>&;

            /**
             * @brief In-place limits the components of the vector to a minimum and maximum value.
             * 
             * @param minimum Minimum value of the components.
             * @param maximum Maximum value of the components.
             * @return Vec2& Reference to the vector, used to chain operations.
             */
            template <class T2, class T3>
            __host__ __device__
            constexpr auto clamp(T2 minimum, T3 maximum) -> Vec2<T>&;

            /**
             * @brief Returns a reference to the x component of the vector
             * 
             * @return T& Reference to the x component of the vector.
             */
            __host__ __device__
            constexpr auto x() -> T&;
            
            /**
             * @brief Returns a reference to the y component of the vector
             * 
             * @return T& Reference to the y component of the vector.
             */
            __host__ __device__
            constexpr auto y() -> T&;

            /**
             * @brief Returns a const reference to the x component of the vector
             * 
             * @return T& Const reference to the x component of the vector.
             */
             __host__ __device__
             constexpr auto x() const -> const T&;
             
             /**
              * @brief Returns a const reference to the y component of the vector
              * 
              * @return T& Const reference to the y component of the vector.
              */
             __host__ __device__
             constexpr auto y() const -> const T&;
    };
}}

/**
 * @brief Formats a vector to be displayed.
 * 
 * @param output Output stream.
 * @param v Vector to be displayed.
 * @return std::ostream& Output stream.
 */
template <class T>
__host__
auto operator<<(std::ostream &output, const SEM::Entities::Vec2<T> &v) -> std::ostream&;

/**
 * @brief Multiplies a factor with a vector.
 * 
 * Returns (a*x, a*y).
 * 
 * @param factor Factor multiplying the vector.
 * @param v Vector to be multiplied.
 * @return SEM::Entities::Vec2 Resulting Vector, (a*x, a*y).
 */
template <class T, class T2>
__host__ __device__
constexpr auto operator*(const T2 factor, const SEM::Entities::Vec2<T> &v) -> SEM::Entities::Vec2<T>;

#include "entities/Vec2.tcu"

#endif