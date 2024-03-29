#include <algorithm>
#include <cmath>
#include <limits>

template <typename T>
constexpr SEM::Host::Entities::Vec2<T>::Vec2() : v_{0, 0} {}

template <typename T>
constexpr SEM::Host::Entities::Vec2<T>::Vec2(T x, T y) : v_{x, y} {}

template <typename T>
constexpr SEM::Host::Entities::Vec2<T>::Vec2(T x) : v_{x, x} {}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator[](T2 index) -> T& {
    return v_[index];
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator[](T2 index) const -> const T& {
    return v_[index];
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator*(T2 scale) const -> Vec2<T> {
    return {v_[0] * scale, v_[1] * scale};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator*(const Vec2<T2> &other) const -> Vec2<T> {
    return {v_[0] * other[0], v_[1] * other[1]};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator/(T2 scale) const -> Vec2<T> {
    return {v_[0] / scale, v_[1] / scale};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator/(const Vec2<T2> &other) const -> Vec2<T> {
    return {v_[0] / other[0], v_[1] / other[1]};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator+(const Vec2<T2> &other) const -> Vec2<T> {
    return {v_[0] + other[0], v_[1] + other[1]};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator+(T2 factor) const -> Vec2<T> {
    return {v_[0] + factor, v_[1] + factor};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator-(const Vec2<T2> &other) const -> Vec2<T> {
    return {v_[0] - other[0], v_[1] - other[1]};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator-(T2 factor) const -> Vec2<T> {
    return {v_[0] - factor, v_[1] - factor};
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::operator-() const -> Vec2<T> {
    return {-v_[0], -v_[1]};
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::operator==(const Vec2<T> &other) const -> bool {
    return (v_[0] == other[0]) && (v_[1] == other[1]);
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator*=(T2 scale) -> const Vec2<T>& {
    v_[0] *= scale;
    v_[1] *= scale;
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator*=(const Vec2<T2> &other) -> const Vec2<T>& {
    v_[0] *= other[0];
    v_[1] *= other[1];
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator/=(T2 scale) -> const Vec2<T>& {
    v_[0] /= scale;
    v_[1] /= scale;
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator/=(const Vec2<T2> &other) -> const Vec2<T>& {
    v_[0] /= other[0];
    v_[1] /= other[1];
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator+=(const Vec2<T2> &other) -> const Vec2<T>& {
    v_[0] += other[0];
    v_[1] += other[1];
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator+=(T2 factor) -> const Vec2<T>& {
    v_[0] += factor;
    v_[1] += factor;
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator-=(const Vec2<T2> &other) -> const Vec2<T>& {
    v_[0] -= other.v_[0];
    v_[1] -= other.v_[1];
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::operator-=(T2 factor) -> const Vec2<T>& {
    v_[0] -= factor;
    v_[1] -= factor;
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::min(const Vec2<T2> &other) -> Vec2<T>& {
    v_[0] = std::min(v_[0], other[0]);
    v_[1] = std::min(v_[1], other[1]);
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::min(T2 other) -> Vec2<T>& {
    v_[0] = std::min(v_[0], other);
    v_[1] = std::min(v_[1], other);
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::max(const Vec2<T2> &other) -> Vec2<T>& {
    v_[0] = std::max(v_[0], other[0]);
    v_[1] = std::max(v_[1], other[1]);
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::max(T2 other) -> Vec2<T>& {
    v_[0] = std::max(v_[0], other);
    v_[1] = std::max(v_[1], other);
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::getMin(const Vec2<T2> &other) const -> Vec2<T> {
    return {std::min(v_[0], other[0]), std::min(v_[1], other[1])};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::getMin(T2 other) const -> Vec2<T> {
    return {std::min(v_[0], other), std::min(v_[1], other)};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::getMax(const Vec2<T2> &other) const -> Vec2<T> {
    return {std::max(v_[0], other[0]), std::max(v_[1], other[1])};
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::getMax(T2 other) const -> Vec2<T> {
    return {std::max(v_[0], other), std::max(v_[1], other)};
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::magnitude() const -> T {
    return std::sqrt(v_[0] * v_[0] + v_[1] * v_[1]);
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::magnitudeSquared() const -> T {
    return v_[0] * v_[0] + v_[1] * v_[1];
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::normalize() const -> Vec2<T> {
    const T m = magnitude();
    return {v_[0] / m, v_[1] / m};
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::normalize_inplace() -> const Vec2<T>& {
    const T m = magnitude();
    v_[0] /= m;
    v_[1] /= m;
    return *this;
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::dot(const Vec2<T2> &other) const -> T {
    return v_[0] * other[0] + v_[1] * other[1];
}

template <typename T>
template <class T2>
constexpr auto SEM::Host::Entities::Vec2<T>::cross(const Vec2<T2> &other) const -> T {
    return v_[0] * other[1] - v_[1] * other[0];
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::to_sph() -> const Vec2<T>& {
    // [r, phi]
    const T temp = std::atan2(v_[1], v_[0]);
    v_[0] = magnitude();
    v_[1] = temp;
    return *this;
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::to_xy() -> const Vec2<T>& {
    const T temp = v_[0];
    v_[0] = temp*std::cos(v_[1]);
    v_[1] = temp*std::sin(v_[1]);
    return *this;
}

template <typename T>
template <class T2, class T3>
auto SEM::Host::Entities::Vec2<T>::to_xy_offset(const Vec2<T2>& ref1, const Vec2<T3>& ref2) -> const Vec2<T>& {
    const Vec2 temp = Vec2(v_[0]*std::cos(v_[1]), v_[0]*std::sin(v_[1])); // CHECK could be better
    v_[0] = ref1[0] * temp[0] + ref2[0] * temp[1];
    v_[1] = ref1[1] * temp[0] + ref2[1] * temp[1];
    return *this;
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::get_sph() const -> Vec2<T> {
    return {magnitude(), std::atan2(v_[1], v_[0])};
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::get_xy() const -> Vec2<T> {
    return {v_[0]*std::cos(v_[1]), v_[0]*std::sin(v_[1])};
}

template <typename T>
template <class T2, class T3>
auto SEM::Host::Entities::Vec2<T>::get_xy_offset(const Vec2<T2>& ref1, const Vec2<T3>& ref2) const -> Vec2<T> {
    return ref1 * v_[0]*std::cos(v_[1]) + ref2 * v_[0]*std::sin(v_[1]);
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::ln() const -> Vec2<T> {
    return {std::log(v_[0]), std::log(v_[1])};
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::sqrt() const -> Vec2<T> {
    return {std::sqrt(v_[0]), std::sqrt(v_[1])};
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::exp() const -> Vec2<T> {
    return {std::exp(v_[0]), std::exp(v_[1])};
}

template <typename T>
template <class T2>
auto SEM::Host::Entities::Vec2<T>::pow(T2 exp) const -> Vec2<T> {
    return {std::pow(v_[0], exp), std::pow(v_[1], exp)};
}

template <typename T>
template <class T2>
auto SEM::Host::Entities::Vec2<T>::pow_inplace(T2 exp) -> Vec2<T>& {
    v_[0] = std::pow(v_[0], exp);
    v_[1] = std::pow(v_[1], exp);
    return *this;
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::floor() const -> Vec2<T> {
    return {std::floor(v_[0]), std::floor(v_[1])};
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::ceil() const -> Vec2<T> {
    return {std::ceil(v_[0]), std::ceil(v_[1])};
}

template <typename T>
auto SEM::Host::Entities::Vec2<T>::round_inplace() -> Vec2<T>& {
    v_[0] = std::round(v_[0]);
    v_[1] = std::round(v_[1]);
    return *this;
}

template <typename T>
template <class T2, class T3>
constexpr auto SEM::Host::Entities::Vec2<T>::clamp(T2 minimum, T3 maximum) -> Vec2<T>& {
    min(maximum);
    max(minimum);
    return *this;
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::x() -> T& {
    return v_[0];
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::y() -> T& {
    return v_[1];
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::x() const -> const T& {
    return v_[0];
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::y() const -> const T& {
    return v_[1];
}

template <typename T>
constexpr auto SEM::Host::Entities::Vec2<T>::almost_equal(const Vec2<T> &other) const -> bool {
    constexpr int ulp = 2; // ULP
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return (std::abs(v_[0] - other.v_[0]) <= std::numeric_limits<T>::epsilon() * std::abs(v_[0] + other.v_[0]) * ulp
        // unless the result is subnormal
        || std::abs(v_[0] - other.v_[0]) < std::numeric_limits<T>::min()) 
        
        && (std::abs(v_[1] - other.v_[1]) <= std::numeric_limits<T>::epsilon() * std::abs(v_[1] + other.v_[1]) * ulp
        || std::abs(v_[1] - other.v_[1]) < std::numeric_limits<T>::min());
}

template <class T>
auto operator<<(std::ostream &output, const SEM::Host::Entities::Vec2<T> &v) -> std::ostream& {
    output << '[' << v[0] << ", " << v[1] << ']';
    return output;
}

template <class T, class T2>
constexpr auto operator*(const T2 factor, const SEM::Host::Entities::Vec2<T> &v) -> SEM::Host::Entities::Vec2<T> {
    return {v[0] * factor, v[1] * factor};
}