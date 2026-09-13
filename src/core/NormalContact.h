#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cmath>
#endif

namespace surface_audio {
template<typename T> struct NormalContact {
    T Force{}, Energy{}, Dissipation{}, ElasticTangent{};
};
// Positive rate increases penetration; stiffness and damping coefficients must be nonnegative.
template<typename T> NormalContact<T> EvaluateNormalContact(T penetration, T rate, T stiffness, T damping, T exponent, T damping_exponent) {
#ifdef __METAL_VERSION__
    using metal::pow;
    using metal::sqrt;
#else
    using std::pow;
    using std::sqrt;
#endif
    if (penetration <= 0) return {};
    const auto power = [penetration](T n) { return n == T(1.5) ? penetration * sqrt(penetration) : n == T(1) ? penetration :
                                                                                                               pow(penetration, n); };
    const T elastic = stiffness * power(exponent), trial = elastic + damping * power(damping_exponent) * rate;
    const T force = trial > 0 ? trial : T(0);
    return {force, elastic * penetration / (exponent + 1), (force - elastic) * rate, exponent * elastic / penetration};
}
}
