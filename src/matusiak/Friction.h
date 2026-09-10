// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <cmath>
using std::abs;
using std::cos;
using std::exp;
using std::sin;
using std::sqrt;
#endif

namespace surface_audio::matusiak {
template<typename T> struct FrictionParameters {
    T Stiffness{T(318600)}, Damping{T(.0027)}, StribeckVelocity{T(.228)};
    T Dynamic{T(.5071)}, Static{T(1.0207)}, BreakawayRatio{T(.7)};
};
template<typename T> struct FrictionValue {
    T Rate{}, RateVelocity{}, RateBristle{}, Damping{}, DampingVelocity{}, Steady{}, Adhesion{}, Force{}, Dissipation{};
};
template<typename T> FrictionValue<T> Evaluate(FrictionParameters<T> p, T normal_force, T z, T v) {
    const T fc = p.Dynamic * normal_force, fs = p.Static * normal_force;
    const T sign_v = v < 0 ? T(-1) : T(1), sign_z = z < 0 ? T(-1) : T(1);
    const T e = exp(-v * v / (p.StribeckVelocity * p.StribeckVelocity));
    const T ss = (fc + (fs - fc) * e) / p.Stiffness;
    const T dss = -2 * v * (fs - fc) * e / (p.Stiffness * p.StribeckVelocity * p.StribeckVelocity);
    const T ba = p.BreakawayRatio * fc / p.Stiffness;
    T alpha{}, az{}, av{};
    if (v * z > 0 && abs(z) > ba) {
        if (abs(z) >= ss) alpha = 1;
        else {
            const T theta = T(3.14159265358979323846) * (abs(z) - (ss + ba) / 2) / (ss - ba);
            alpha = (1 + sin(theta)) / 2;
            az = sign_z * T(3.14159265358979323846 / 2) * cos(theta) / (ss - ba);
            av = dss * (ba - abs(z)) * T(3.14159265358979323846 / 2) * cos(theta) / ((ss - ba) * (ss - ba));
        }
    }
    const T zss = sign_v * ss, dzss = sign_v * dss;
    const T rate = v * (1 - alpha * z / zss);
    const T rz = -v * (az * z + alpha) / zss;
    const T rv = 1 - z * ((alpha + av * v) * zss - dzss * alpha * v) / (zss * zss);
    const T epsilon = p.Damping > 0 ? fc / p.Damping : T(1);
    const T denominator = sqrt(v * v + epsilon * epsilon);
    const T damping = p.Damping > 0 ? fc / denominator : T(0);
    const T dd = -damping * v / (denominator * denominator);
    const T force = p.Stiffness * z + damping * rate;
    return {rate, rv, rz, damping, dd, zss, alpha, force, v * force - p.Stiffness * z * rate};
}
}
