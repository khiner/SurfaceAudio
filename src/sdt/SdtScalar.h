// SPDX-License-Identifier: GPL-3.0-or-later
// Adapted from SkAT-VG/SDT, commit 0509de418e7bebc8b37866b3b4458e0acc8cf1f4.
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define SDT_THREAD thread
#else
#include <algorithm>
#include <cmath>
#define SDT_THREAD
#endif

namespace surface_audio::sdt {
#ifdef __METAL_VERSION__
using metal::abs;
using metal::exp;
using metal::max;
using metal::min;
using metal::pow;
using metal::sin;
using metal::sqrt;
#else
using std::abs;
using std::exp;
using std::max;
using std::min;
using std::pow;
using std::sin;
using std::sqrt;
#endif

template<typename T> struct ImpactParametersT {
    T Stiffness{1000000}, Dissipation{0.1}, Shape{1.5};
};

template<typename T> struct FrictionParametersT {
    T NormalForce{1}, StribeckVelocity{0.1}, StaticCoefficient{0.8}, DynamicCoefficient{0.2}, BreakAway{0.1};
    T Stiffness{1000}, Dissipation{10}, Viscosity{10}, Noisiness{0.5};
};

template<typename T> struct FrictionStateT {
    T Bristle{};
};

template<typename T> struct ContactEnergyT {
    T Free{}, Linear{}, Quadratic{};
};

// Solve A f² + B f <= energy on the segment from zero to the candidate force.
template<typename T> inline T LimitContactForce(ContactEnergyT<T> polynomial, SDT_THREAD T &energy, T force) {
    const T change = (polynomial.Quadratic * force + polynomial.Linear) * force;
    if (change > energy) {
        const T direction = force < T(0) ? T(-1) : T(1);
        const T linear = polynomial.Linear * direction;
        T limit{};
        if (polynomial.Quadratic > T(0)) {
            const T root = sqrt(linear * linear + T(4) * polynomial.Quadratic * energy);
            limit = linear >= T(0) ? (energy > T(0) ? T(2) * energy / (linear + root) : T(0)) : (root - linear) / (T(2) * polynomial.Quadratic);
        } else if (linear > T(0)) limit = energy / linear;
        force = direction * min(abs(force), limit);
    }
    energy = max(T(0), energy - (polynomial.Quadratic * force + polynomial.Linear) * force);
    return force;
}

template<typename T> struct RollingParametersT {
    T Grain{0.1}, Depth{1}, Mass{0.01}, Velocity{1};
};

template<typename T> struct RollingStateT {
    T GroundTrace{}, BallFlight{};
};

template<typename T> struct ScrapingParametersT {
    T Grain{0.1}, Force{1}, Velocity{1};
};

template<typename T> struct ScrapingStateT {
    T GroundTrace{};
};

template<typename T> inline T ImpactForce(ImpactParametersT<T> parameters, T compression, T velocity) {
    if (compression <= T(0)) return T(0);
    return parameters.Stiffness * pow(compression, parameters.Shape) * (T(1) + parameters.Dissipation * velocity);
}

template<typename T> inline T UnilateralImpactForce(ImpactParametersT<T> parameters, T compression, T velocity) { return max(T(0), ImpactForce(parameters, compression, velocity)); }

template<typename T> inline T FrictionForce(FrictionParametersT<T> parameters, SDT_THREAD FrictionStateT<T> &state, T velocity, T noise_sample, T time_step) {
    if (parameters.NormalForce <= T(0)) {
        state.Bristle = T(0);
        return T(0);
    }
    const T ratio = velocity / parameters.StribeckVelocity;
    const T sign = velocity > T(0) ? T(1) : velocity < T(0) ? T(-1) :
                                                              T(0);
    const T bristle_sign = state.Bristle > T(0) ? T(1) : state.Bristle < T(0) ? T(-1) :
                                                                                T(0);
    const T coulomb = parameters.NormalForce * parameters.DynamicCoefficient;
    const T steady = sign * (coulomb + parameters.NormalForce * (parameters.StaticCoefficient - parameters.DynamicCoefficient) * exp(-ratio * ratio)) / parameters.Stiffness;
    const T breakaway = sign * parameters.BreakAway * coulomb / parameters.Stiffness;
    T alpha{};
    if (sign == bristle_sign && abs(state.Bristle) >= abs(breakaway)) {
        alpha = abs(state.Bristle) < abs(steady) ? T(0.5) + T(0.5) * sin(T(3.14159265358979323846) * (state.Bristle - T(0.5) * (steady + breakaway)) / (steady - breakaway)) : T(1);
    }
    const T derivative = steady != T(0) ? velocity * (T(1) - alpha * state.Bristle / steady) : T(0);
    const T force = parameters.Stiffness * state.Bristle + parameters.Dissipation * derivative + parameters.Viscosity * velocity + parameters.Noisiness * noise_sample * sqrt(abs(velocity) * parameters.NormalForce);
    state.Bristle += derivative * time_step;
    return force;
}

template<typename T> inline T GroundDecay(T grain, T velocity) { return min(T(2), max(T(0), T(2) * grain * abs(velocity))); }

template<typename T> inline T StepRolling(RollingParametersT<T> parameters, SDT_THREAD RollingStateT<T> &state, T surface_sample) {
    const T decay = GroundDecay(parameters.Grain, parameters.Velocity);
    const T gravity = T(9.81) * parameters.Mass;
    const T ground = max(state.GroundTrace - decay, surface_sample);
    T out = -gravity;
    if (ground > state.GroundTrace && state.BallFlight == T(0) && decay > T(0)) {
        const T bump = (ground - state.GroundTrace) * parameters.Depth * T(0.5) * parameters.Mass * parameters.Velocity * parameters.Velocity / sqrt(decay);
        state.BallFlight = T(2) * bump;
        out += bump;
    }
    state.GroundTrace = ground;
    state.BallFlight = max(T(0), state.BallFlight - gravity);
    return out;
}

template<typename T> inline T StepScraping(ScrapingParametersT<T> parameters, SDT_THREAD ScrapingStateT<T> &state, T surface_sample) {
    const T decay = GroundDecay(parameters.Grain, parameters.Velocity);
    const T ground = max(state.GroundTrace - decay, surface_sample);
    const T out = ground > state.GroundTrace && decay > T(0) ? -parameters.Force * parameters.Velocity * parameters.Velocity * (ground - state.GroundTrace) / sqrt(decay) : T(0);
    state.GroundTrace = ground;
    return out;
}
} // namespace surface_audio::sdt

#undef SDT_THREAD
