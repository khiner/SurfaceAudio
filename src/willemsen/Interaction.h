#pragma once
#include "core/FiniteDifference.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define WILLEMSEN_THREAD thread
#else
#include <cmath>
#define WILLEMSEN_THREAD
#endif
namespace surface_audio::willemsen {
#ifdef __METAL_VERSION__
using namespace metal;
#else
using std::abs;
using std::cos;
using std::exp;
using std::sin;
using std::sqrt;
#endif
enum : unsigned { MaximumPoints = 256 };
template<typename T> struct Model {
    T SampleRate{}, Spacing{}, Density{}, Loss0{}, Loss1{}, WaveSpeedSquared{}, StiffnessSquared{};
    T Dynamic{T(.3)}, Static{T(.8)}, Stribeck{T(.1)}, BristleStiffness{T(10000)}, BristleDamping{T(.1)}, Viscous{T(.4)};
    T Breakaway{T(.7)}, Noise{T(.02)}, Coupling{}, Update[5]{}, ContactUpdate[5]{}, Weight[4]{};
    unsigned Intervals{}, BowIndex{}, Pickup{}, AuthorFigure{};
};
template<typename T> struct State {
    T U[MaximumPoints]{}, Previous[MaximumPoints]{}, Z{}, Rate{}, Velocity{};
};
template<typename T> inline State<T> MakeState(const WILLEMSEN_THREAD Model<T> &m, T velocity) {
    return {.Rate = m.AuthorFigure ? velocity : T(0), .Velocity = m.AuthorFigure ? -velocity : T(0)};
}
template<typename T> struct Friction {
    T Steady{}, Adhesion{}, Rate{}, RateVelocity{}, RateState{};
};
template<typename T> inline Friction<T> Evaluate(const WILLEMSEN_THREAD Model<T> &m, T force, T v, T z) {
    const T sign = v < 0 ? T(-1) : T(1), zs = z < 0 ? T(-1) : T(1);
    const T e = exp(-v * v / (m.Stribeck * m.Stribeck));
    const T steady = force * (m.Dynamic + (m.Static - m.Dynamic) * e) / m.BristleStiffness;
    const T dsteady = -T(2) * v * force * (m.Static - m.Dynamic) * e / (m.Stribeck * m.Stribeck * m.BristleStiffness);
    const T ba = m.Breakaway * force * m.Dynamic / m.BristleStiffness;
    T alpha{}, av{}, az{};
    if (v * z > 0) {
        if (abs(z) >= steady) alpha = 1;
        else if (abs(z) > ba) {
            const T pi = T(3.14159265358979323846), width = steady - ba;
            const T phase = pi * ((abs(z) - ba) / width - T(.5));
            alpha = T(.5) * (T(1) + sin(phase));
            const T derivative = T(.5) * pi * cos(phase) / width;
            az = derivative * zs;
            av = derivative * (ba - abs(z)) / width * dsteady;
        }
    }
    const T ss = sign * steady, dss = sign * dsteady;
    return {ss, alpha, v * (T(1) - alpha * z / ss), T(1) - z * ((alpha + v * av) / ss - v * alpha * dss / (ss * ss)), -v * (alpha + z * az) / ss};
}
template<typename T> struct Sample {
    T Displacement{}, BowDisplacement{}, Force{}, Velocity{}, Residual{}, KinematicResidual{};
    unsigned Iterations{}, Failed{};
};
template<typename T> inline Sample<T> Step(const WILLEMSEN_THREAD Model<T> &m, WILLEMSEN_THREAD State<T> &s, T bow_velocity, T normal_force, T noise, T tolerance) {
    T next[MaximumPoints]{};
    const unsigned n = m.Intervals;
    for (unsigned j = 2; j < n - 1; ++j)
        next[j] = m.Update[0] * s.U[j] + m.Update[1] * (s.U[j - 1] + s.U[j + 1]) + m.Update[2] * (s.U[j - 2] + s.U[j + 2]) + m.Update[3] * s.Previous[j] + m.Update[4] * (s.Previous[j - 1] + s.Previous[j + 1]);
    for (unsigned side = 0; side < (m.AuthorFigure ? 0u : 2u); ++side) {
        const unsigned j = side ? n - 1 : 1, inward = side ? n - 2 : 2, second = side ? n - 3 : 3;
        next[j] = (m.Update[0] - m.Update[2]) * s.U[j] + m.Update[1] * s.U[inward] + m.Update[2] * s.U[second] + m.Update[3] * s.Previous[j] + m.Update[4] * s.Previous[inward];
    }
    T free_velocity{}, old_bow{};
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned i = m.BowIndex + j - 1;
        T contact = next[i];
        // The figure-generating MATLAB scales contact losses separately from the string update.
        if (m.AuthorFigure && m.Weight[j] != 0)
            contact = m.ContactUpdate[0] * s.U[i] + m.ContactUpdate[1] * (s.U[i - 1] + s.U[i + 1]) + m.ContactUpdate[2] * (s.U[i - 2] + s.U[i + 2]) + m.ContactUpdate[3] * s.Previous[i] + m.ContactUpdate[4] * (s.Previous[i - 1] + s.Previous[i + 1]);
        free_velocity += m.Weight[j] * (contact - s.Previous[i]) * (m.SampleRate / T(2));
        old_bow += m.Weight[j] * s.Previous[i];
    }
    const T target = free_velocity - bow_velocity, history = s.Z + s.Rate / (T(2) * m.SampleRate);
    T v = s.Velocity, z = s.Z, rate{}, force{}, residual{};
    unsigned iterations{};
    bool converged = normal_force <= 0;
    if (normal_force > 0) {
        for (; iterations < 50; ++iterations) {
            const auto f = Evaluate(m, normal_force, v, z);
            force = m.BristleStiffness * z + m.BristleDamping * f.Rate + m.Viscous * v + m.Noise * normal_force * noise;
            const T g1 = v + m.Coupling * force - target, g2 = z - history - f.Rate / (T(2) * m.SampleRate);
            const T a = T(1) + m.Coupling * (m.BristleDamping * f.RateVelocity + m.Viscous);
            const T b = m.Coupling * (m.BristleStiffness + m.BristleDamping * f.RateState);
            const T c = -f.RateVelocity / (T(2) * m.SampleRate), d = T(1) - f.RateState / (T(2) * m.SampleRate);
            const T determinant = a * d - b * c, dv = (d * g1 - b * g2) / determinant, dz = (a * g2 - c * g1) / determinant;
            v -= dv;
            z -= dz;
            if (sqrt(dv * dv + dz * dz) <= tolerance) {
                converged = true;
                ++iterations;
                break;
            }
        }
        const auto f = Evaluate(m, normal_force, v, z);
        rate = f.Rate;
        force = m.BristleStiffness * z + m.BristleDamping * rate + m.Viscous * v + m.Noise * normal_force * noise;
        residual = abs(v + m.Coupling * force - target) + abs(z - history - rate / (T(2) * m.SampleRate)) * m.SampleRate;
    } else {
        v = target;
        z = 0;
    }
    const T spread = force / (m.Density * m.Spacing * m.SampleRate * (m.SampleRate + m.Loss0));
    for (unsigned j = 0; j < 4; ++j) next[m.BowIndex + j - 1] -= spread * m.Weight[j];
    T bow{};
    for (unsigned j = 0; j < 4; ++j) bow += m.Weight[j] * next[m.BowIndex + j - 1];
    for (unsigned j = 0; j <= n; ++j) {
        s.Previous[j] = s.U[j];
        s.U[j] = next[j];
    }
    s.Z = z;
    s.Rate = rate;
    s.Velocity = v;
    return {next[m.Pickup], bow, force, v, residual, abs((bow - old_bow) * m.SampleRate / T(2) - bow_velocity - v), iterations, unsigned(!converged)};
}
}
#undef WILLEMSEN_THREAD
