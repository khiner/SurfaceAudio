// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "matusiak/Friction.h"
#ifdef __METAL_VERSION__
#define MAT_THREAD thread
#else
#define MAT_THREAD
#endif
namespace surface_audio::matusiak {
template<typename T> struct LumpedParameters {
    T SampleRate{44100}, Mass{T(.0028)}, Stiffness{T(1055.7)}, Damping{T(.0095)};
    T HairMass{T(.0042)}, HairStiffness{T(48297)}, HairDamping{T(57.674)};
    T NormalForce{T(1.6403)}, BowVelocity{T(.3439)}, Acceleration{T(3.439)};
    FrictionParameters<T> Friction{T(100000), T(.5), T(.228), T(.5071), T(1.0207), T(.7)};
};
template<typename T> struct LumpedState {
    T U{}, PreviousU{}, Hair{}, PreviousHair{}, Z{}, Velocity{}, MidpointZ{}, Loss{};
};
template<typename T> struct LumpedSample {
    T Displacement{}, Energy{}, EnergyError{}, BristleDissipation{}, Residual{};
    unsigned Iterations{};
};
// Eqs. 25-35: central mass update, averaged bow spring, interleaved bristles.
template<typename T> LumpedSample<T> StepLumped(LumpedParameters<T> p, MAT_THREAD LumpedState<T> &s, T bow_velocity, T tolerance) {
    const T k = 1 / p.SampleRate, xs = 1 / (2 * p.Mass / k + p.Damping), xh = 1 / (2 * p.HairMass / k + k * p.HairStiffness / 2 + p.HairDamping);
    const T su = xs * (-p.Stiffness * s.U + 2 * p.Mass / (k * k) * (s.U - s.PreviousU));
    const T sh = xh * (-p.HairStiffness * s.Hair + (2 * p.HairMass / k + k * p.HairStiffness / 2) * (s.Hair - s.PreviousHair) / k);
    const T coupling = xs + xh, free = su + sh - bow_velocity;
    LumpedSample<T> result;
    for (unsigned i = 0; i < 100; ++i) {
        const auto f = Evaluate(p.Friction, p.NormalForce, s.MidpointZ, s.Velocity);
        const T r1 = s.Velocity + coupling * f.Force - free, r2 = f.Rate - 2 / k * (s.MidpointZ - s.Z);
        result.Residual = abs(r1) > abs(r2) ? abs(r1) : abs(r2);
        if (result.Residual < tolerance) break;
        const T a = 1 + coupling * (f.Damping * f.RateVelocity + f.DampingVelocity * f.Rate), b = coupling * (p.Friction.Stiffness + f.Damping * f.RateBristle);
        const T c = f.RateVelocity, d = f.RateBristle - 2 / k, det = a * d - b * c;
        const T scale = i > 50 ? T(1.1) : T(1);
        s.Velocity -= (d * r1 - b * r2) / (det * scale);
        s.MidpointZ -= (-c * r1 + a * r2) / (det * scale);
        result.Iterations = i + 1;
    }
    const auto f = Evaluate(p.Friction, p.NormalForce, s.MidpointZ, s.Velocity);
    const T rate = 2 / k * (s.MidpointZ - s.Z), force = p.Friction.Stiffness * s.MidpointZ + f.Damping * rate;
    const T u = s.PreviousU + 2 * k * (su - xs * force), hair = s.PreviousHair + 2 * k * (sh - xh * force);
    const T vu = (u - s.PreviousU) / (2 * k), vh = (hair - s.PreviousHair) / (2 * k);
    result.BristleDissipation = s.Velocity * force - p.Friction.Stiffness * s.MidpointZ * rate;
    s.Loss += k * (bow_velocity * force + p.Damping * vu * vu + p.HairDamping * vh * vh + result.BristleDissipation);
    s.Z = 2 * s.MidpointZ - s.Z;
    s.PreviousU = s.U;
    s.U = u;
    s.PreviousHair = s.Hair;
    s.Hair = hair;
    result.Energy = p.Mass / (2 * k * k) * (u - s.PreviousU) * (u - s.PreviousU) + p.Stiffness / 2 * u * s.PreviousU + p.HairMass / (2 * k * k) * (hair - s.PreviousHair) * (hair - s.PreviousHair) + p.HairStiffness / 8 * (hair + s.PreviousHair) * (hair + s.PreviousHair) + p.Friction.Stiffness / 2 * s.Z * s.Z;
    result.EnergyError = result.Energy + s.Loss;
    result.Displacement = u;
    return result;
}
} // namespace surface_audio::matusiak
#undef MAT_THREAD
