#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define FAL_THREAD thread
#else
#include <cmath>
using std::abs;
using std::cos;
using std::exp;
using std::pow;
using std::sin;
#define FAL_THREAD
#endif
namespace surface_audio::falaize {
enum : unsigned { MaximumModes = 128 };
template<typename T> struct Parameters {
    T SampleRate{96000}, Length{T(1.8)}, Density{T(.0245)}, Tension{61483}, StringDamping{T(.1)}, Position{T(.3)};
    T NormalForce{1}, Dynamic{T(.3)}, Static{T(.8)}, StribeckVelocity{T(.1)}, Stiffness{10000};
    T ComplianceDamping{T(.1)}, FluidDamping{T(.4)}, BreakawayRatio{T(.7)}, Theta{};
    T HammerStiffness{T(13.8)}, HammerDamping{T(.184)}, HammerExponent{T(2.5)}, HammerThickness{T(.015)}, HammerMass{T(.03)};
    unsigned Modes{10}, Elements{};
};
template<typename T> struct Friction {
    T Resistance{}, ResistanceVelocity{}, ResistanceState{}, Adhesion{}, Force{}, Rate{}, Dissipation{}, Determinant{};
};
template<typename T> Friction<T> Evaluate(Parameters<T> p, T state, T elastic, T velocity) {
    const T e = exp(-velocity * velocity / (p.StribeckVelocity * p.StribeckVelocity));
    const T fc = p.NormalForce * p.Dynamic, fs = fc + p.NormalForce * (p.Static - p.Dynamic) * e;
    const T dfs = -2 * velocity * p.NormalForce * (p.Static - p.Dynamic) * e / (p.StribeckVelocity * p.StribeckVelocity);
    const T ss = fs / p.Stiffness, ba = p.BreakawayRatio * fc / p.Stiffness;
    T alpha{}, az{}, av{};
    if (velocity * state > 0 && abs(state) > ba) {
        if (abs(state) >= ss) alpha = 1;
        else {
            const T angle = T(3.14159265358979323846) * (abs(state) - (ss + ba) / 2) / (ss - ba);
            alpha = (1 + sin(angle)) / 2;
            az = (state < 0 ? T(-1) : T(1)) * T(1.57079632679489661923) * cos(angle) / (ss - ba);
            av = dfs / p.Stiffness * (ba - abs(state)) * T(1.57079632679489661923) * cos(angle) / ((ss - ba) * (ss - ba));
        }
    }
    const T r = alpha * abs(velocity) / fs;
    const T rv = (av * abs(velocity) + alpha * (velocity < 0 ? T(-1) : T(1))) / fs - r * dfs / fs;
    const T rz = az * abs(velocity) / fs, force_e = p.Stiffness * elastic, rate = velocity - r * force_e;
    const T force = force_e + p.ComplianceDamping * rate + p.FluidDamping * velocity;
    return {r, rv, rz, alpha, force, rate, velocity * force - force_e * rate, r * (p.ComplianceDamping + p.FluidDamping) - p.ComplianceDamping * p.ComplianceDamping * r * r / 4};
}
template<typename T> T HammerEnergy(Parameters<T> p, T q) {
    return q > 0 ? p.HammerStiffness * p.HammerThickness * p.HammerThickness / (p.HammerExponent + 1) * pow(q / p.HammerThickness, p.HammerExponent + 1) : T(0);
}
template<typename T> T HammerForce(Parameters<T> p, T q) {
    return q > 0 ? p.HammerStiffness * p.HammerThickness * pow(q / p.HammerThickness, p.HammerExponent) : T(0);
}
template<typename T> T HammerGradient(Parameters<T> p, T q0, T q1) {
    const T delta = q1 - q0, scale = abs(q0) + abs(q1) + T(1e-20);
    if (abs(delta) < T(1e-4) * scale && q0 > 0 && q1 > 0) {
        const T mid = (q0 + q1) / 2, ratio = delta / (2 * mid), b = p.HammerExponent;
        return HammerForce(p, mid) * (1 + b * (b - 1) * ratio * ratio / 6 + b * (b - 1) * (b - 2) * (b - 3) * pow(ratio, T(4)) / 120);
    }
    return delta != 0 ? (HammerEnergy(p, q1) - HammerEnergy(p, q0)) / delta : HammerForce(p, q0);
}
template<typename T> struct Model {
    Parameters<T> Config;
    T OmegaSquared[MaximumModes]{}, Shape[MaximumModes]{}, Response[MaximumModes]{};
    T Coupling{};
};
template<typename T> struct State {
    T Displacement[MaximumModes]{}, Velocity[MaximumModes]{};
    T Elastic{}, HammerVelocity{}, ContactVelocity{}, Force{}, Loss{};
};
template<typename T> struct Sample {
    T Displacement{}, Velocity{}, Force{}, RelativeVelocity{}, Energy{}, Dissipation{}, InputPower{}, BalanceError{}, Residual{};
    unsigned Iterations{}, Failed{};
};
template<typename T> T Energy(FAL_THREAD const Model<T> &m, FAL_THREAD const State<T> &s, bool hammer) {
    T energy{};
    for (unsigned i = 0; i < m.Config.Modes; ++i) energy += m.Config.Density / 2 * (s.Velocity[i] * s.Velocity[i] + m.OmegaSquared[i] * s.Displacement[i] * s.Displacement[i]);
    return energy + (hammer ? HammerEnergy(m.Config, s.Elastic) + m.Config.HammerMass / 2 * s.HammerVelocity * s.HammerVelocity : m.Config.Stiffness / 2 * s.Elastic * s.Elastic);
}
template<typename T> Sample<T> StepString(FAL_THREAD const Model<T> &m, FAL_THREAD State<T> &s) {
    const T dt = 1 / m.Config.SampleRate, before = Energy(m, s, false);
    Sample<T> out;
    for (unsigned i = 0; i < m.Config.Modes; ++i) {
        const T mid = (s.Velocity[i] - dt / 2 * m.OmegaSquared[i] * s.Displacement[i]) / (1 + dt * m.Config.StringDamping / (2 * m.Config.Density) + dt * dt / 4 * m.OmegaSquared[i]);
        s.Displacement[i] += dt * mid;
        s.Velocity[i] = 2 * mid - s.Velocity[i];
        out.Displacement += m.Shape[i] * s.Displacement[i];
        out.Velocity += m.Shape[i] * s.Velocity[i];
        out.Dissipation += m.Config.StringDamping * mid * mid;
    }
    s.Loss += dt * out.Dissipation;
    out.Energy = Energy(m, s, false);
    out.BalanceError = out.Energy - before + dt * out.Dissipation;
    return out;
}
template<typename T> Sample<T> Step(FAL_THREAD const Model<T> &m, FAL_THREAD State<T> &s, T bow_velocity, bool hammer, T tolerance) {
    const auto p = m.Config;
    const T dt = 1 / p.SampleRate, before = Energy(m, s, hammer);
    T free_velocity{}, free[MaximumModes];
    for (unsigned i = 0; i < p.Modes; ++i) {
        free[i] = (s.Velocity[i] - dt / 2 * m.OmegaSquared[i] * s.Displacement[i]) / (1 + dt * p.StringDamping / (2 * p.Density) + dt * dt / 4 * m.OmegaSquared[i]);
        free_velocity += m.Shape[i] * free[i];
    }
    Sample<T> out;
    T v = s.ContactVelocity, z = s.Elastic, force{}, dissipation{};
    if (hammer) {
        const T coupling = m.Coupling + dt / (2 * p.HammerMass), drive = s.HammerVelocity - free_velocity;
        v = drive - coupling * s.Force;
        for (unsigned iteration = 0; iteration < 40; ++iteration) {
            const T next = s.Elastic + dt * v, q = s.Elastic + p.Theta * dt * v;
            const T c = q > 0 ? q / p.HammerThickness : T(0);
            const T damping = q > 0 ? p.HammerDamping * p.HammerExponent * pow(c, p.HammerExponent - 1) : T(0);
            const T gradient = HammerGradient(p, s.Elastic, next);
            force = gradient + damping * v;
            const T residual = v + coupling * force - drive;
            out.Residual = abs(residual);
            if (out.Residual <= tolerance) break;
            const T delta = next - s.Elastic;
            const T slope = abs(delta) > T(1e-4) * (abs(next) + abs(s.Elastic) + T(1e-20)) ? (HammerForce(p, next) - gradient) / delta : (next > 0 ? p.HammerStiffness * p.HammerExponent / 2 * pow(next / p.HammerThickness, p.HammerExponent - 1) : T(0));
            const T dd = q > 0 ? damping * (p.HammerExponent - 1) / q * p.Theta * dt : T(0);
            v -= residual / (1 + coupling * (dt * slope + damping + dd * v));
            out.Iterations = iteration + 1;
        }
        z = s.Elastic + dt * v;
        const T q = s.Elastic + p.Theta * dt * v;
        dissipation = q > 0 ? p.HammerDamping * p.HammerExponent * pow(q / p.HammerThickness, p.HammerExponent - 1) * v * v : T(0);
    } else {
        v = bow_velocity - free_velocity - m.Coupling * s.Force;
        T rate = v;
        for (unsigned iteration = 0; iteration < 40; ++iteration) {
            z = s.Elastic + dt / 2 * rate;
            const auto f = Evaluate(p, s.Elastic + p.Theta * dt * rate, z, v);
            const T r1 = v + m.Coupling * f.Force - (bow_velocity - free_velocity), r2 = rate - f.Rate;
            out.Residual = abs(r1) > abs(r2) ? abs(r1) : abs(r2);
            if (out.Residual <= tolerance) break;
            const T rv = 1 - p.Stiffness * z * f.ResistanceVelocity;
            const T rz = -p.Stiffness * (f.Resistance + 2 * p.Theta * z * f.ResistanceState);
            const T a = 1 + m.Coupling * (p.ComplianceDamping * rv + p.FluidDamping), b = dt / 2 * m.Coupling * (p.Stiffness + p.ComplianceDamping * rz);
            const T c = -rv, d = 1 - dt / 2 * rz, det = a * d - b * c;
            v -= (d * r1 - b * r2) / det;
            rate -= (a * r2 - c * r1) / det;
            out.Iterations = iteration + 1;
        }
        const auto f = Evaluate(p, s.Elastic + p.Theta * dt * rate, s.Elastic + dt / 2 * rate, v);
        force = f.Force;
        dissipation = f.Dissipation;
        z = s.Elastic + dt * rate;
        out.InputPower = bow_velocity * force;
    }
    out.Failed = out.Residual > tolerance || !(out.Residual >= 0);
    for (unsigned i = 0; i < p.Modes; ++i) {
        const T mid = free[i] + m.Response[i] * force;
        s.Displacement[i] += dt * mid;
        s.Velocity[i] = 2 * mid - s.Velocity[i];
        dissipation += p.StringDamping * mid * mid;
        out.Displacement += m.Shape[i] * s.Displacement[i];
        out.Velocity += m.Shape[i] * s.Velocity[i];
    }
    if (hammer) s.HammerVelocity -= dt / p.HammerMass * force;
    s.Elastic = z;
    s.ContactVelocity = v;
    s.Force = force;
    s.Loss += dt * (dissipation - out.InputPower);
    out.Force = force;
    out.RelativeVelocity = v;
    out.Energy = Energy(m, s, hammer);
    out.Dissipation = dissipation;
    out.BalanceError = out.Energy - before + dt * (dissipation - out.InputPower);
    return out;
}
}
#undef FAL_THREAD
