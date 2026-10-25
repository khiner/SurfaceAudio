#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define BARRIER_THREAD thread
#else
#include "core/Gpu.h"
#include <cmath>
#include <vector>
#define BARRIER_THREAD
#endif

namespace surface_audio::barrier {
template<typename T> struct Force {
    T X{}, Y{};
};
template<typename T> struct Parameters {
    T Dt{}, Mass{}, Stiffness{}, Damping{}, Barrier{}, Kappa{}, Exponent{}, Restitution{}, Regularization{}, Friction{};
    T Difference{}, B{}, C{}, Epsilon{};
    unsigned Implicit{};
};
template<typename T> struct State {
    T X{}, Y{}, DeltaX{}, DeltaY{}, Psi{}, Eta{1};
#ifndef __METAL_VERSION__
    bool operator==(const State &) const = default;
#endif
};
template<typename T> struct Sample {
    T X{}, Y{}, Vx{}, Vy{}, NormalForce{}, TangentialNormalForce{}, FrictionForce{};
    T Energy{}, LinearLoss{}, ContactLoss{}, ExternalPower{}, Psi{}, BalanceResidual{}, Iterations{}, RawPsi{};
};

#ifdef __METAL_VERSION__
using metal::abs;
using metal::max;
using metal::min;
using metal::pow;
using metal::sqrt;
#else
using std::abs;
using std::max;
using std::min;
using std::pow;
using std::sqrt;
#endif

template<typename T> inline T Damping(T velocity, T eta) { return velocity * eta < -1 ? -1 / velocity : eta; }
template<typename T> inline T Potential(BARRIER_THREAD const Parameters<T> &p, T w) { return p.Kappa / (p.Exponent + 1) * pow(max(w, T(0)), p.Exponent + 1); }

template<typename T> inline State<T> MakeState(BARRIER_THREAD const Parameters<T> &p, T x = 0, T y = 0, T vx = 0, T vy = 0) {
    return {x, y, vx * p.Dt, vy * p.Dt, sqrt(2 * Potential(p, x - vx * p.Dt / 2 - p.Barrier)), T(1)};
}

template<typename T> inline T LinearEnergy(BARRIER_THREAD const Parameters<T> &p, T x, T y, T dx, T dy) {
    const T mx = x + dx / 2, my = y + dy / 2;
    return p.Mass * (dx * dx + dy * dy) / (2 * p.Dt * p.Dt) + p.Stiffness * (mx * mx + my * my) / 2;
}

template<typename T> inline T DiscreteGradient(BARRIER_THREAD const Parameters<T> &p, T w, T step) {
    if (step == 0) return p.Kappa * pow(max(w, T(0)), p.Exponent);
    if (w > 0) {
        const T q = step / (2 * w), a = p.Exponent;
        // The series avoids cancellation in the divided difference of the contact potential.
        if (abs(q) < T(.001)) return p.Kappa * pow(w, a) * (1 + a * q * (T(.5) + (a - 1) * q * (T(1. / 6) + (a - 2) * q * (T(1. / 24) + (a - 3) * q / 120))));
    }
    return 2 * (Potential(p, w + step / 2) - Potential(p, w)) / step;
}

template<typename T> inline Sample<T> Step(BARRIER_THREAD const Parameters<T> &p, BARRIER_THREAD State<T> &s, BARRIER_THREAD const Force<T> &force) {
    const T zx = (p.Difference * s.X - p.B * s.DeltaX - p.C * force.X) / 2;
    const T zy = (p.Difference * s.Y - p.B * s.DeltaY - p.C * force.Y) / 2;
    const T w = s.X - p.Barrier, free_step = -2 * zx, impact_velocity = (free_step - s.DeltaX) / p.Dt;
    if (w <= 0 && w + free_step - s.DeltaX > 0)
        s.Eta = (T(.2) * p.Exponent + T(1.3)) * (1 - p.Restitution) / ((p.Restitution + p.Regularization * p.Dt * p.Dt) * (impact_velocity + p.Regularization * p.Dt * p.Dt));
    const T previous_potential = p.Implicit ? Potential(p, w - s.DeltaX / 2) : s.Psi * s.Psi / 2;
    const T previous_energy = LinearEnergy(p, s.X, s.Y, -s.DeltaX, -s.DeltaY) + previous_potential;
    T sx = free_step, psi = s.Psi, raw_psi = psi, normal{}, friction_normal{}, normal_loss{}, contact_potential{};
    unsigned iterations = 0;
    if (p.Implicit) {
        const T midpoint = w - s.DeltaX / 2;
        const T initial_force = max(T(0), 1 + s.Eta * free_step / (2 * p.Dt)) * DiscreteGradient(p, midpoint, free_step);
        T upper = free_step, lower = free_step - p.C * initial_force;
        // Monotonicity bounds the root between the free step and one force-corrected step.
        for (; iterations < 80 && lower < upper; ++iterations) {
            sx = (lower + upper) / 2;
            if (sx == lower || sx == upper) break;
            const T residual = sx - free_step + p.C * max(T(0), 1 + s.Eta * sx / (2 * p.Dt)) * DiscreteGradient(p, midpoint, sx);
            if (abs(residual) <= 2 * p.Epsilon * (abs(sx) + abs(free_step))) break;
            if (residual > 0) upper = sx;
            else lower = sx;
        }
        const T gradient = DiscreteGradient(p, midpoint, sx), velocity = sx / (2 * p.Dt);
        normal = friction_normal = -max(T(0), 1 + s.Eta * velocity) * gradient;
        normal_loss = velocity * velocity * Damping(velocity, s.Eta) * gradient;
        contact_potential = Potential(p, midpoint + sx / 2);
        raw_psi = psi = sqrt(2 * contact_potential);
    } else {
        const T h = Damping(s.DeltaX / p.Dt, s.Eta), gamma = h * psi / (2 * p.Dt);
        const T lambda = zx - gamma * p.C * psi, radical = sqrt(lambda * lambda + p.C * psi * psi);
        const T denominator = sqrt(p.Epsilon * p.C * lambda * lambda + (p.C * psi) * (p.C * psi));
        const T numerator = lambda > 0 ? p.C * psi * psi / (radical + lambda) : radical - lambda;
        const T bound = denominator > 0 ? 2 * numerator / denominator : T(0);
        const T g = w > 0 && p.Kappa > 0 ? min(sqrt(T(.5) * (p.Exponent + 1) * p.Kappa * pow(w, p.Exponent - 1)), bound) : (w <= 0 && w - s.DeltaX > 0 ? bound : T(0));
        sx = -(2 * zx + p.C * psi * g) / (1 + p.C * gamma * g + p.C * g * g / 4);
        raw_psi = psi + g * sx / 2;
        // The author's optional roundoff guard enforces the nonnegative auxiliary-energy domain.
        psi = max(T(0), raw_psi);
        const T velocity = sx / (2 * p.Dt), restoring = -(psi + s.Psi) * g / 2;
        normal = restoring - h * g * velocity * s.Psi;
        friction_normal = restoring * max(T(0), 1 + velocity * s.Eta);
        normal_loss = velocity * velocity * g * h * s.Psi;
        contact_potential = psi * psi / 2;
    }
    const T threshold = -p.C * friction_normal * p.Friction;
    const T sy = zy < -threshold / 2 ? -threshold - 2 * zy : (zy > threshold / 2 ? threshold - 2 * zy : T(0));
    const T friction = sy == 0 ? 2 * zy / p.C : (sy > 0 ? friction_normal * p.Friction : -friction_normal * p.Friction);
    const T vx = sx / (2 * p.Dt), vy = sy / (2 * p.Dt), dx = sx - s.DeltaX, dy = sy - s.DeltaY;
    const T energy = LinearEnergy(p, s.X, s.Y, dx, dy) + contact_potential;
    const T linear_loss = p.Damping * (vx * vx + vy * vy), contact_loss = normal_loss - friction * vy;
    const T power = vx * force.X + vy * force.Y;
    const Sample<T> output{s.X, s.Y, vx, vy, normal, friction_normal, friction, energy, linear_loss, contact_loss, power, psi, energy - previous_energy - p.Dt * (power - linear_loss - contact_loss), T(iterations), raw_psi};
    s = {s.X + dx, s.Y + dy, dx, dy, psi, s.Eta};
    return output;
}

#ifndef __METAL_VERSION__
struct Controls {
    double SampleRate{44100}, Mass{.001}, Stiffness{1000}, Damping{.002}, Barrier{}, Kappa{1e7}, Exponent{1.25};
    double Restitution{.1}, MaxVelocity{.5}, Regularization{1e7}, Friction{1};
    bool CapStiffness{true}, ModalCorrection{true}, Implicit{};
};
template<typename T> Parameters<T> MakeParameters(const Controls &);
template<typename T> void Render(const Parameters<T> &, State<T> &, std::span<const Force<T>>, std::span<Sample<T>>);

struct GpuSource {
    unsigned Voices{}, Frames{};
    GpuBuffer Config, States, Input, Output;
    GpuKernel Kernel;
};
GpuSource CreateGpuSource(Gpu &, std::span<const Parameters<float>>, unsigned frames);
void EncodeSource(Gpu &, const GpuSource &);
#endif
}

#undef BARRIER_THREAD
