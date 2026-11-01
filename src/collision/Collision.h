#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define COLLISION_DEVICE device
#define COLLISION_THREAD thread
#define COLLISION_SHARED threadgroup
#else
#include "core/Gpu.h"
#include <cmath>
#include <filesystem>
#include <vector>
#define COLLISION_DEVICE
#define COLLISION_THREAD
#define COLLISION_SHARED
#endif

namespace surface_audio::collision {
enum GradientRule : unsigned { StrictPositive, MassBarrier, Nonnegative };
#ifdef __METAL_VERSION__
using metal::abs;
using metal::exp;
using metal::pow;
using metal::sqrt;
#else
using std::abs;
using std::exp;
using std::pow;
using std::sqrt;
#endif

template<typename T> struct Parameters {
    unsigned Dofs{}, Contacts{}, StiffnessEntries{}, LossEntries{}, ContactEntries{}, Smoothing{};
    T Dt{}, BowForce{}, BowSpeed{}, BowShape{100};
};
template<typename T> struct Sample {
    T Output{}, Energy{}, Force{}, Psi{}, Coordinate{}, Penetration{}, BowVelocity{}, BowForce{}, Dissipation{}, Power{}, Balance{};
};
template<typename T> inline unsigned StateSize(COLLISION_THREAD const Parameters<T> &p) {
    return 3 * p.Dofs + 3 * p.Contacts + p.Contacts * p.Contacts + 1;
}

inline void Synchronize() {
#ifdef __METAL_VERSION__
    metal::threadgroup_barrier(metal::mem_flags::mem_device | metal::mem_flags::mem_threadgroup);
#endif
}
template<typename T> inline T Sum(T value, COLLISION_SHARED T *shared, unsigned lane, unsigned width) {
    if (width == 1) return value;
    shared[lane] = value;
    Synchronize();
    for (unsigned stride = width / 2; stride; stride /= 2) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        Synchronize();
    }
    const T result = shared[0];
    Synchronize();
    return result;
}

template<typename T> inline Sample<T> Step(COLLISION_THREAD const Parameters<T> &p,
    COLLISION_DEVICE const T *data, COLLISION_DEVICE const unsigned *indices,
    COLLISION_DEVICE T *state, COLLISION_DEVICE unsigned *active, unsigned lane = 0, unsigned width = 1, COLLISION_SHARED T *shared = nullptr) {
    const unsigned n = p.Dofs, c = p.Contacts;
    const T dt = p.Dt;
    const auto mass = data, damping = mass + n, pickup = damping + n, bow = pickup + n;
    const auto barrier = bow + n, kappa = barrier + c, exponent = kappa + c, bilateral = exponent + c, rule = bilateral + c;
    const auto stiffness = rule + c, loss = stiffness + p.StiffnessEntries, contact = loss + p.LossEntries, gram = contact + p.ContactEntries;
    const auto kr = indices, kc = kr + n + 1, rr = kc + p.StiffnessEntries, rc = rr + n + 1, jr = rc + p.LossEntries, jc = jr + c + 1;
    auto q = state, delta = q + n, next = delta + n, psi = next + n, g = psi + c, lambda = g + c, matrix = lambda + c;
    const T coordinate = q[n - 1];
    T previous_energy = 0;
    if (lane == 0) for (unsigned a = 0; a < c; ++a) previous_energy += psi[a] * psi[a] / 2;
    for (unsigned i = lane; i < n; i += width) {
        const T velocity = delta[i] / dt;
        previous_energy += mass[i] * velocity * velocity / 2;
        for (unsigned a = kr[i]; a < kr[i + 1]; ++a) previous_energy += q[i] * stiffness[a] * (q[kc[a]] - delta[kc[a]]) / 2;
        for (unsigned a = rr[i]; a < rr[i + 1]; ++a) previous_energy -= velocity * loss[a] * delta[rc[a]] / 4;
        T f = 0, r = 0;
        for (unsigned a = kr[i]; a < kr[i + 1]; ++a) f += stiffness[a] * q[kc[a]];
        for (unsigned a = rr[i]; a < rr[i + 1]; ++a) r += loss[a] * delta[rc[a]];
        next[i] = ((mass[i] - dt * damping[i] / 2) * delta[i] - dt * dt * f - dt * r) / (mass[i] + dt * damping[i] / 2);
    }
    previous_energy = Sum(previous_energy, shared, lane, width);
    Synchronize();
    T bow_velocity = 0, bow_response = 0, bow_force = 0;
    T energy = 0, output = 0, dissipation = 0, penetration = 0, first_force = 0, old_psi = 0;
    if (lane == 0) {
        old_psi = c ? psi[0] : T(0);
        if (p.BowForce != 0) {
            for (unsigned i = 0; i < n; ++i) {
                bow_velocity += bow[i] * (next[i] + delta[i]) / (2 * dt);
                bow_response += bow[i] * bow[i] * dt / (2 * (mass[i] + dt * damping[i] / 2));
            }
            const T free_velocity = bow_velocity - p.BowSpeed, scale = sqrt(2 * p.BowShape) * exp(T(.5));
            T v = state[StateSize(p) - 1];
            for (unsigned iteration = 0; iteration < 100; ++iteration) {
                const T e = scale * exp(-p.BowShape * v * v);
                const T change = (v + bow_response * p.BowForce * v * e - free_velocity) / (1 + bow_response * p.BowForce * e * (1 - 2 * p.BowShape * v * v));
                const T following = v - change;
                if (following == v) break;
                v = following;
                if (abs(change) < T(1e-12)) break;
            }
            state[StateSize(p) - 1] = v;
            bow_force = -p.BowForce * scale * v * exp(-p.BowShape * v * v);
            bow_velocity = v + p.BowSpeed;
            for (unsigned i = 0; i < n; ++i) next[i] += dt * dt * bow[i] * bow_force / (mass[i] + dt * damping[i] / 2);
        }
        unsigned count = 0;
        for (unsigned a = 0; a < c; ++a) {
            T eta = barrier[a], before = barrier[a], free_change = 0;
            for (unsigned j = jr[a]; j < jr[a + 1]; ++j) {
                eta += contact[j] * q[jc[j]];
                before += contact[j] * (q[jc[j]] - delta[jc[j]]);
                free_change += contact[j] * (next[jc[j]] + delta[jc[j]]);
            }
            if (a == 0) penetration = eta;
            T gradient = 0;
            if (kappa[a] != 0) {
                if (eta > 0 || (rule[a] == T(Nonnegative) && eta == 0) || (bilateral[a] != 0 && eta != 0)) {
                    gradient = sqrt(T(.5) * kappa[a] * (exponent[a] + 1)) * pow(abs(eta), (exponent[a] - 1) / 2);
                    if (bilateral[a] == 0 && psi[a] < 0) gradient = -gradient;
                    if (bilateral[a] != 0 && eta < 0) gradient = -gradient;
                    if (rule[a] == T(MassBarrier) && gradient * before / 4 > psi[a]) gradient = 0;
                } else if (bilateral[a] == 0 && (eta < 0 || rule[a] == T(MassBarrier)) && psi[a] != 0 && free_change != 0) gradient = -2 * psi[a] / free_change;
            }
            g[a] = gradient;
            if (gradient != 0) {
                active[count] = a;
                lambda[count++] = psi[a] + gradient * free_change / 4;
            }
        }
        // Eliminate the mechanical coordinates and factor the symmetric contact-space system.
        for (unsigned a = 0; a < count; ++a) {
            for (unsigned b = 0; b <= a; ++b) {
                T v = (a == b ? T(1) : T(0)) + g[active[a]] * gram[active[a] * c + active[b]] * g[active[b]] / 4;
                for (unsigned k = 0; k < b; ++k) v -= matrix[a * c + k] * matrix[b * c + k];
                matrix[a * c + b] = a == b ? sqrt(v) : v / matrix[b * c + b];
            }
            for (unsigned b = 0; b < a; ++b) lambda[a] -= matrix[a * c + b] * lambda[b];
            lambda[a] /= matrix[a * c + a];
        }
        for (unsigned a = count; a-- > 0;) {
            for (unsigned b = a + 1; b < count; ++b) lambda[a] -= matrix[b * c + a] * lambda[b];
            lambda[a] /= matrix[a * c + a];
        }
        for (unsigned k = 0; k < count; ++k) {
            const unsigned a = active[k];
            const T force = g[a] * lambda[k];
            if (a == 0) first_force = force;
            for (unsigned j = jr[a]; j < jr[a + 1]; ++j) {
                const unsigned i = jc[j];
                next[i] -= dt * dt * contact[j] * force / (mass[i] + dt * damping[i] / 2);
            }
        }
        for (unsigned a = 0; a < c; ++a) {
            T change = 0;
            for (unsigned j = jr[a]; j < jr[a + 1]; ++j) change += contact[j] * (next[jc[j]] + delta[jc[j]]);
            psi[a] += g[a] * change / 2;
            energy += psi[a] * psi[a] / 2;
        }
    }
    Synchronize();
    for (unsigned i = lane; i < n; i += width) {
        const T velocity = next[i] / dt;
        energy += mass[i] * velocity * velocity / 2;
        const T centered = (next[i] + delta[i]) / (2 * dt);
        dissipation += damping[i] * centered * centered;
        for (unsigned a = rr[i]; a < rr[i + 1]; ++a) dissipation += centered * loss[a] * (next[rc[a]] + delta[rc[a]]) / (2 * dt);
        for (unsigned a = kr[i]; a < kr[i + 1]; ++a) energy += (q[i] + next[i]) * stiffness[a] * q[kc[a]] / 2;
        for (unsigned a = rr[i]; a < rr[i + 1]; ++a) energy -= velocity * loss[a] * next[rc[a]] / 4;
        output += pickup[i] * (p.Smoothing == 0 ? q[i] : (p.Smoothing == 1 ? q[i] + next[i] / 2 : q[i] + (next[i] - delta[i]) / 4));
    }
    energy = Sum(energy, shared, lane, width);
    output = Sum(output, shared, lane, width);
    dissipation = Sum(dissipation, shared, lane, width);
    const T power = bow_velocity * bow_force;
    const Sample<T> sample{output, energy, first_force, old_psi, coordinate, penetration, bow_velocity, bow_force, dissipation, power, energy - previous_energy - dt * (power - dissipation)};
    for (unsigned i = lane; i < n; i += width) { delta[i] = next[i]; q[i] += next[i]; }
    Synchronize();
    return sample;
}
#ifndef __METAL_VERSION__
template<typename T> struct Model {
    Parameters<T> Config;
    std::vector<T> Data, State;
    std::vector<unsigned> Indices;
};
Model<double> ReadModel(const std::filesystem::path &);
Model<float> FloatModel(const Model<double> &);
template<typename T> void Render(const Model<T> &, std::span<T> state, std::span<Sample<T>>);
struct GpuSource {
    Parameters<float> Config;
    unsigned Voices{}, Frames{}, Threads{};
    GpuBuffer Data, Indices, State, Active, Output;
    GpuKernel Kernel;
};
GpuSource CreateGpuSource(Gpu &, const Model<float> &, unsigned voices, unsigned frames);
void EncodeSource(Gpu &, const GpuSource &);
#endif
}
#undef COLLISION_DEVICE
#undef COLLISION_THREAD
#undef COLLISION_SHARED
