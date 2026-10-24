#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define REN_THREAD thread
#define REN_DEVICE const device
#else
#include "core/Gpu.h"
#include <cmath>
#define REN_THREAD
#define REN_DEVICE const
#endif

namespace surface_audio::ren {
struct Normal {
    float X{}, Y{}, Z{1};
};
struct Parameters {
    float SampleRate{44100}, Mass{.01f}, RoughnessGain{.1f}, RoughnessSpacing{.00002f};
    unsigned Width{}, Height{}, TableSize{};
};
// Coordinates are periodic UVs; velocities and normals use the surface tangent frame (N = +Z).
struct Contact {
    float U{}, V{}, VelocityX{}, VelocityY{}, NormalVelocity{}, NormalForce{}, MacroImpulse{};
    unsigned Overlap{};
};
struct State {
    float Momentum{}, Phase{}, PhaseError{};
    unsigned Lasting{};
#ifndef __METAL_VERSION__
    bool operator==(const State &) const = default;
#endif
};
struct Impulses {
    float Macro{}, Meso{}, Micro{};
    float Total() const { return Macro + Meso + Micro; }
};

inline Impulses Step(REN_THREAD const Parameters &p, REN_THREAD State &s, REN_THREAD const Contact &c, REN_DEVICE Normal *normals, REN_DEVICE float *roughness) {
#ifdef __METAL_VERSION__
    using namespace metal;
#else
    using std::floor;
    using std::sqrt;
#endif
    Impulses result;
    const bool contacting = c.Overlap && c.NormalVelocity < 0;
    if (contacting) result.Macro = c.MacroImpulse;
    const float speed = sqrt(c.VelocityX * c.VelocityX + c.VelocityY * c.VelocityY);
    if (!contacting || speed == 0) {
        s.Lasting = 0;
        return result;
    }
    const unsigned x = unsigned(floor((c.U - floor(c.U)) * p.Width)) % p.Width;
    const unsigned y = unsigned(floor((c.V - floor(c.V)) * p.Height)) % p.Height;
    const auto n = normals[y * p.Width + x];
    const float momentum = -p.Mass * (c.VelocityX * n.X + c.VelocityY * n.Y) / n.Z;
    // Section 5.2 uses p_N(i) - p_N(j) for a transition from texel i to texel j.
    if (s.Lasting) result.Meso = s.Momentum - momentum;
    s.Momentum = momentum;
    s.Lasting = 1;
    const unsigned first = unsigned(s.Phase);
    const float fraction = s.Phase - first;
    const float value = roughness[first] + fraction * (roughness[(first + 1) % p.TableSize] - roughness[first]);
    result.Micro = p.RoughnessGain * c.NormalForce * value / p.SampleRate;
    const float increment = speed / (p.SampleRate * p.RoughnessSpacing) - s.PhaseError;
    const float phase = s.Phase + increment;
    s.PhaseError = (phase - s.Phase) - increment;
    s.Phase = phase - floor(phase / p.TableSize) * p.TableSize;
    return result;
}

#ifndef __METAL_VERSION__
void Validate(const Parameters &, std::span<const Normal>, std::span<const float> roughness);
void ValidateContacts(std::span<const Contact>);
void Render(const Parameters &, State &, std::span<const Normal>, std::span<const float> roughness, std::span<const Contact>, std::span<Impulses>);

struct GpuSource {
    unsigned Voices{}, Frames{};
    GpuBuffer Config, States, Normals, Roughness, Contacts, Output;
    GpuKernel Kernel;
};
// Voices share a surface and wavetable; contact and output buffers are voice-major.
GpuSource CreateGpuSource(Gpu &, const Parameters &, std::span<const Normal>, std::span<const float> roughness, unsigned voices, unsigned frames);
// Upload validated contacts and set states before encoding; states persist across dispatches.
void EncodeSource(Gpu &, const GpuSource &);
#endif
}

#undef REN_THREAD
#undef REN_DEVICE
