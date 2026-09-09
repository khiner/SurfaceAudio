#pragma once
#include "core/GpuMix.h"
#include <array>

namespace surface_audio::agarwal {
inline constexpr float ContactFitMinimumDecay = .0001f, ContactFitMaximumDecay = .25f;
inline constexpr float ContactFitMinimumLogAmplitude = -30, ContactFitMaximumLogAmplitude = 20;

// Interior float log bounds keep exp(log_tau) within the physical interval after text round trips.
std::array<float, 2> ContactFitLogDecayBounds();

struct ContactFitMode {
    float Frequency, Decay, Amplitude;
};
struct ContactFitGpu {
    uint32_t Modes, ForceFrames, Frames, Taps, SampleRate, GradientGroups;
    GpuBuffer Parameters, Force, Frequencies, Block, ModeOutput, ModeDecayDerivative, PartialGradient, Gradient, Output;
    GpuMix Mix;
    GpuKernel Synthesize, Differentiate, Reduce;
};

// Fixed finite impulse response h[k] = sum A exp(-k/(rate*tau)) sin(2*pi*f*k/rate).
// Parameters: log amplitudes then log decay seconds; frequencies/force stay fixed. Full forceFrames+taps-1 output, unnormalized.
ContactFitGpu CreateContactFitGpu(Gpu &, std::span<const float> force, std::span<const ContactFitMode>, uint32_t sample_rate, uint32_t taps);
// Parameters must stay finite and within the named amplitude/decay bounds.
// Caller owns Begin/Submit/Wait. Encoding reuses all buffers and never waits.
void EncodeContactFit(Gpu &, const ContactFitGpu &);
void EncodeContactFitGradient(Gpu &, const ContactFitGpu &, GpuBuffer sample_adjoint);
} // namespace surface_audio::agarwal
