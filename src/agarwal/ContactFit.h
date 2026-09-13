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

// Parameters contain log amplitudes followed by log decay seconds, with fixed frequencies and force.
// Returns a workspace for forceFrames+taps-1 unscaled output samples.
ContactFitGpu CreateContactFitGpu(Gpu &, std::span<const float> force, std::span<const ContactFitMode>, uint32_t sample_rate, uint32_t taps);
// Requires finite, bounded parameters and a caller-managed GPU batch; encoding is allocation-free.
void EncodeContactFit(Gpu &, const ContactFitGpu &);
void EncodeContactFitGradient(Gpu &, const ContactFitGpu &, GpuBuffer sample_adjoint);
}
