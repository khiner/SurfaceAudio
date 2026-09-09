#pragma once
#include "GpuMix.h"

namespace surface_audio {
inline constexpr float EndpointMixMinimumLogAmplitude = -30, EndpointMixMaximumLogAmplitude = 20;

struct GpuEndpointMix {
    uint32_t Modes, Frames, GradientGroups;
    GpuBuffer Parameters, Basis, Location, Block, Components, PartialGradient, Gradient, Output;
    GpuMix Mix;
    GpuKernel Synthesize, Differentiate, Reduce;
};

// y[n] = sum_m basis[m,n] * exp((1-location[n])*u[m] + location[n]*v[m]).
// Mode-major basis; parameters contain endpoint-0 logs then endpoint-1 logs. Location lies in [0,1].
GpuEndpointMix CreateGpuEndpointMix(Gpu &, std::span<const float> basis, std::span<const float> location, std::span<const float> log_parameters);
// Caller owns Begin/Submit/Wait. Allocation-free encoding requires finite, bounded parameters.
// Differentiate after synthesis with the same parameters and an adjoint covering all Frames, including tail.
void EncodeEndpointMix(Gpu &, const GpuEndpointMix &);
void EncodeEndpointMixGradient(Gpu &, const GpuEndpointMix &, GpuBuffer sample_adjoint);
} // namespace surface_audio
