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
// Requires a mode-major basis, endpoint-0 logs followed by endpoint-1 logs, and location in [0,1].
GpuEndpointMix CreateGpuEndpointMix(Gpu &, std::span<const float> basis, std::span<const float> location, std::span<const float> log_parameters);
// Requires finite, bounded parameters and a caller-managed GPU batch; encoding is allocation-free.
// Differentiate after synthesis with the same parameters and an adjoint covering all Frames, including tail.
void EncodeEndpointMix(Gpu &, const GpuEndpointMix &);
void EncodeEndpointMixGradient(Gpu &, const GpuEndpointMix &, GpuBuffer sample_adjoint);
}
