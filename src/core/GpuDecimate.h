#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
// Planar channels, with output n centered exactly on input n * Factor.
// FIR cutoff .475/Factor cycles/input sample, radius 64*Factor, Kaiser beta 10.
// Boundary assumption: constant endpoint extension.
// Factors are integers in [1, 128]. Factor 1 copies the input exactly. Buffers and kernel belong to the GPU context.
struct GpuDecimatePlan {
    uint32_t InputFrames{}, OutputFrames{}, Channels{}, Factor{};
    GpuBuffer Parameters{}, Coefficients{};
    GpuKernel Kernel{};
};

// FP64, symmetric, unit-sum coefficients.
std::vector<double> KaiserDecimateCoefficients(uint32_t factor);
GpuDecimatePlan CreateDecimatePlan(Gpu &, uint32_t input_frames, uint32_t channels, uint32_t factor);
// Records into an active GPU batch. No allocation, CPU input readback or wait.
// Exact planar extents and nonoverlapping GPU ranges required; adjacent views of one allocation are allowed.
void DispatchDecimateGpu(Gpu &, const GpuDecimatePlan &, GpuBuffer input, GpuBuffer output);
} // namespace surface_audio
