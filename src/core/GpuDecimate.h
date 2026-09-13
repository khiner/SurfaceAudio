#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
// Planar channels use constant endpoint extension and output n centered on input n * Factor.
// Factor is in [1, 128], with exact copying at 1; the GPU context owns buffers and kernel.
struct GpuDecimatePlan {
    uint32_t InputFrames{}, OutputFrames{}, Channels{}, Factor{};
    GpuBuffer Parameters{}, Coefficients{};
    GpuKernel Kernel{};
};

// FP64, symmetric, unit-sum coefficients.
std::vector<double> KaiserDecimateCoefficients(uint32_t factor);
GpuDecimatePlan CreateDecimatePlan(Gpu &, uint32_t input_frames, uint32_t channels, uint32_t factor);
// Requires an active GPU batch, exact planar extents and disjoint GPU ranges, including views of one allocation.
// Dispatch is allocation-free and returns before completion.
void DispatchDecimateGpu(Gpu &, const GpuDecimatePlan &, GpuBuffer input, GpuBuffer output);
}
