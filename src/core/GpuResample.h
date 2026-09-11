#pragma once
#include "core/Gpu.h"
#include <vector>

namespace surface_audio {
struct FirResampleJob {
    uint32_t InputOffset{}, InputFrames{}, OutputOffset{}, OutputFrames{}, TapOffset{}, Taps{};
    uint32_t Upsample{1}, Downsample{1};
    int32_t Delay{};
};
struct FirResampleGpu {
    uint32_t Count{}, MaximumFrames{};
    GpuBuffer Input, Taps, Jobs, Output;
    GpuKernel Kernel;
};
// Requires disjoint output ranges and extends inputs with zeros.
FirResampleGpu CreateFirResampleGpu(Gpu &, std::span<const float> input, std::span<const float> taps, std::span<const FirResampleJob>);
void EncodeFirResample(Gpu &, const FirResampleGpu &);
std::vector<float> ResampleFirGpu(Gpu &, std::span<const float> input, std::span<const float> taps, std::span<const FirResampleJob>);
}
