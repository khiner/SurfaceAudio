#pragma once
#include "Gpu.h"

namespace surface_audio {
struct MixBlock {
    uint32_t Voices{}, Frames{}, Stride{1}, Field{};
    float Gain{1};
};
struct GpuMix {
    MixBlock Block;
    GpuBuffer Parameters, Output;
    GpuKernel Kernel;
};
GpuMix CreateGpuMix(Gpu &, MixBlock);
void EncodeMix(Gpu &, const GpuMix &, GpuBuffer input);
} // namespace surface_audio
