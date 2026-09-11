#pragma once
#include "core/Gpu.h"

namespace surface_audio {
struct FftGpu {
    uint32_t Size{}, Count{};
    GpuBuffer Output, Scratch, Twiddles;
    GpuKernel Local, Stage;
};
// Requires record-major interleaved complex FP32 values and a power-of-two size.
FftGpu CreateFftGpu(Gpu &, uint32_t size, uint32_t count = 1);
// Requires an open GPU batch and input distinct from the plan's buffers.
// Inverse output is divided by size.
void EncodeFftGpu(Gpu &, const FftGpu &, GpuBuffer input, bool inverse = false);
}
