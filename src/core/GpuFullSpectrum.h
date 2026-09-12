#pragma once
#include "GpuFft.h"
#include "GpuSpectral.h"

namespace surface_audio {
struct FullSpectrumGpu {
    uint32_t Samples{}, Size{};
    FftGpu Fft;
    GpuBuffer Parameters, Input, Target, BinLoss, Loss;
    GpuKernel Pack, Compare, Accumulate, Reduce;
};
// Requires 1..2^22 finite samples.
FullSpectrumGpu CreateFullSpectrumGpu(Gpu &, std::span<const float> target, SpectralLossOptions);
// Adds its sample adjoint and scalar loss to the supplied buffers after the base objective is encoded.
void EncodeFullSpectrumGpu(Gpu &, const FullSpectrumGpu &, GpuBuffer waveform, GpuBuffer gradient, GpuBuffer loss);
}
