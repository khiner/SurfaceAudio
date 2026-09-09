#pragma once
#include "Gpu.h"
#include <array>

namespace surface_audio {
enum class SpectralMagnitudeScale : uint32_t { Decibels,
                                               NaturalLog,
                                               Linear };
struct SpectralLossOptions {
    float MagnitudeFloor{1e-6f}, HuberDelta{1};
    uint32_t HopDivisor{4};
    SpectralMagnitudeScale Scale{SpectralMagnitudeScale::Decibels};
};
struct SpectralResolutionGpu {
    uint32_t Size{}, Hop{}, Frames{};
    GpuBuffer Parameters, Target, Spectrum, Adjoint, FrameLoss, Window, Twiddles, Transform;
};
struct SpectralLossGpu {
    uint32_t Samples{}, SampleRate{};
    SpectralLossOptions Options;
    std::array<SpectralResolutionGpu, 4> Resolutions;
    GpuBuffer Gradient, Loss;
    GpuKernel Forward, FinishForward, CompareAdjoint, Overlap, Reduce;
};

// FFT sizes 4096, 1024, 256, 64 follow Agarwal section 2.2.2. Unspecified conventions: periodic Hann,
// centered zero padding, unnormalized one-sided FFT, sqrt(power+floor^2), and summed per-resolution mean Huber loss.
// Loss contains total then four resolution losses; Gradient is the full sample adjoint. Silence has finite loss/zero gradient.
SpectralLossGpu CreateSpectralLossGpu(Gpu &, std::span<const float> target, uint32_t sample_rate, SpectralLossOptions = {});
// Encode after BeginGpu. Read Loss and Gradient only after SubmitGpu/WaitGpu.
// Target transforms and allocations happen only in Create.
void EncodeSpectralLoss(Gpu &, const SpectralLossGpu &, GpuBuffer waveform);
} // namespace surface_audio
