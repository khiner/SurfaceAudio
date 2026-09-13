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

// Returns a workspace for four mean Huber spectral losses with centered periodic-Hann windows and unnormalized one-sided FFTs.
// Loss stores the total followed by resolution losses; Gradient stores the full sample adjoint.
SpectralLossGpu CreateSpectralLossGpu(Gpu &, std::span<const float> target, uint32_t sample_rate, SpectralLossOptions = {});
// Requires an active GPU batch; read Loss and Gradient after SubmitGpu/WaitGpu.
// Encoding reuses the allocated workspace and target transforms.
void EncodeSpectralLoss(Gpu &, const SpectralLossGpu &, GpuBuffer waveform);
}
