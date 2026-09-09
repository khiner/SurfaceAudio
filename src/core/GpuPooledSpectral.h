#pragma once
#include "GpuSpectral.h"

namespace surface_audio {
struct SpectralPoolBand {
    float LowerHz{}, UpperHz{};
};
struct SpectralPoolOptions {
    float Weight{}, RelativeFloor{.1f};
    uint32_t Resolution{}, EventSamples{}, TimePoolSamples{};
};
struct SpectralPoolRegion {
    uint32_t FirstBin{}, EndBin{}, FirstFrame{}, EndFrame{}, Band{};
};
struct GpuPooledSpectral {
    SpectralPoolOptions Options;
    uint32_t Pools{}, Bands{};
    GpuBuffer Parameters{}, Regions{}, Membership{}, TargetScale{}, Energy{}, Coefficients{}, Adjoint{}, Loss{};
    GpuKernel ReduceEnergy{}, Compare{}, Differentiate{}, Add{};
};

// Optional inference objective, absent from the paper. Unnormalized one-sided spectral RMS, with equal bin weights.
// Bands are [LowerHz, UpperHz), including at Nyquist. Frame centers in [0, EventSamples) form integer sample spans.
// Empty spans are omitted; centered windows may overlap the tail.
// R=sqrt(mean|X|^2+epsilon^2), T likewise; epsilon=MagnitudeFloor. Add Weight*mean_pools(.5*((R-T)/D)^2).
// Fixed D=max(target whole-event band RMS, RelativeFloor*maximum target band RMS, epsilon), weighting each frame/bin equally.
// Zero weight allocates/encodes nothing. Positive weight requires linear base loss, including its complete-tail penalty.
GpuPooledSpectral CreatePooledSpectral(Gpu &, const SpectralLossGpu &, std::span<const SpectralPoolBand>, SpectralPoolOptions);
// After EncodeSpectralLoss in the same batch: add to spectral.Gradient/Loss[0], preserving the four base losses.
// pool.Loss contains the weighted total followed by cell terms.
void EncodePooledSpectral(Gpu &, const SpectralLossGpu &, const GpuPooledSpectral &);
} // namespace surface_audio
