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

// Returns an optional pooled-RMS loss workspace over [LowerHz, UpperHz) and frame centers in [0, EventSamples).
// Positive weight requires linear base loss with complete-tail coverage; zero weight disables allocation and encoding.
// Centered windows may overlap the tail.
GpuPooledSpectral CreatePooledSpectral(Gpu &, const SpectralLossGpu &, std::span<const SpectralPoolBand>, SpectralPoolOptions);
// Encode after EncodeSpectralLoss in the same batch to add the adjoint and total loss while preserving base resolution losses.
// pool.Loss stores the weighted total followed by cell losses.
void EncodePooledSpectral(Gpu &, const SpectralLossGpu &, const GpuPooledSpectral &);
}
