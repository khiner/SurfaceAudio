#pragma once

#include "Agarwal.h"

namespace surface_audio::agarwal {

struct GpuImpulseParameters {
    uint32_t TapCount{}, FrameCount{}, SurfaceModeCount{}, ObjectModeCount{};
    float SampleRate{}, ObjectGain{};
};
static_assert(sizeof(GpuImpulseParameters) == 24);

struct GpuImpulseData {
    GpuImpulseParameters Parameters;
    // Concatenated Surface0, Surface1, Object arrays, retaining mode-major envelopes.
    std::vector<float> Frequencies, Amplitudes;
};

GpuImpulseData PrepareGpuImpulseData(const ImpulseResponses &responses, uint32_t frame_count);

struct GpuTrajectoryParameters {
    uint32_t Width{}, Height{}, FrameCount{}, Rolling{};
    float SpacingX{}, SpacingY{};
    float NormalMin{}, NormalMax{}, AlphaMin{}, AlphaMax{}, Exponent{}, ConstantAlpha{};
    float ReferenceAlpha{}, GaussianHalfWidth{}, GaussianSigmaRatio{};
    float Mass{}, Beta1{}, Beta2{};
    float Radius{}, Eccentricity{}, Stiffness{}, Dissipation{};
};
static_assert(sizeof(GpuTrajectoryParameters) == 88);

struct GpuTrajectoryData {
    GpuTrajectoryParameters Parameters;
    std::vector<float> Heights;
    // Left-edge slopes followed by top-edge slopes. Kernel buffer 6.
    std::vector<float> BoundarySlopes;
    // SoA planes: X, Y, VelocityX, VelocityY, NormalForce, each FrameCount floats.
    std::vector<float> Motions;
};

GpuTrajectoryData PrepareGpuTrajectoryData(const SurfaceGrid &surface, std::span<const Motion> motion, const ConstraintSettings &constraints, const ScrapingSettings &scraping, const RollingSettings *rolling = nullptr);
// Status 1 means invalid rolling penetration. Status 2 means nonfinite arithmetic.
void ValidateGpuTrajectoryStatus(std::span<const uint32_t> status);

} // namespace surface_audio::agarwal
