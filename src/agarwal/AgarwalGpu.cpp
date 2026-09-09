#include "AgarwalGpu.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace surface_audio::agarwal {

GpuImpulseData PrepareGpuImpulseData(const ImpulseResponses &responses, uint32_t frame_count) {
    Validate(responses);
    const size_t mode_count = 2 * responses.Surface0.Frequencies.size() + responses.Object.Frequencies.size();
    if (!frame_count || !mode_count || uint64_t(frame_count) * responses.TapCount > UINT32_MAX || uint64_t(mode_count) * responses.TapCount > UINT32_MAX) throw std::invalid_argument("Agarwal GPU IR dimensions exceed the 32-bit shader ABI");
    GpuImpulseData result{.Parameters = {responses.TapCount, frame_count, uint32_t(responses.Surface0.Frequencies.size()), uint32_t(responses.Object.Frequencies.size()), float(responses.SampleRate), float(responses.ObjectGain)}, .Frequencies = {}, .Amplitudes = {}};
    result.Frequencies.reserve(mode_count);
    result.Amplitudes.reserve(mode_count * responses.TapCount);
    for (const auto &endpoint : {responses.Surface0, responses.Surface1, responses.Object}) {
        for (double frequency : endpoint.Frequencies) result.Frequencies.push_back(float(frequency));
        for (double amplitude : endpoint.Amplitudes) {
            const auto value = float(amplitude);
            if (!std::isfinite(value) || value < std::numeric_limits<float>::min()) throw std::invalid_argument("Agarwal GPU modal amplitudes must be positive normal floats");
            result.Amplitudes.push_back(value);
        }
    }
    if (!std::isfinite(result.Parameters.SampleRate) || !std::isfinite(result.Parameters.ObjectGain)) throw std::invalid_argument("Agarwal GPU parameters exceed float range");
    return result;
}

GpuTrajectoryData PrepareGpuTrajectoryData(const SurfaceGrid &surface, std::span<const Motion> motion, const ConstraintSettings &constraints, const ScrapingSettings &scraping, const RollingSettings *rolling) {
    Validate(surface, constraints);
    if (motion.empty() || motion.size() > UINT32_MAX / 5 || surface.Heights.size() > UINT32_MAX) throw std::invalid_argument("Agarwal GPU trajectory dimensions exceed the 32-bit shader ABI");
    const auto convert = [](double value) {
        const auto result = float(value);
        if (!std::isfinite(result) || (value != 0 && std::abs(result) < std::numeric_limits<float>::min())) throw std::invalid_argument("Agarwal GPU inputs must be representable as normal floats");
        return result;
    };
    const RollingSettings ball = rolling ? *rolling : RollingSettings{};
    // Validate force parameters without preparing the surface.
    if (rolling) RollingForce({}, {.NormalForce = 0}, scraping, ball);
    else ScrapingForce({}, {.NormalForce = 0}, scraping);
    GpuTrajectoryData result{};
    result.Parameters = {
        surface.Width, surface.Height, uint32_t(motion.size()), rolling ? 1u : 0u,
        convert(surface.SpacingX), convert(surface.SpacingY),
        convert(constraints.NormalMin), convert(constraints.NormalMax), convert(constraints.AlphaMin), convert(constraints.AlphaMax), convert(constraints.Exponent), convert(constraints.ConstantAlpha),
        convert(constraints.ReferenceAlpha), convert(constraints.GaussianHalfWidth), convert(constraints.GaussianSigmaRatio),
        convert(scraping.Mass), convert(scraping.Beta1), convert(scraping.Beta2),
        convert(ball.Radius), convert(ball.Eccentricity), convert(ball.Stiffness), convert(ball.Dissipation)
    };
    const auto &params = result.Parameters;
    if ((constraints.NormalMin != constraints.NormalMax && params.NormalMin == params.NormalMax) || params.SpacingX * params.SpacingX < std::numeric_limits<float>::min() || params.SpacingY * params.SpacingY < std::numeric_limits<float>::min()) throw std::invalid_argument("Agarwal GPU spacing or normal-force range loses float precision");
    result.Heights.reserve(surface.Heights.size());
    for (double height : surface.Heights) result.Heights.push_back(convert(height));
    for (double slope : surface.LeftSlopes) result.BoundarySlopes.push_back(convert(slope));
    for (double slope : surface.TopSlopes) result.BoundarySlopes.push_back(convert(slope));
    result.Motions.resize(5 * motion.size());
    for (size_t i = 0; i < motion.size(); ++i) {
        const auto &sample = motion[i];
        if (!std::isfinite(sample.X) || !std::isfinite(sample.Y) || sample.X < 0 || sample.Y < 0 || sample.X > double(surface.Width - 1) * surface.SpacingX || sample.Y > double(surface.Height - 1) * surface.SpacingY || sample.NormalForce < 0 || (rolling && sample.VelocityY != 0)) throw std::invalid_argument("Invalid Agarwal GPU motion sample");
        result.Motions[i] = convert(sample.X);
        result.Motions[motion.size() + i] = convert(sample.Y);
        result.Motions[2 * motion.size() + i] = convert(sample.VelocityX);
        result.Motions[3 * motion.size() + i] = convert(sample.VelocityY);
        result.Motions[4 * motion.size() + i] = convert(sample.NormalForce);
    }
    return result;
}

void ValidateGpuTrajectoryStatus(std::span<const uint32_t> status) {
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 1) throw std::runtime_error("Agarwal GPU rolling penetration is negative at frame " + std::to_string(i));
        if (status[i]) throw std::runtime_error("Agarwal GPU nonfinite trajectory or force at frame " + std::to_string(i));
    }
}

} // namespace surface_audio::agarwal
