#pragma once

#include "core/ErbNoise.h"

#include <array>
#include <vector>

namespace surface_audio::agarwal {

inline constexpr uint32_t ResponseModeCount = 10, ResponseParameterCount = 50;
// [frequency Hz, mode amplitude dB, mode RT60 sec, noise amplitude dB, noise RT60 sec], ten each.
using ResponseParameters = std::array<float, ResponseParameterCount>;
inline constexpr uint32_t ObjectResponseNoiseCount = 20, ObjectResponseParameterCount = 70;
using ObjectResponseParameters = std::array<float, ObjectResponseParameterCount>;

using ResponseNoiseSettings = ErbNoiseSettings;

// Seeded noise includes both FIR margins, avoiding padding transients. Hold returned band-major noise fixed during fitting.
std::vector<float> CreateResponseNoise(Gpu &, uint32_t frames, double sample_rate, const ResponseNoiseSettings & = {}, uint32_t noise_bands = 10);
std::vector<double> ResponseNoiseReference(uint32_t frames, double sample_rate, const ResponseNoiseSettings & = {}, uint32_t noise_bands = 10);

struct ResponseGpu {
    uint32_t Frames{}, GradientGroups{}, NoiseBands{10}, ParameterCount{50};
    float SampleRate{};
    GpuBuffer Parameters, Output, Gradient, Noise, Block, PartialGradient;
    GpuKernel Synthesize, Differentiate, Reduce;
};

ResponseGpu CreateResponseGpu(Gpu &, uint32_t frames, float sample_rate, std::span<const float> noise_bandmajor, uint32_t noise_bands = 10);
// Caller owns Begin/Submit/Wait. Parameters must be finite, frequencies in [0, Nyquist], RT60 strictly positive.
void EncodeResponse(Gpu &, const ResponseGpu &);
void EncodeResponseGradient(Gpu &, const ResponseGpu &, GpuBuffer sample_adjoint);

// Raw Eq. 2/3 at t=frame/sample_rate, with no output normalization or parameter transformation.
void EvaluateResponse(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise_bandmajor, std::span<double> output);
void EvaluateResponseGradient(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise_bandmajor, std::span<const double> sample_adjoint, std::span<double> gradient);

} // namespace surface_audio::agarwal
