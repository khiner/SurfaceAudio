#include "Response.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace surface_audio::agarwal {
namespace {
constexpr double Tau = 2 * std::numbers::pi;
constexpr uint32_t GradientLanes = 256;
struct ResponseBlock {
    uint32_t Frames, Groups, NoiseBands;
    float SampleRate;
};
void ValidateDimensions(uint32_t frames, double sample_rate, uint32_t bands) {
    if ((bands != 10 && bands != 20) || !frames || frames > std::numeric_limits<uint32_t>::max() / bands || !std::isfinite(sample_rate) || sample_rate <= 0) throw std::invalid_argument("Invalid response dimensions");
}
void Validate(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise) {
    const uint32_t bands = parameters.size() == 50 ? 10 : parameters.size() == 70 ? 20 : 0;
    ValidateDimensions(frames, sample_rate, bands);
    if (noise.size() != size_t(frames) * bands) throw std::invalid_argument("Invalid response parameter or noise dimensions");
    for (double value : parameters)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite response parameter");
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        if (parameters[mode] < 0 || parameters[mode] > sample_rate / 2 || parameters[20 + mode] <= 0) throw std::invalid_argument("Invalid response frequency or RT60");
    }
    for (uint32_t band = 0; band < bands; ++band)
        if (parameters[30 + bands + band] <= 0) throw std::invalid_argument("Invalid response noise RT60");
}
} // namespace

std::vector<float> CreateResponseNoise(Gpu &gpu, uint32_t frames, double sample_rate, const ResponseNoiseSettings &settings, uint32_t bands) {
    return CreateErbNoise(gpu, bands, frames, sample_rate, settings);
}

std::vector<double> ResponseNoiseReference(uint32_t frames, double sample_rate, const ResponseNoiseSettings &settings, uint32_t bands) {
    return ErbNoiseReference(bands, frames, sample_rate, settings);
}

ResponseGpu CreateResponseGpu(Gpu &gpu, uint32_t frames, float sample_rate, std::span<const float> noise, uint32_t bands) {
    ValidateDimensions(frames, sample_rate, bands);
    if (noise.size() != size_t(frames) * bands || !std::ranges::all_of(noise, [](float x) { return std::isfinite(x); })) throw std::invalid_argument("Invalid response noise");
    const uint32_t groups = (frames + GradientLanes - 1) / GradientLanes;
    const uint32_t count = 30 + 2 * bands;
    std::vector<float> initial(count);
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        initial[mode] = sample_rate * float(mode + 1) / 24;
        initial[10 + mode] = -60;
        initial[20 + mode] = .1f;
    }
    for (uint32_t band = 0; band < bands; ++band) {
        initial[30 + band] = -60;
        initial[30 + bands + band] = .1f;
    }
    return {
        .Frames = frames,
        .GradientGroups = groups,
        .NoiseBands = bands,
        .ParameterCount = count,
        .SampleRate = sample_rate,
        .Parameters = Upload<float>(gpu, initial),
        .Output = CreateBuffer(gpu, size_t(frames) * sizeof(float)),
        .Gradient = CreateBuffer(gpu, count * sizeof(float)),
        .Noise = Upload<float>(gpu, noise),
        .Block = Upload(gpu, ResponseBlock{frames, groups, bands, sample_rate}),
        .PartialGradient = CreateBuffer(gpu, size_t(groups) * count * sizeof(float)),
        .Synthesize = CreateKernel(gpu, "ResponseSynthesize"),
        .Differentiate = CreateKernel(gpu, "ResponseDifferentiate"),
        .Reduce = CreateKernel(gpu, "ResponseReduce")
    };
}

void EncodeResponse(Gpu &gpu, const ResponseGpu &state) {
    const std::array bindings{GpuBinding{state.Block, 0}, GpuBinding{state.Parameters, 1}, GpuBinding{state.Noise, 2}, GpuBinding{state.Output, 3}};
    DispatchGpu(gpu, state.Synthesize, bindings, {state.Frames, 1, 1});
}

void EncodeResponseGradient(Gpu &gpu, const ResponseGpu &state, GpuBuffer sample_adjoint) {
    if (sample_adjoint.Size != size_t(state.Frames) * sizeof(float)) throw std::invalid_argument("Invalid response sample adjoint dimensions");
    const std::array bindings{GpuBinding{state.Block, 0}, GpuBinding{state.Parameters, 1}, GpuBinding{state.Noise, 2}, GpuBinding{sample_adjoint, 3}, GpuBinding{state.PartialGradient, 4}};
    DispatchGroupsGpu(gpu, state.Differentiate, bindings, {state.GradientGroups, state.ParameterCount, 1}, {GradientLanes, 1, 1});
    const std::array reduce_bindings{GpuBinding{state.Block, 0}, GpuBinding{state.PartialGradient, 1}, GpuBinding{state.Gradient, 2}};
    DispatchGroupsGpu(gpu, state.Reduce, reduce_bindings, {state.ParameterCount, 1, 1}, {GradientLanes, 1, 1});
}

void EvaluateResponse(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise, std::span<double> output) {
    Validate(parameters, frames, sample_rate, noise);
    const uint32_t bands = uint32_t((parameters.size() - 30) / 2);
    if (output.size() != frames) throw std::invalid_argument("Invalid response output dimensions");
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double t = frame / sample_rate;
        double sum = 0;
        for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
            sum += std::pow(10., parameters[10 + mode] / 20 - 3 * t / parameters[20 + mode]) * std::sin(Tau * parameters[mode] * t);
            sum += std::pow(10., parameters[30 + mode] / 20 - 3 * t / parameters[30 + bands + mode]) * noise[size_t(mode) * frames + frame];
        }
        for (uint32_t band = 10; band < bands; ++band)
            sum += std::pow(10., parameters[30 + band] / 20 - 3 * t / parameters[30 + bands + band]) * noise[size_t(band) * frames + frame];
        output[frame] = sum;
    }
}

void EvaluateResponseGradient(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise, std::span<const double> sample_adjoint, std::span<double> gradient) {
    Validate(parameters, frames, sample_rate, noise);
    const uint32_t bands = uint32_t((parameters.size() - 30) / 2);
    if (sample_adjoint.size() != frames || gradient.size() != parameters.size()) throw std::invalid_argument("Invalid response gradient dimensions");
    std::ranges::fill(gradient, 0);
    const double log10 = std::log(10.);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double t = frame / sample_rate, adjoint = sample_adjoint[frame];
        for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
            const double rt = parameters[20 + mode];
            const double envelope = std::pow(10., parameters[10 + mode] / 20 - 3 * t / rt);
            const double value = envelope * std::sin(Tau * parameters[mode] * t);
            gradient[mode] += adjoint * envelope * Tau * t * std::cos(Tau * parameters[mode] * t);
            gradient[10 + mode] += adjoint * value * log10 / 20;
            gradient[20 + mode] += adjoint * value * log10 * 3 * t / (rt * rt);
        }
        for (uint32_t band = 0; band < bands; ++band) {
            const double rt = parameters[30 + bands + band];
            const double value = std::pow(10., parameters[30 + band] / 20 - 3 * t / rt) * noise[size_t(band) * frames + frame];
            gradient[30 + band] += adjoint * value * log10 / 20;
            gradient[30 + bands + band] += adjoint * value * log10 * 3 * t / (rt * rt);
        }
    }
}
} // namespace surface_audio::agarwal
