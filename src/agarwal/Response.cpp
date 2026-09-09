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
    uint32_t Frames, Groups;
    float SampleRate;
};
struct NoiseBlock {
    uint32_t Frames, Taps;
};
void ValidateDimensions(uint32_t frames, double sample_rate) {
    if (!frames || frames > std::numeric_limits<uint32_t>::max() / ResponseModeCount || !std::isfinite(sample_rate) || sample_rate <= 0) throw std::invalid_argument("Invalid response dimensions");
}
void Validate(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise) {
    ValidateDimensions(frames, sample_rate);
    if (parameters.size() != ResponseParameterCount || noise.size() != size_t(frames) * ResponseModeCount) throw std::invalid_argument("Invalid response parameter or noise dimensions");
    for (double value : parameters)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite response parameter");
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        if (parameters[mode] < 0 || parameters[mode] > sample_rate / 2 || parameters[20 + mode] <= 0 || parameters[40 + mode] <= 0) throw std::invalid_argument("Invalid response frequency or RT60");
    }
}
uint64_t Mix(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
double Uniform(uint64_t x) { return (double(Mix(x) >> 11) + .5) * 0x1p-53; }
struct NoiseInputs {
    std::vector<float> White, Filters;
};
NoiseInputs PrepareNoise(uint32_t frames, double sample_rate, const ResponseNoiseSettings &settings) {
    ValidateDimensions(frames, sample_rate);
    const double low = settings.LowHz, high = settings.HighHz == 0 ? sample_rate / 2 : settings.HighHz;
    const uint32_t taps = settings.TapCount;
    if (taps < 3 || !(taps & 1) || taps > 65535 || !std::isfinite(low) || !std::isfinite(high) || low < 0 || high > sample_rate / 2 || low >= high) throw std::invalid_argument("Invalid response noise filter settings");
    NoiseInputs result{std::vector<float>(size_t(frames) + taps - 1), std::vector<float>(ResponseModeCount * taps)};
    for (size_t sample = 0; sample < result.White.size(); ++sample) {
        const uint64_t key = settings.Seed + uint64_t(sample) * 0x9e3779b97f4a7c15ULL;
        result.White[sample] = float(std::sqrt(-2 * std::log(Uniform(key))) * std::cos(Tau * Uniform(key + 0x632be59bd9b4e019ULL)));
    }
    const auto erb = [](double hz) { return 21.4 * std::log10(1 + .00437 * hz); };
    const auto hz = [](double erb_value) { return std::expm1(erb_value * std::log(10.) / 21.4) / .00437; };
    std::array<double, ResponseModeCount + 1> edges{};
    for (uint32_t edge = 0; edge <= ResponseModeCount; ++edge) edges[edge] = hz(std::lerp(erb(low), erb(high), double(edge) / ResponseModeCount)) / sample_rate;
    edges.front() = low / sample_rate;
    edges.back() = high / sample_rate;
    const auto lowpass = [](double cutoff, int lag) { return lag == 0 ? 2 * cutoff : std::sin(Tau * cutoff * lag) / (std::numbers::pi * lag); };
    std::vector<double> filter(taps);
    for (uint32_t band = 0; band < ResponseModeCount; ++band) {
        double energy = 0;
        for (uint32_t tap = 0; tap < taps; ++tap) {
            const int lag = int(tap) - int(taps / 2);
            filter[tap] = (lowpass(edges[band + 1], lag) - lowpass(edges[band], lag)) * (.54 - .46 * std::cos(Tau * tap / (taps - 1)));
            energy += filter[tap] * filter[tap];
        }
        if (!(energy > 0)) throw std::invalid_argument("Degenerate response noise filter");
        for (uint32_t tap = 0; tap < taps; ++tap) result.Filters[band * taps + tap] = float(filter[tap] / std::sqrt(energy));
    }
    return result;
}
} // namespace

std::vector<float> CreateResponseNoise(Gpu &gpu, uint32_t frames, double sample_rate, const ResponseNoiseSettings &settings) {
    const auto inputs = PrepareNoise(frames, sample_rate, settings);
    const auto block = Upload(gpu, NoiseBlock{frames, settings.TapCount});
    const auto white = Upload<float>(gpu, inputs.White), filters = Upload<float>(gpu, inputs.Filters);
    const auto output = CreateBuffer(gpu, size_t(frames) * ResponseModeCount * sizeof(float));
    const auto kernel = CreateKernel(gpu, "ResponseNoise");
    BeginGpu(gpu);
    const std::array bindings{GpuBinding{block, 0}, GpuBinding{white, 1}, GpuBinding{filters, 2}, GpuBinding{output, 3}};
    DispatchGpu(gpu, kernel, bindings, {frames, ResponseModeCount, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output);
    return {samples.begin(), samples.end()};
}

std::vector<double> ResponseNoiseReference(uint32_t frames, double sample_rate, const ResponseNoiseSettings &settings) {
    const auto inputs = PrepareNoise(frames, sample_rate, settings);
    std::vector<double> result(size_t(frames) * ResponseModeCount);
    for (uint32_t band = 0; band < ResponseModeCount; ++band) {
        for (uint32_t frame = 0; frame < frames; ++frame) {
            double sum = 0;
            for (uint32_t tap = 0; tap < settings.TapCount; ++tap) sum += double(inputs.Filters[band * settings.TapCount + tap]) * inputs.White[frame + tap];
            result[size_t(band) * frames + frame] = sum;
        }
    }
    return result;
}

ResponseGpu CreateResponseGpu(Gpu &gpu, uint32_t frames, float sample_rate, std::span<const float> noise) {
    ValidateDimensions(frames, sample_rate);
    if (noise.size() != size_t(frames) * ResponseModeCount || !std::ranges::all_of(noise, [](float x) { return std::isfinite(x); })) throw std::invalid_argument("Invalid response noise");
    const uint32_t groups = (frames + GradientLanes - 1) / GradientLanes;
    ResponseParameters initial{};
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        initial[mode] = sample_rate * float(mode + 1) / 24;
        initial[10 + mode] = initial[30 + mode] = -60;
        initial[20 + mode] = initial[40 + mode] = .1f;
    }
    return {
        .Frames = frames,
        .GradientGroups = groups,
        .SampleRate = sample_rate,
        .Parameters = Upload<float>(gpu, initial),
        .Output = CreateBuffer(gpu, size_t(frames) * sizeof(float)),
        .Gradient = CreateBuffer(gpu, ResponseParameterCount * sizeof(float)),
        .Noise = Upload<float>(gpu, noise),
        .Block = Upload(gpu, ResponseBlock{frames, groups, sample_rate}),
        .PartialGradient = CreateBuffer(gpu, size_t(groups) * ResponseParameterCount * sizeof(float)),
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
    DispatchGroupsGpu(gpu, state.Differentiate, bindings, {state.GradientGroups, ResponseParameterCount, 1}, {GradientLanes, 1, 1});
    const std::array reduce_bindings{GpuBinding{state.Block, 0}, GpuBinding{state.PartialGradient, 1}, GpuBinding{state.Gradient, 2}};
    DispatchGroupsGpu(gpu, state.Reduce, reduce_bindings, {ResponseParameterCount, 1, 1}, {GradientLanes, 1, 1});
}

void EvaluateResponse(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise, std::span<double> output) {
    Validate(parameters, frames, sample_rate, noise);
    if (output.size() != frames) throw std::invalid_argument("Invalid response output dimensions");
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double t = frame / sample_rate;
        double sum = 0;
        for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
            sum += std::pow(10., parameters[10 + mode] / 20 - 3 * t / parameters[20 + mode]) * std::sin(Tau * parameters[mode] * t);
            sum += std::pow(10., parameters[30 + mode] / 20 - 3 * t / parameters[40 + mode]) * noise[size_t(mode) * frames + frame];
        }
        output[frame] = sum;
    }
}

void EvaluateResponseGradient(std::span<const double> parameters, uint32_t frames, double sample_rate, std::span<const float> noise, std::span<const double> sample_adjoint, std::span<double> gradient) {
    Validate(parameters, frames, sample_rate, noise);
    if (sample_adjoint.size() != frames || gradient.size() != ResponseParameterCount) throw std::invalid_argument("Invalid response gradient dimensions");
    std::ranges::fill(gradient, 0);
    const double log10 = std::log(10.);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double t = frame / sample_rate, adjoint = sample_adjoint[frame];
        for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
            const double rt = parameters[20 + mode], noise_rt = parameters[40 + mode];
            const double envelope = std::pow(10., parameters[10 + mode] / 20 - 3 * t / rt);
            const double value = envelope * std::sin(Tau * parameters[mode] * t);
            const double noise_value = std::pow(10., parameters[30 + mode] / 20 - 3 * t / noise_rt) * noise[size_t(mode) * frames + frame];
            gradient[mode] += adjoint * envelope * Tau * t * std::cos(Tau * parameters[mode] * t);
            gradient[10 + mode] += adjoint * value * log10 / 20;
            gradient[20 + mode] += adjoint * value * log10 * 3 * t / (rt * rt);
            gradient[30 + mode] += adjoint * noise_value * log10 / 20;
            gradient[40 + mode] += adjoint * noise_value * log10 * 3 * t / (noise_rt * noise_rt);
        }
    }
}
} // namespace surface_audio::agarwal
