#include "ErbNoise.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio {
namespace {
constexpr double Tau = 2 * std::numbers::pi;
struct NoiseBlock {
    uint32_t Frames, Taps, Bands;
};
uint64_t Mix(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
double Uniform(uint64_t x) { return (double(Mix(x) >> 11) + .5) * 0x1p-53; }
struct NoiseInputs {
    std::vector<float> White, Filters;
};
NoiseInputs PrepareNoise(uint32_t bands, uint32_t frames, double sample_rate, const ErbNoiseSettings &settings) {
    if (!bands || !frames || frames > UINT32_MAX / bands || !std::isfinite(sample_rate) || sample_rate <= 0) throw std::invalid_argument("Invalid ERB noise dimensions");
    const double low = settings.LowHz, high = settings.HighHz == 0 ? sample_rate / 2 : settings.HighHz;
    const uint32_t taps = settings.TapCount;
    if (taps < 3 || !(taps & 1) || taps > 65535 || bands > UINT32_MAX / taps || frames > UINT32_MAX - (taps - 1) || !std::isfinite(low) || !std::isfinite(high) || low < 0 || high > sample_rate / 2 || low >= high) throw std::invalid_argument("Invalid response noise filter settings");
    NoiseInputs result{std::vector<float>(size_t(frames) + taps - 1), std::vector<float>(bands * taps)};
    for (size_t sample = 0; sample < result.White.size(); ++sample) {
        const uint64_t key = settings.Seed + uint64_t(sample) * 0x9e3779b97f4a7c15ULL;
        result.White[sample] = float(std::sqrt(-2 * std::log(Uniform(key))) * std::cos(Tau * Uniform(key + 0x632be59bd9b4e019ULL)));
    }
    const auto erb = [](double hz) { return 21.4 * std::log10(1 + .00437 * hz); };
    const auto hz = [](double erb_value) { return std::expm1(erb_value * std::log(10.) / 21.4) / .00437; };
    const double low_erb = erb(low), high_erb = erb(high);
    std::vector<double> edges(size_t(bands) + 1);
    for (uint32_t edge = 0; edge <= bands; ++edge) edges[edge] = hz(std::lerp(low_erb, high_erb, double(edge) / bands)) / sample_rate;
    edges.front() = low / sample_rate;
    edges.back() = high / sample_rate;
    const auto lowpass = [](double cutoff, int lag) { return lag == 0 ? 2 * cutoff : std::sin(Tau * cutoff * lag) / (std::numbers::pi * lag); };
    std::vector<double> filter(taps), window(taps);
    for (uint32_t tap = 0; tap < taps; ++tap) window[tap] = .54 - .46 * std::cos(Tau * tap / (taps - 1));
    for (uint32_t band = 0; band < bands; ++band) {
        double energy = 0;
        for (uint32_t tap = 0; tap < taps; ++tap) {
            const int lag = int(tap) - int(taps / 2);
            filter[tap] = (lowpass(edges[band + 1], lag) - lowpass(edges[band], lag)) * window[tap];
            energy += filter[tap] * filter[tap];
        }
        if (!(energy > 0)) throw std::invalid_argument("Degenerate response noise filter");
        const double norm = std::sqrt(energy);
        for (uint32_t tap = 0; tap < taps; ++tap) result.Filters[band * taps + tap] = float(filter[tap] / norm);
    }
    return result;
}
}

std::vector<float> CreateErbNoise(Gpu &gpu, uint32_t bands, uint32_t frames, double sample_rate, const ErbNoiseSettings &settings) {
    const auto inputs = PrepareNoise(bands, frames, sample_rate, settings);
    const auto block = Upload(gpu, NoiseBlock{frames, settings.TapCount, bands});
    const auto white = Upload<float>(gpu, inputs.White), filters = Upload<float>(gpu, inputs.Filters);
    const auto output = CreateBuffer(gpu, size_t(frames) * bands * sizeof(float));
    const auto kernel = CreateKernel(gpu, "ErbNoise");
    BeginGpu(gpu);
    const std::array bindings{GpuBinding{block, 0}, GpuBinding{white, 1}, GpuBinding{filters, 2}, GpuBinding{output, 3}};
    DispatchGpu(gpu, kernel, bindings, {frames, bands, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output);
    return {samples.begin(), samples.end()};
}

std::vector<double> ErbNoiseReference(uint32_t bands, uint32_t frames, double sample_rate, const ErbNoiseSettings &settings) {
    const auto inputs = PrepareNoise(bands, frames, sample_rate, settings);
    std::vector<double> result(size_t(frames) * bands);
    for (uint32_t band = 0; band < bands; ++band) {
        for (uint32_t frame = 0; frame < frames; ++frame) {
            double sum = 0;
            for (uint32_t tap = 0; tap < settings.TapCount; ++tap) sum += double(inputs.Filters[band * settings.TapCount + tap]) * inputs.White[frame + tap];
            result[size_t(band) * frames + frame] = sum;
        }
    }
    return result;
}

}
