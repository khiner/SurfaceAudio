#include "Impact.h"
#include "core/GpuConvolution.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio::agarwal {
namespace {
struct ImpactParameters {
    float Amplitude, Omega, Limit, Duration;
};
struct ImpactBlock {
    uint32_t Frames, Voices, Sampling;
    float SampleRate;
};
bool Positive(double x) { return std::isfinite(x) && x > 0; }
}

double EquivalentStiffness(double a, double b, StiffnessCombination combination) {
    if (!Positive(a) || !Positive(b) || uint32_t(combination) > 1) throw std::invalid_argument("Invalid impact stiffness");
    const double value = combination == StiffnessCombination::Sum ? a + b : std::min(a, b) / (1 + std::min(a, b) / std::max(a, b));
    if (!Positive(value)) throw std::invalid_argument("Impact stiffness overflow");
    return value;
}

double PlateStiffness(double young, double poisson, double side, double thickness) {
    if (!Positive(young) || !Positive(side) || !Positive(thickness) || !std::isfinite(poisson) || poisson <= -1 || poisson >= .5) throw std::invalid_argument("Invalid plate material or dimensions");
    const double value = 16 * young * thickness * thickness * thickness / (3 * (1 - poisson * poisson) * side * side);
    if (!Positive(value)) throw std::invalid_argument("Plate stiffness overflow");
    return value;
}

double PublishedStiffness(ImpactMaterial material, uint32_t size) {
    constexpr std::array<std::array<double, 4>, 7> table{{{3.04e7, 3.89e7, 1.45e7, .48e7}, {11.95e7, 15.29e7, 5.74e7, 1.92e7}, {7.03e7, 9e7, 3.38e7, 1.12e7}, {2.28e7, 2.92e7, 1.08e7, .40e7}, {.69e7, .88e7, .34e7, .08e7}, {.16e7, .21e7, .07e7, .024e7}, {.20e7, .25e7, .10e7, .032e7}}};
    if (uint32_t(material) >= table.size() || size >= 4) throw std::invalid_argument("Invalid material or size category");
    return table[uint32_t(material)][size];
}

double ImpactDuration(const ImpactSettings &s) {
    if (!Positive(s.Mass) || !std::isfinite(s.Velocity) || s.Velocity < 0 || !std::isfinite(s.Scale) || s.Scale < 0 || !(s.ForceLimit > 0)) throw std::invalid_argument("Invalid impact controls");
    const double duration = std::numbers::pi * std::sqrt(s.Mass / EquivalentStiffness(s.StiffnessA, s.StiffnessB, s.Combination));
    if (!Positive(duration)) throw std::invalid_argument("Invalid impact duration");
    return duration;
}

double ImpactForce(double time, const ImpactSettings &s) {
    const double duration = ImpactDuration(s);
    if (!std::isfinite(time)) throw std::invalid_argument("Nonfinite impact time");
    if (time <= 0 || time >= duration) return 0;
    const double force = s.Scale * s.Mass * s.Velocity * std::sin(std::numbers::pi * time / duration);
    if (!std::isfinite(force)) throw std::invalid_argument("Impact force overflow");
    return std::isinf(s.ForceLimit) ? force : s.ForceLimit * std::tanh(force / s.ForceLimit);
}

std::vector<float> RenderImpactForcesGpu(Gpu &gpu, std::span<const ImpactSettings> settings, uint32_t frames, double rate, ImpactSampling sampling) {
    if (settings.empty() || !frames || settings.size() > UINT32_MAX / frames || !Positive(rate) || rate > 1e9 || uint32_t(sampling) > 1) throw std::invalid_argument("Invalid impact batch dimensions");
    std::vector<ImpactParameters> parameters;
    parameters.reserve(settings.size());
    for (const auto &s : settings) {
        const double duration = ImpactDuration(s), amplitude = s.Scale * s.Mass * s.Velocity;
        const ImpactParameters p{float(amplitude), float(std::numbers::pi / duration), float(s.ForceLimit), float(duration)};
        if (duration * rate > frames || !std::isfinite(p.Amplitude) || !std::isfinite(p.Omega) || p.Duration <= 0 || p.Limit <= 0 || (std::isfinite(s.ForceLimit) && !std::isfinite(p.Limit))) throw std::invalid_argument("Incomplete or unrepresentable impact pulse");
        parameters.push_back(p);
    }
    const auto controls = Upload<ImpactParameters>(gpu, parameters);
    const auto block = Upload(gpu, ImpactBlock{frames, uint32_t(settings.size()), uint32_t(sampling), float(rate)});
    const auto output = CreateBuffer(gpu, size_t(frames) * settings.size() * sizeof(float));
    const auto kernel = CreateKernel(gpu, "ImpactForces");
    BeginGpu(gpu);
    const std::array bindings{GpuBinding{block, 0}, GpuBinding{controls, 1}, GpuBinding{output, 2}};
    DispatchGpu(gpu, kernel, bindings, {frames, uint32_t(settings.size()), 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output);
    return {samples.begin(), samples.end()};
}

std::vector<float> MixContactForcesGpu(Gpu &gpu, std::span<const float> scraping, std::span<const float> elastic, std::span<const float> dissipative, double stiffness, double dissipation) {
    if (scraping.empty() || scraping.size() > UINT32_MAX || elastic.size() != scraping.size() || dissipative.size() != scraping.size() || !std::isfinite(stiffness) || stiffness < 0 || !std::isfinite(dissipation) || dissipation < 0 || !std::isfinite(float(stiffness)) || !std::isfinite(float(dissipation))) throw std::invalid_argument("Invalid contact force components");
    for (auto values : {scraping, elastic, dissipative})
        if (!std::ranges::all_of(values, [](float x) { return std::isfinite(x); })) throw std::invalid_argument("Nonfinite contact force component");
    struct Block {
        uint32_t Frames;
        float Stiffness, Dissipation;
    };
    const auto block = Upload(gpu, Block{uint32_t(scraping.size()), float(stiffness), float(dissipation)});
    const auto a = Upload<float>(gpu, scraping), b = Upload<float>(gpu, elastic), c = Upload<float>(gpu, dissipative);
    const auto output = CreateBuffer(gpu, scraping.size_bytes());
    const auto kernel = CreateKernel(gpu, "ContactForceMix");
    BeginGpu(gpu);
    const std::array bindings{GpuBinding{block, 0}, GpuBinding{a, 1}, GpuBinding{b, 2}, GpuBinding{c, 3}, GpuBinding{output, 4}};
    DispatchGpu(gpu, kernel, bindings, {uint32_t(scraping.size())});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output);
    if (!std::ranges::all_of(samples, [](float x) { return std::isfinite(x); })) throw std::overflow_error("Contact force overflow");
    return {samples.begin(), samples.end()};
}

std::vector<float> RenderMicroImpactGpu(Gpu &gpu, const ImpactSettings &s, double rate, ImpactSampling sampling) {
    const ImpactSettings linear{.Mass = s.Mass, .Velocity = s.Velocity, .StiffnessA = s.StiffnessA, .StiffnessB = s.StiffnessB, .Scale = s.Scale, .Combination = s.Combination};
    const double frames = std::ceil(ImpactDuration(linear) * rate);
    if (!Positive(rate) || !Positive(frames) || frames >= UINT32_MAX) throw std::invalid_argument("Invalid micro-impact sampling");
    return RenderImpactForcesGpu(gpu, std::span{&linear, 1}, uint32_t(frames), rate, sampling);
}

std::vector<float> RenderContactGpu(Gpu &gpu, std::span<const float> excitation, const ImpactSettings &s, double rate, std::span<const float> a, std::span<const float> b) {
    if (excitation.empty() || (a.empty() && b.empty())) throw std::invalid_argument("Contact rendering requires excitation and an object response");
    std::vector<float> response(std::max(a.size(), b.size()));
    std::ranges::copy(a, response.begin());
    if (!b.empty()) vDSP_vadd(response.data(), 1, b.data(), 1, response.data(), 1, b.size());
    const auto pulse = RenderMicroImpactGpu(gpu, s, rate);
    const auto impact = ConvolveFixedGpu(gpu, pulse, response);
    return ConvolveFixedGpu(gpu, excitation, impact);
}
}
