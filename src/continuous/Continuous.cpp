#include "Continuous.h"

#include <algorithm>
#include <numbers>
#include <stdexcept>

namespace surface_audio::continuous {
std::array<double, 3> ActionWeights(double angle, double radius) {
    if (!std::isfinite(angle) || !std::isfinite(radius) || radius < 0 || radius > 1) throw std::invalid_argument("Invalid continuous action position");
    constexpr double turn = 2 * std::numbers::pi;
    angle = std::fmod(angle, turn);
    if (angle < 0) angle += turn;
    const double sector = angle * 3 / turn;
    const unsigned first = std::min(unsigned(sector), 2u);
    std::array<double, 3> weights{(1 - radius) / 3, (1 - radius) / 3, (1 - radius) / 3};
    weights[first] += radius * (1 - (sector - first));
    weights[(first + 1) % 3] += radius * (sector - first);
    return weights;
}

namespace {
double GaussianCdf(double value, double sigma) { return .5 * std::erfc(-value / (sigma * std::sqrt(2.))); }
template<typename Cdf> void Tabulate(conan::Process &process, Cdf cdf, double lower, double upper) {
    process.Empirical = 1;
    for (unsigned index = 0; index < conan::QuantileCount; ++index) {
        const double probability = std::clamp(double(conan::QuantileProbability(index)), 1e-7, 1 - 1e-7);
        double low = lower, high = upper;
        for (unsigned step = 0; step < 60; ++step) {
            const double center = .5 * (low + high);
            if (cdf(center) < probability) low = center;
            else high = center;
        }
        process.Quantiles[index] = float(.5 * (low + high));
    }
}
}

Parameters MakeParameters(Controls c, float sample_rate) {
    if (!std::isfinite(sample_rate) || sample_rate < 8000 || sample_rate > 96000 || !std::isfinite(c.Size) || c.Size < .1 || c.Size > 1 || !std::isfinite(c.Velocity) || c.Velocity < 0 || c.Velocity > 1 || !std::isfinite(c.Roughness) || c.Roughness < 0 || c.Roughness > 1 || !std::isfinite(c.Asymmetry) || c.Asymmetry < 0 || c.Asymmetry > 1 || !std::isfinite(c.ScratchDensity) || c.ScratchDensity <= 0 || c.ScratchDensity > sample_rate || !std::isfinite(c.FrictionSigma) || c.FrictionSigma <= 0 || !std::isfinite(c.FrictionDurationSamples) || c.FrictionDurationSamples < 1 || c.FrictionDurationSamples > .008 * sample_rate || !std::isfinite(c.CutoffHz) || c.CutoffHz < 0 || c.CutoffHz > sample_rate / 2 || !std::isfinite(c.Gain)) throw std::invalid_argument("Invalid continuous interaction controls");
    const auto w = ActionWeights(c.Angle, c.Radius);
    const auto rolling = conan::MakeParameters({c.Size, std::max(c.Velocity, .1f), c.Roughness, c.Asymmetry}, sample_rate);
    Parameters p{.Amplitude = {.Mean = float(w[2] * rolling.Amplitude.Mean), .Sigma = c.FrictionSigma, .A1 = float(w[2] * rolling.Amplitude.A1), .B1 = float(w[2] * rolling.Amplitude.B1)}, .Interval = {.Mean = float(w[2] * rolling.Interval.Mean), .A1 = float(w[2] * rolling.Interval.A1), .B1 = float(w[2] * rolling.Interval.B1)}, .SampleRate = sample_rate, .DurationScale = float((w[0] + w[1]) * c.FrictionDurationSamples / sample_rate + w[2] * rolling.DurationScale), .DurationExponent = float(w[2] * .29), .ModulationHz = 3 * c.Velocity / c.Size, .ModulationDepth = float(w[2] * c.Asymmetry), .LowpassPole = c.CutoffHz == 0 ? 0 : float(std::exp(-2 * std::numbers::pi * c.CutoffHz * c.Velocity / sample_rate)), .Gain = c.Velocity > 0 ? c.Gain : 0};
    if (w[2] == 1) {
        p.Amplitude = rolling.Amplitude;
        p.Interval = rolling.Interval;
    } else {
        if (w[2] > 0) Tabulate(p.Amplitude, [&](double x) { return (w[0] + w[1]) * GaussianCdf(x, c.FrictionSigma) + w[2] * GaussianCdf(x, rolling.Amplitude.Sigma); }, -8 * std::max(c.FrictionSigma, rolling.Amplitude.Sigma), 8 * std::max(c.FrictionSigma, rolling.Amplitude.Sigma));
        if (w[0] == 1) p.Interval.Mean = 1 / sample_rate;
        else Tabulate(p.Interval, [&](double x) { return w[0] * (x >= 1 / double(sample_rate)) + w[1] * (x > 0 ? -std::expm1(-c.ScratchDensity * x) : 0) + w[2] * GaussianCdf(x, rolling.Interval.Sigma); }, -8 * rolling.Interval.Sigma, std::max(20 / double(c.ScratchDensity), 8 * double(rolling.Interval.Sigma)));
    }
    Validate(p);
    return p;
}

void Validate(const Parameters &p) {
    if (!std::isfinite(p.SampleRate) || p.SampleRate < 8000 || p.SampleRate > 96000 || !std::isfinite(p.DurationScale) || p.DurationScale <= 0 || !std::isfinite(p.DurationExponent) || p.DurationExponent < 0 || p.DurationExponent > 1 || !std::isfinite(p.MaximumDuration) || p.MaximumDuration * p.SampleRate < 1 || std::ceil(p.MaximumDuration * p.SampleRate) + 2 >= RingSize || !std::isfinite(p.ModulationHz) || p.ModulationHz < 0 || p.ModulationHz >= p.SampleRate / 2 || !std::isfinite(p.ModulationDepth) || p.ModulationDepth < 0 || p.ModulationDepth > 1 || !std::isfinite(p.LowpassPole) || p.LowpassPole < 0 || p.LowpassPole > 1 || !std::isfinite(p.Gain)) throw std::invalid_argument("Invalid continuous source parameters");
    for (const auto *value : {&p.Amplitude, &p.Interval}) {
        const auto &process = *value;
        if (!std::isfinite(process.Mean) || !std::isfinite(process.Sigma) || process.Sigma < 0 || !std::isfinite(process.A1) || std::abs(process.A1) >= 1 || !std::isfinite(process.B1) || std::abs(process.B1) >= 1 || process.Empirical > 1) throw std::invalid_argument("Invalid continuous source process");
        if (process.Empirical && (!std::ranges::all_of(process.Quantiles, [](float x) { return std::isfinite(x); }) || !std::ranges::is_sorted(process.Quantiles))) throw std::invalid_argument("Invalid continuous inverse distribution");
    }
}
State MakeState(uint64_t seed, uint64_t stream) { return {.Random = MakeRandom(seed, stream)}; }
void Render(const Parameters &p, State &s, std::span<float> output) {
    Validate(p);
    for (float &sample : output) sample = Step(p, s);
}
GpuSource CreateGpuSource(Gpu &gpu, std::span<const Parameters> parameters, std::span<const State> states, uint32_t frames) {
    if (parameters.empty() || parameters.size() != states.size() || !frames || parameters.size() * uint64_t(frames) > UINT32_MAX) throw std::invalid_argument("Invalid continuous GPU dimensions");
    for (const auto &p : parameters) Validate(p);
    return {uint32_t(parameters.size()), frames, Upload<Parameters>(gpu, parameters), Upload<State>(gpu, states), CreateBuffer(gpu, parameters.size() * frames * sizeof(float)), Upload(gpu, frames), CreateKernel(gpu, "ContinuousSynthesize")};
}
void EncodeSource(Gpu &gpu, const GpuSource &source) {
    const std::array bindings{GpuBinding{source.Parameters, 0}, GpuBinding{source.States, 1}, GpuBinding{source.Output, 2}, GpuBinding{source.Block, 3}};
    DispatchGpu(gpu, source.Kernel, bindings, {source.Voices});
}
}
