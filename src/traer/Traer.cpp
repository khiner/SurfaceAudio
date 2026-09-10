#include "Traer.h"
#include "core/GpuConvolution.h"
#include "core/GpuInterpolation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace surface_audio::traer {
namespace {
struct ResponseBlock {
    uint32_t Frames, Voices, Modes, Bands;
    float SampleRate;
};
void Validate(std::span<const Resonance> modes, std::span<const Transient> bands, uint32_t frames, double rate, std::span<const float> noise, size_t output) {
    if (!frames || !std::isfinite(rate) || rate <= 0 || noise.size() != size_t(frames) * bands.size() || output != frames)
        throw std::invalid_argument("Invalid Traer response dimensions");
    for (const auto &mode : modes)
        if (!std::isfinite(mode.Frequency) || mode.Frequency < 0 || mode.Frequency > rate / 2 || !std::isfinite(mode.OnsetDb) ||
            !std::isfinite(mode.DecayDbPerSecond) || mode.DecayDbPerSecond <= 0)
            throw std::invalid_argument("Invalid Traer resonance");
    for (const auto &band : bands)
        if (!std::isfinite(band.OnsetDb) || !std::isfinite(band.DecayDbPerSecond) || band.DecayDbPerSecond <= 0)
            throw std::invalid_argument("Invalid Traer transient");
    if (!std::ranges::all_of(noise, [](float value) { return std::isfinite(value); })) throw std::invalid_argument("Nonfinite noise");
}
template<typename T>
void Evaluate(std::span<const Resonance> modes, std::span<const Transient> bands, uint32_t frames, T rate, std::span<const float> noise, std::span<T> output) {
    Validate(modes, bands, frames, rate, noise, output.size());
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const T time = T(frame) / rate;
        T sum = 0;
        for (const auto &mode : modes) {
            if (frame >= mode.EndFrame) continue;
            T cycles = T(mode.Frequency) * time;
            if constexpr (std::is_same_v<T, float>) {
                const float increment = mode.Frequency / rate;
                const float increment_error = std::fma(-increment, rate, mode.Frequency) / rate;
                const float high = float(frame) * increment;
                cycles = (high - std::floor(high)) + std::fma(float(frame), increment, -high) + float(frame) * increment_error;
            }
            sum += std::pow(T(10), (T(mode.OnsetDb) - T(mode.DecayDbPerSecond) * time) / T(20)) *
                std::cos(T(2) * std::numbers::pi_v<T> * cycles);
        }
        for (size_t band = 0; band < bands.size(); ++band)
            sum += std::pow(T(10), (T(bands[band].OnsetDb) - T(bands[band].DecayDbPerSecond) * time) / T(20)) *
                T(noise[band * frames + frame]);
        output[frame] = sum;
    }
}
}
Gaussian MakeGaussian(std::span<const double> mean, std::span<const double> covariance) {
    const size_t n = mean.size();
    if (!n || covariance.size() != n * n) throw std::invalid_argument("Invalid Gaussian dimensions");
    if (!std::ranges::all_of(mean, [](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(covariance, [](double x) { return std::isfinite(x); })) throw std::invalid_argument("Nonfinite Gaussian");
    Gaussian result{{mean.begin(), mean.end()}, std::vector<double>(n * n)};
    double scale = 1;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::abs(covariance[i * n + i]));
    const double tolerance = 1e-12 * scale;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (std::abs(covariance[i * n + j] - covariance[j * n + i]) > tolerance)
                throw std::invalid_argument("Asymmetric covariance");
            double value = covariance[i * n + j];
            for (size_t k = 0; k < j; ++k) value -= result.Lower[i * n + k] * result.Lower[j * n + k];
            if (i == j) {
                if (value < -tolerance) throw std::invalid_argument("Indefinite covariance");
                result.Lower[i * n + j] = std::sqrt(std::max(0., value));
            } else if (result.Lower[j * n + j] > 0) result.Lower[i * n + j] = value / result.Lower[j * n + j];
            else if (std::abs(value) > tolerance) throw std::invalid_argument("Indefinite singular covariance");
        }
    }
    return result;
}
std::vector<double> SampleGaussian(const Gaussian &distribution, RandomState &random) {
    const size_t n = distribution.Mean.size();
    if (!n || distribution.Lower.size() != n * n) throw std::invalid_argument("Invalid Gaussian dimensions");
    std::vector<double> normal(n), sample(distribution.Mean);
    for (double &value : normal) value = Normal(random);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j <= i; ++j) sample[i] += distribution.Lower[i * n + j] * normal[j];
    return sample;
}
std::vector<Resonance> SampleModes(const Gaussian &distribution, double spacing, double rate, RandomState &random, uint32_t attempts) {
    const size_t n = distribution.Mean.size() / 3;
    if (n < 2 || distribution.Mean.size() != 3 * n || !std::isfinite(spacing) || spacing <= 0 || !std::isfinite(rate) || rate <= 0)
        throw std::invalid_argument("Invalid modal distribution");
    for (uint32_t attempt = 0; attempt < attempts; ++attempt) {
        const auto sample = SampleGaussian(distribution, random);
        bool valid = true;
        for (size_t i = 0; i < n; ++i)
            valid &= std::isfinite(sample[i]) && sample[i] > 0 && sample[i] < rate / 2 && std::isfinite(sample[n + i]) &&
                std::isfinite(sample[2 * n + i]) && sample[2 * n + i] > 0;
        if (!valid) continue;
        const auto [low, high] = std::minmax_element(sample.begin(), sample.begin() + n);
        if (std::abs((*high - *low) / double(n - 1) - spacing) > .1 * spacing) continue;
        std::vector<Resonance> result(n);
        for (size_t i = 0; i < n; ++i) result[i] = {float(sample[i]), float(sample[n + i]), float(sample[2 * n + i])};
        return result;
    }
    throw std::runtime_error("Traer mode spacing rejection exhausted");
}
void PerturbOnsets(std::span<Resonance> modes, RandomState &random) {
    if (modes.empty()) return;
    double mean = 0;
    for (const auto &mode : modes) mean += mode.OnsetDb;
    const float sigma = float(.2 * std::abs(mean / double(modes.size())));
    for (auto &mode : modes) mode.OnsetDb += sigma * Normal(random);
}
void EvaluateResponse(std::span<const Resonance> modes, std::span<const Transient> bands, uint32_t frames, double rate, std::span<const float> noise, std::span<double> output) { Evaluate(modes, bands, frames, rate, noise, output); }
void EvaluateResponseFloat(std::span<const Resonance> modes, std::span<const Transient> bands, uint32_t frames, float rate, std::span<const float> noise, std::span<float> output) { Evaluate(modes, bands, frames, rate, noise, output); }
ResponseGpu CreateResponseGpu(Gpu &gpu, uint32_t frames, uint32_t voices, float rate, std::span<const Resonance> modes, std::span<const Transient> bands, std::span<const float> noise) {
    if (!voices || !frames || frames > 16777216 || uint64_t(frames) * voices > UINT32_MAX || modes.size() > UINT32_MAX || bands.size() > UINT32_MAX ||
        uint64_t(frames) * bands.size() > UINT32_MAX || modes.size() % voices || bands.size() % voices)
        throw std::invalid_argument("Invalid response batch dimensions");
    const auto mode_count = uint32_t(modes.size() / voices), band_count = uint32_t(bands.size() / voices);
    Validate(modes, bands, frames, rate, noise, frames);
    return {frames, voices, Upload(gpu, ResponseBlock{frames, voices, mode_count, band_count, rate}), modes.empty() ? CreateBuffer(gpu, 16) : Upload<Resonance>(gpu, modes), bands.empty() ? CreateBuffer(gpu, 8) : Upload<Transient>(gpu, bands), noise.empty() ? CreateBuffer(gpu, 4) : Upload<float>(gpu, noise), CreateBuffer(gpu, size_t(frames) * voices * sizeof(float)), CreateKernel(gpu, "TraerResponse")};
}
void EncodeResponse(Gpu &gpu, const ResponseGpu &response) {
    const std::array bindings{GpuBinding{response.Parameters, 0}, GpuBinding{response.Resonances, 1}, GpuBinding{response.Transients, 2}, GpuBinding{response.Noise, 3}, GpuBinding{response.Output, 4}};
    DispatchGpu(gpu, response.Synthesize, bindings, {response.Frames, response.Voices, 1});
}
double SpringContactDuration(double mass, double stiffness) {
    if (!std::isfinite(mass) || !std::isfinite(stiffness) || mass <= 0 || stiffness <= 0) throw std::invalid_argument("Invalid spring");
    return std::numbers::pi * std::sqrt(mass / stiffness);
}
std::vector<float> ImpactForce(double duration, double peak, double rate) {
    if (!std::isfinite(duration) || !std::isfinite(peak) || !std::isfinite(rate) || duration <= 0 || peak < 0 || rate <= 0 ||
        duration * rate > std::numeric_limits<uint32_t>::max() - 1) throw std::invalid_argument("Invalid impact force");
    std::vector<float> result(size_t(std::ceil(duration * rate)) + 1);
    for (size_t i = 0; i < result.size(); ++i) {
        const double t = double(i) / rate;
        if (t < duration) result[i] = float(peak * std::sin(std::numbers::pi * t / duration));
    }
    return result;
}
double ScrapeForce(double slope, double curvature, double velocity, double mass, double shear, double gamma) {
    if (!std::isfinite(slope) || !std::isfinite(curvature) || !std::isfinite(velocity) || !std::isfinite(mass) ||
        !std::isfinite(shear) || !std::isfinite(gamma) || mass < 0 || shear < 0 || gamma <= 0)
        throw std::invalid_argument("Invalid scrape parameters");
    const double tangential = velocity * slope;
    const double friction = gamma == std::floor(gamma) ? std::pow(tangential, gamma) : std::copysign(std::pow(std::abs(tangential), gamma), tangential);
    return mass * curvature * velocity * velocity + shear * friction;
}
std::vector<float> ScrapeExcitation(std::span<const double> profile, double spacing, std::span<const double> position, std::span<const double> velocity, double mass, double shear, double gamma) {
    if (profile.size() < 3 || position.size() != velocity.size() || !std::isfinite(spacing) || spacing <= 0 ||
        !std::ranges::all_of(profile, [](double x) { return std::isfinite(x); })) throw std::invalid_argument("Invalid scrape profile");
    std::vector<float> output(position.size());
    for (size_t i = 0; i < position.size(); ++i) {
        const double x = position[i] / spacing;
        if (!std::isfinite(x) || x < 1 || x >= double(profile.size() - 2)) throw std::invalid_argument("Scrape outside profile interior");
        const auto j = size_t(x);
        const auto slope = [&](size_t k) { return (profile[k + 1] - profile[k - 1]) / (2 * spacing); };
        const auto curve = [&](size_t k) { return (profile[k + 1] - 2 * profile[k] + profile[k - 1]) / (spacing * spacing); };
        output[i] = float(ScrapeForce(std::lerp(slope(j), slope(j + 1), x - j), std::lerp(curve(j), curve(j + 1), x - j), velocity[i], mass, shear, gamma));
    }
    return output;
}
std::vector<double> QuiltProfile(std::span<const double> rows, uint32_t columns, uint32_t samples, uint32_t overlap, RandomState &random) {
    if (columns < 2 || rows.empty() || rows.size() % columns || !samples || overlap >= columns ||
        !std::ranges::all_of(rows, [](double x) { return std::isfinite(x); })) throw std::invalid_argument("Invalid quilting dimensions");
    const size_t count = rows.size() / columns;
    std::vector<double> result;
    result.reserve(samples);
    size_t chosen = NextRandom(random) % count;
    result.insert(result.end(), rows.begin() + chosen * columns, rows.begin() + chosen * columns + std::min(samples, columns));
    while (result.size() < samples) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t trial = 0; trial < std::min<size_t>(count, 16); ++trial) {
            const size_t candidate = NextRandom(random) % count;
            double error = 0;
            for (uint32_t j = 0; j < overlap; ++j) error += std::pow(result[result.size() - overlap + j] - rows[candidate * columns + j], 2);
            if (error < best) {
                best = error;
                chosen = candidate;
            }
        }
        for (uint32_t j = 0; j < overlap; ++j)
            result[result.size() - overlap + j] = std::lerp(result[result.size() - overlap + j], rows[chosen * columns + j], double(j + 1) / (overlap + 1));
        const size_t append = std::min<size_t>(columns - overlap, samples - result.size());
        result.insert(result.end(), rows.begin() + chosen * columns + overlap, rows.begin() + chosen * columns + overlap + append);
    }
    return result;
}
std::vector<float> ConvolveSpatialGpu(Gpu &gpu, std::span<const float> excitation, std::span<const float> responses, uint32_t taps, std::span<const float> nodes, std::span<const float> positions) {
    if (!taps || responses.empty() || responses.size() % taps || excitation.empty() || excitation.size() > UINT32_MAX - uint64_t(taps) + 1)
        throw std::invalid_argument("Invalid spatial convolution dimensions");
    const size_t regions = responses.size() / taps, frames = excitation.size() + taps - 1;
    if (regions > UINT32_MAX / frames || positions.size() != excitation.size() || nodes.size() != regions)
        throw std::invalid_argument("Invalid spatial positions extent");
    std::vector<float> components(regions * frames), tail_positions(positions.begin(), positions.end());
    tail_positions.resize(frames, positions.back());
    for (size_t region = 0; region < regions; ++region) {
        const auto convolved = ConvolveFixedGpu(gpu, excitation, responses.subspan(region * taps, taps));
        std::copy(convolved.begin(), convolved.end(), components.begin() + region * frames);
    }
    return InterpolateSignalsGpu(gpu, components, nodes, tail_positions);
}
}
