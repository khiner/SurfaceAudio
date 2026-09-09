#include "core/GpuSpectral.h"
#include "core/Random.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace surface_audio;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

// Scalar double direct DFT, with no FFT implementation or GPU intermediates.
std::vector<double> Magnitudes(std::span<const double> waveform, uint32_t size, SpectralLossOptions options) {
    const uint32_t hop = size / options.HopDivisor, frames = 1 + uint32_t(waveform.size()) / hop, bins = size / 2 + 1;
    std::vector<double> result(size_t(frames) * bins);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const int first = std::max(0, int(size / 2) - int(frame * hop));
        const int last = std::min(int(size), int(waveform.size()) + int(size / 2) - int(frame * hop));
        std::vector<double> window(last - first);
        for (int n = first; n < last; ++n) window[n - first] = waveform[frame * hop + n - size / 2] * (.5 - .5 * std::cos(2 * std::numbers::pi * n / size));
        for (uint32_t bin = 0; bin < bins; ++bin) {
            const double angle = -2 * std::numbers::pi * bin / size;
            const std::complex<double> step = std::polar(1., angle);
            auto phase = std::polar(1., angle * first);
            std::complex<double> sum = 0;
            for (double sample : window) {
                sum += sample * phase;
                phase *= step;
            }
            const double power = std::norm(sum) + double(options.MagnitudeFloor) * options.MagnitudeFloor;
            result[frame * bins + bin] = options.Scale == SpectralMagnitudeScale::Decibels ? 10 * std::log10(power) : (options.Scale == SpectralMagnitudeScale::NaturalLog ? .5 * std::log(power) : std::sqrt(power));
        }
    }
    return result;
}

using Representations = std::array<std::vector<double>, 4>;
Representations ReferenceTarget(std::span<const float> target, SpectralLossOptions options) {
    Representations result;
    const std::vector<double> waveform(target.begin(), target.end());
    constexpr std::array sizes{4096u, 1024u, 256u, 64u};
    for (uint32_t index = 0; index < 4; ++index) result[index] = Magnitudes(waveform, sizes[index], options);
    return result;
}
std::array<double, 5> ReferenceLoss(std::span<const double> waveform, const Representations &target, SpectralLossOptions options) {
    std::array<double, 5> loss{};
    constexpr std::array sizes{4096u, 1024u, 256u, 64u};
    for (uint32_t index = 0; index < 4; ++index) {
        const auto magnitudes = Magnitudes(waveform, sizes[index], options);
        for (size_t bin = 0; bin < magnitudes.size(); ++bin) {
            const double difference = std::abs(magnitudes[bin] - target[index][bin]);
            loss[index + 1] += difference <= options.HuberDelta ? .5 * difference * difference : options.HuberDelta * (difference - .5 * options.HuberDelta);
        }
        loss[index + 1] /= magnitudes.size();
        loss[0] += loss[index + 1];
    }
    return loss;
}
void Evaluate(Gpu &gpu, const SpectralLossGpu &state, GpuBuffer waveform) {
    BeginGpu(gpu);
    EncodeSpectralLoss(gpu, state, waveform);
    SubmitGpu(gpu);
    WaitGpu(gpu);
}

void ReferenceTest(Gpu &gpu, SpectralMagnitudeScale scale) {
    constexpr uint32_t samples = 1157;
    auto random = MakeRandom(913);
    std::vector<float> target(samples), waveform(samples);
    for (uint32_t sample = 0; sample < samples; ++sample) {
        target[sample] = .1f * std::sin(.097f * sample) + .03f * (Uniform(random) - .5f);
        waveform[sample] = .93f * target[sample] + .002f * (Uniform(random) - .5f) + .003f * std::cos(.037f * sample);
    }
    const SpectralLossOptions options{.Scale = scale};
    const auto state = CreateSpectralLossGpu(gpu, target, 22050, options);
    const auto input = Upload<float>(gpu, waveform);
    const auto reference = ReferenceTarget(target, options);
    const std::vector<double> reference_waveform(waveform.begin(), waveform.end());
    const auto expected = ReferenceLoss(reference_waveform, reference, options);
    Evaluate(gpu, state, input);
    const auto actual = BufferSpan<float>(state.Loss), gradient = BufferSpan<float>(state.Gradient);
    for (uint32_t index = 0; index < 5; ++index) Require(std::abs(actual[index] - expected[index]) < 2e-5 * std::max(1., expected[index]), "Spectral loss versus independent direct DFT");
    const std::vector<float> saved_gradient(gradient.begin(), gradient.end()), saved_loss(actual.begin(), actual.end());
    double maximum_error = 0;
    for (uint32_t trial = 0; trial < (scale == SpectralMagnitudeScale::Decibels ? 5u : 1u); ++trial) {
        std::vector<float> direction(samples);
        std::vector<double> plus(samples), minus(samples);
        if (trial == 1) direction[0] = 1;
        else if (trial == 2) direction.back() = 1;
        else if (trial == 3) direction[1024] = 1;
        else
            for (float &value : direction) value = Uniform(random) - .5f;
        const double analytic = std::inner_product(gradient.begin(), gradient.end(), direction.begin(), 0.);
        for (double step : {1e-5, 3e-6}) {
            for (uint32_t sample = 0; sample < samples; ++sample) {
                plus[sample] = waveform[sample] + step * direction[sample];
                minus[sample] = waveform[sample] - step * direction[sample];
            }
            const double numeric = (ReferenceLoss(plus, reference, options)[0] - ReferenceLoss(minus, reference, options)[0]) / (2 * step);
            const double error = std::abs(analytic - numeric) / std::max(1., std::abs(numeric));
            maximum_error = std::max(maximum_error, error);
            Require(error < .001, "Spectral analytic sample adjoint versus central finite difference");
        }
    }
    Evaluate(gpu, state, input);
    Require(std::equal(saved_gradient.begin(), saved_gradient.end(), gradient.begin()) && std::equal(saved_loss.begin(), saved_loss.end(), actual.begin()), "Spectral scratch reuse is deterministic");
    std::copy(target.begin(), target.end(), BufferSpan<float>(input).begin());
    Evaluate(gpu, state, input);
    Require(std::ranges::all_of(actual, [](float value) { return value == 0; }) && std::ranges::all_of(gradient, [](float value) { return value == 0; }), "Identical waveforms have zero loss and gradient");
    std::cout << "Spectral scale " << uint32_t(scale) << " loss " << expected[0] << ", maximum directional error " << maximum_error << '\n';
}

void SilenceTest(Gpu &gpu) {
    const std::vector<float> zero(65), impulse = [] {
        std::vector<float> values(65);
        values[0] = .2f;
        return values;
    }();
    const auto input = Upload<float>(gpu, zero);
    for (const auto &target : {zero, impulse}) {
        const auto state = CreateSpectralLossGpu(gpu, target, 48000);
        Evaluate(gpu, state, input);
        Require(std::ranges::all_of(BufferSpan<float>(state.Loss), [](float value) { return std::isfinite(value) && value >= 0; }), "Silence has finite nonnegative loss");
        Require(std::ranges::all_of(BufferSpan<float>(state.Gradient), [](float value) { return value == 0; }), "Smoothed magnitude has zero gradient at exact silence");
        Require(target[0] == 0 ? BufferSpan<float>(state.Loss)[0] == 0 : BufferSpan<float>(state.Loss)[0] > 0, "Silent target distinction");
    }
    bool rejected = false;
    try {
        CreateSpectralLossGpu(gpu, {}, 48000);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Empty spectral target rejected");
}

void OptionsTest(Gpu &gpu) {
    const SpectralLossOptions options{.MagnitudeFloor = .02f, .HuberDelta = .25f, .HopDivisor = 2};
    for (uint32_t samples : {1u, 63u, 64u, 65u, 257u}) {
        std::vector<float> target(samples), waveform(samples);
        for (uint32_t sample = 0; sample < samples; ++sample) {
            target[sample] = .03f * std::cos(.137f * sample);
            waveform[sample] = .02f * std::cos(.173f * sample);
        }
        const auto state = CreateSpectralLossGpu(gpu, target, 48000, options);
        const auto input = Upload<float>(gpu, waveform);
        Evaluate(gpu, state, input);
        const auto reference = ReferenceTarget(target, options);
        std::vector<double> values(waveform.begin(), waveform.end());
        const auto expected = ReferenceLoss(values, reference, options);
        for (uint32_t index = 0; index < 5; ++index) Require(std::abs(BufferSpan<float>(state.Loss)[index] - expected[index]) < 2e-5 * std::max(1., expected[index]), "Custom floor, Huber delta and hop match scalar short-signal loss");
        values.back() += 1e-6;
        const double plus = ReferenceLoss(values, reference, options)[0];
        values.back() -= 2e-6;
        const double minus = ReferenceLoss(values, reference, options)[0];
        const double numeric = (plus - minus) / 2e-6, analytic = BufferSpan<float>(state.Gradient).back();
        Require(std::abs(numeric - analytic) < .001 * std::max(1., std::abs(numeric)), "Custom spectral options sample adjoint");
    }
}

void NarrowbandTest(Gpu &gpu) {
    constexpr uint32_t samples = 4096;
    for (double bin : {64., 64.125}) {
        std::vector<float> target(samples), waveform(samples);
        for (uint32_t sample = 0; sample < samples; ++sample) {
            const double phase = 2 * std::numbers::pi * bin * sample / samples;
            target[sample] = float(.5 * std::sin(phase));
            waveform[sample] = float(.51 * std::sin(phase));
        }
        const auto state = CreateSpectralLossGpu(gpu, target, 48000);
        const auto input = Upload<float>(gpu, waveform);
        Evaluate(gpu, state, input);
        const std::vector<double> values(waveform.begin(), waveform.end());
        const auto reference = ReferenceTarget(target, {});
        const auto expected = ReferenceLoss(values, reference, {});
        const auto actual = BufferSpan<float>(state.Loss);
        double maximum_error = 0;
        for (uint32_t index = 0; index < 5; ++index) maximum_error = std::max(maximum_error, std::abs(actual[index] - expected[index]));
        std::cout << std::setprecision(12) << "Default-floor narrowband bin " << bin << " GPU loss " << actual[0] << ", direct DFT " << expected[0] << ", maximum part error " << maximum_error << '\n';
        Require(maximum_error < 2e-5, "Default-floor narrowband spectral loss versus direct DFT");
        // Perturb one edge sample so the oracle also probes near-floor bins.
        std::vector<double> plus(values), minus(values);
        plus.front() += 1e-10;
        minus.front() -= 1e-10;
        const double numeric = (ReferenceLoss(plus, reference, {})[0] - ReferenceLoss(minus, reference, {})[0]) / 2e-10;
        const double analytic = BufferSpan<float>(state.Gradient).front();
        const double error = std::abs(numeric - analytic) / std::max(1., std::abs(numeric));
        std::cout << "Narrowband adjoint analytic " << analytic << ", numeric " << numeric << ", relative error " << error << '\n';
        Require(error < .002, "Default-floor narrowband sample adjoint versus central finite difference");
    }
}

} // namespace

int main() {
    try {
        auto gpu = CreateGpu();
        std::cout << "Spectral on " << DeviceName(gpu) << '\n';
        for (auto scale : {SpectralMagnitudeScale::Decibels, SpectralMagnitudeScale::NaturalLog, SpectralMagnitudeScale::Linear}) ReferenceTest(gpu, scale);
        SilenceTest(gpu);
        OptionsTest(gpu);
        NarrowbandTest(gpu);
        std::cout << "Spectral tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
