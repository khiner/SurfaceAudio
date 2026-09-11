#include "lagrange/Lagrange.h"
#include "core/GpuConvolution.h"
#include "core/GpuFft.h"
#include "core/GpuSparseConvolution.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::lagrange;
namespace {
constexpr uint32_t Rate = 44100, Frames = 2 * Rate;
constexpr double Pi = std::numbers::pi;
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
double Error(std::span<const float> actual, std::span<const float> expected) {
    Require(actual.size() >= expected.size(), "Truncated recovery output");
    double error = 0, energy = 0;
    for (size_t n = 0; n < actual.size(); ++n) {
        const double reference = n < expected.size() ? expected[n] : 0;
        error += std::pow(actual[n] - reference, 2);
        energy += reference * reference;
    }
    return std::sqrt(error / std::max(energy, 1e-30));
}
std::vector<float> Reference(std::span<const Trigger> contacts, std::span<const float> impact) {
    std::vector<double> accumulated(Frames);
    for (auto event : contacts)
        for (size_t n = 0; n < impact.size(); ++n) accumulated[event.Sample + n] += event.Amplitude * impact[n];
    return {accumulated.begin(), accumulated.end()};
}
std::vector<float> Envelope() {
    std::vector<float> shape(800);
    for (size_t n = 0; n < shape.size(); ++n) {
        const double x = double(n) / 4;
        shape[n] = std::exp(x * std::log(.89) + .5 * (std::lgamma(10 + x) - std::lgamma(10) - std::lgamma(x + 1)));
    }
    const double peak = *std::max_element(shape.begin(), shape.end());
    for (auto &value : shape) value /= peak;
    return shape;
}
std::vector<Trigger> Contacts(double low, double high) {
    std::vector<Trigger> contacts;
    for (double time = .15; time < 1.8; time += 1 / (low + (high - low) * (time - .15) / 1.65))
        contacts.push_back({uint32_t(std::round(time * Rate)), .8 + .2 * std::sin(contacts.size() * 2.399)});
    return contacts;
}
std::vector<Trigger> Extract(Gpu &gpu, std::span<const float> envelope, std::span<const float> shape, bool split) {
    if (!split) return FitTriggerAmplitudes(envelope, shape, DetectTriggers(DeconvolveEnvelopeGpu(gpu, envelope, shape)));
    auto events = DetectTriggers(DeconvolveEnvelopeSections(envelope, shape), .2);
    const size_t origin = size_t(std::max_element(shape.begin(), shape.end()) - shape.begin()) - 1;
    std::erase_if(events, [&](auto event) { return event.Sample < origin; });
    for (auto &event : events) event.Sample -= origin;
    return events;
}
double Report(Gpu &gpu, const char *case_name, const char *stage, std::span<const Trigger> truth,
              std::span<const Trigger> events, std::span<const float> impact, std::span<const float> reference) {
    const auto reconstruction = ConvolveFftGpu(gpu, TriggerSignal(events, Frames), impact);
    const double error = Error(reconstruction, reference);
    double near = 0, total = 0;
    for (auto event : events) {
        total += event.Amplitude;
        if (std::ranges::any_of(truth, [&](auto contact) { return std::abs(double(event.Sample) - contact.Sample) <= Rate * .0005; }))
            near += event.Amplitude;
    }
    Require(std::isfinite(error) && std::isfinite(total), "Nonfinite contact recovery");
    std::cout << case_name << ',' << stage << ',' << truth.size() << ',' << events.size() << ',' << near / std::max(total, 1e-30) << ',' << error << '\n';
    return error;
}
double Motion(Gpu &gpu, std::span<const float> reference, std::span<const float> actual) {
    constexpr uint32_t width = 4096, hop = 441, count = (Frames - width) / hop + 1, first = 8, last = 371, bins = last - first + 1;
    std::vector<std::array<float, 2>> input(2 * count * width);
    for (uint32_t which = 0; which < 2; ++which)
        for (uint32_t frame = 0; frame < count; ++frame)
            for (uint32_t n = 0; n < width; ++n)
                input[(which * count + frame) * width + n][0] =
                    (which ? actual : reference)[frame * hop + n] * (.5 - .5 * std::cos(2 * Pi * n / width));
    const auto plan = CreateFftGpu(gpu, width, 2 * count);
    const auto buffer = Upload<std::array<float, 2>>(gpu, input);
    BeginGpu(gpu);
    EncodeFftGpu(gpu, plan, buffer);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto spectrum = BufferSpan<std::array<float, 2>>(plan.Output);
    std::array<std::vector<double>, 2> features{std::vector<double>(count * bins), std::vector<double>(count * bins)};
    std::array<double, count> energy{};
    for (uint32_t which = 0; which < 2; ++which) {
        double peak = 0;
        for (uint32_t frame = 0; frame < count; ++frame)
            for (uint32_t k = 0; k <= width / 2; ++k) {
                const auto value = spectrum[(which * count + frame) * width + k];
                const double power = double(value[0]) * value[0] + double(value[1]) * value[1];
                peak = std::max(peak, power);
                if (!which) energy[frame] += power;
                if (k >= first && k <= last) features[which][frame * bins + k - first] = power;
            }
        for (auto &value : features[which]) value = std::log(std::max(value, std::max(peak * 1e-8, 1e-30)));
    }
    const double threshold = *std::max_element(energy.begin(), energy.end()) * .01;
    for (auto &values : features) {
        std::array<double, bins> means{};
        uint32_t active = 0;
        for (uint32_t frame = 0; frame < count; ++frame) {
            if (energy[frame] <= threshold) continue;
            double mean = 0;
            for (uint32_t k = 0; k < bins; ++k) mean += values[frame * bins + k] / bins;
            for (uint32_t k = 0; k < bins; ++k) means[k] += values[frame * bins + k] -= mean;
            ++active;
        }
        for (uint32_t frame = 0; frame < count; ++frame)
            for (uint32_t k = 0; k < bins; ++k)
                values[frame * bins + k] = energy[frame] > threshold ? values[frame * bins + k] - means[k] / active : 0;
    }
    double cross = 0, left = 0, right = 0;
    for (size_t n = 0; n < features[0].size(); ++n) {
        cross += features[0][n] * features[1][n];
        left += features[0][n] * features[0][n];
        right += features[1][n] * features[1][n];
    }
    return cross / std::max(std::sqrt(left * right), 1e-30);
}
void Recovery(Gpu &gpu) {
    const auto shape = Envelope();
    std::vector<float> impact(shape.size());
    for (size_t n = 0; n < impact.size(); ++n) impact[n] = shape[n] * std::cos(2 * Pi * 1100 * n / Rate);
    const auto fitted_shape = FitImpactEnvelope(impact);
    struct Case { const char *Name; double Low, High; };
    std::cout << "case,stage,true_events,estimated_events,mass_within_0.5ms,relative_waveform_error\n";
    for (const auto [name, low, high] : std::array{Case{"sparse", 20, 20}, Case{"medium", 80, 80}, Case{"dense", 200, 200},
                                                Case{"very_dense", 600, 600}, Case{"sweep", 100, 600}}) {
        const auto truth = Contacts(low, high);
        const auto excitation = Reference(truth, impact), ideal = Reference(truth, shape), observed = PeakEnvelope(excitation);
        for (bool split : {false, true}) {
            const auto exact = Extract(gpu, ideal, shape, split);
            const double error = Report(gpu, name, split ? "ideal_split" : "ideal_regular", truth, exact, impact, excitation);
            if (!split && high <= 200) Require(error < .002, "Known additive envelopes fail isolated contact recovery");
            Report(gpu, name, split ? "observed_split" : "observed_regular", truth, Extract(gpu, observed, shape, split), impact, excitation);
            Report(gpu, name, split ? "fitted_split" : "fitted_regular", truth, Extract(gpu, observed, fitted_shape, split), impact, excitation);
        }
        const auto fit = FitSparseConvolutionGpu(gpu, excitation, impact);
        std::vector<Trigger> fitted;
        for (uint32_t n = 0; n < Frames; ++n) if (fit.Coefficients[n] > 0) fitted.push_back({n, fit.Coefficients[n]});
        Require(Report(gpu, name, "waveform", truth, fitted, impact, excitation) < .01, "Waveform fit loses known overlapping impacts");
        if (low != high) {
            const auto fitted_signal = ConvolveFftGpu(gpu, fit.Coefficients, impact);
            Require(Motion(gpu, excitation, fitted_signal) > .99, "Known waveform recovery loses spectral motion");
            Require(std::abs(Motion(gpu, excitation, excitation) - 1) < 1e-10, "Motion diagnostic fails identity");
            const std::vector<float> reversed(excitation.rbegin(), excitation.rend());
            // Independent SciPy STFT comparison of the forward and reversed sweep.
            Require(std::abs(Motion(gpu, excitation, reversed) + .08304023914447536) < 1e-5, "Metal motion diagnostic differs from SciPy");
        }

    }
}
void MovingFilter(Gpu &gpu) {
    const auto shape = Envelope();
    const auto truth = Contacts(200, 200);
    const auto source = Reference(truth, shape);
    std::vector<float> path(Frames), expected(Frames), recovered(Frames);
    const auto sample = [](std::span<const float> data, double position) {
        if (position < 0) return 0.;
        const size_t index = size_t(position);
        return std::lerp(double(data[index]), double(data[std::min(index + 1, data.size() - 1)]), position - index);
    };
    for (uint32_t n = 0; n < Frames; ++n) {
        path[n] = std::lerp(.25f, .5f, float(n) / Frames);
        expected[n] = source[n] + .6 * sample(source, double(n) - double(path[n]) * Rate / 1000);
    }
    const auto actual = MovingCombGpu(gpu, source, path, 1, 1000, Rate, .6f, 0);
    Require(Error(actual, expected) < 2e-4, "Moving filter differs from independent delayed sum");
    for (uint32_t n = 0; n < Frames; ++n)
        recovered[n] = actual[n] - .6 * sample(recovered, double(n) - double(path[n]) * Rate / 1000);
    Require(Error(recovered, source) < 2e-4, "Known moving filter fails causal recovery");
    for (auto &value : recovered) value = std::max(0.f, value);
    const auto uncorrected = Extract(gpu, actual, shape, false), corrected = Extract(gpu, recovered, shape, false);
    Report(gpu, "moving_filter", "uncorrected", truth, uncorrected, shape, source);
    Require(Report(gpu, "moving_filter", "known_path_removed", truth, corrected, shape, source) < .01, "Known motion corrupts contact recovery");
}
void ModalEstimation(Gpu &gpu) {
    const std::array modes{Resonance{811.1, 30, .6, .1}, Resonance{1811.7, 70, -.4, .2}, Resonance{4501.9, 110, .2, -.3}};
    std::vector<float> response(32768);
    for (size_t n = 0; n < response.size(); ++n) {
        double value = 0;
        for (auto mode : modes) {
            const double angle = 2 * Pi * mode.Frequency * n / Rate;
            value += std::exp(-mode.Damping * n / Rate) * (mode.RealGain * std::cos(angle) - mode.ImaginaryGain * std::sin(angle));
        }
        response[n] = value;
    }
    const auto shape = Envelope();
    const auto excitation = Reference(Contacts(100, 600), shape);
    const auto recording = ConvolveFftGpu(gpu, excitation, response);
    auto estimated = AnalyzeModes(response, Rate, {100, 0, 1114}, {.Bands = 8, .ModesPerBand = 6, .MaximumModes = 12, .WhiteningOrder = 0});
    for (auto &mode : estimated) {
        const auto gain = std::complex{mode.RealGain, mode.ImaginaryGain} *
                          std::exp(std::complex{mode.Damping, -2 * Pi * mode.Frequency} * (100. / Rate));
        mode.RealGain = gain.real();
        mode.ImaginaryGain = gain.imag();
    }
    const double known_error = Error(InverseModalGpu(gpu, recording, modes, Rate), excitation);
    const double estimated_error = Error(InverseModalGpu(gpu, recording, estimated, Rate), excitation);
    Require(known_error < .001 && std::isfinite(estimated_error), "Known modal filter fails source recovery");
    std::cout << "modal_inverse,known," << known_error << ",estimated," << estimated_error << '\n';
}

}
int main() {
    try {
        std::cout << std::setprecision(8);
        auto gpu = CreateGpu();
        Recovery(gpu);
        MovingFilter(gpu);
        ModalEstimation(gpu);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
