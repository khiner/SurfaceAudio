#include "lee/Lee.h"
#include "core/SignalAnalysis.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>

using namespace surface_audio;
using namespace surface_audio::lee;
namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
double Relative(std::span<const float> a, std::span<const float> b) {
    Require(a.size() == b.size(), "Equal record dimensions");
    double error = 0, energy = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        error += std::pow(double(a[i]) - b[i], 2);
        energy += double(a[i]) * a[i];
    }
    return std::sqrt(error / std::max(energy, 1e-30));
}
void FrameBounds() {
    const auto rejected = [](const Analysis &analysis, double rate) {
        try {
            Synthesize(analysis, rate);
        } catch (const std::invalid_argument &error) { return std::string_view(error.what()).contains("frame count"); }
        return false;
    };
    const Settings no_tail{.TailSeconds = 0};
    Require(rejected({44100, UINT32_MAX, no_tail, {}, {}, {}}, .01), "Rate-expanded record overflow is rejected before allocation");
    const Settings long_tail{.TailSeconds = 10};
    Require(rejected({UINT32_MAX, 1, long_tail, {}, {}, {}}, 1), "Tail duration overflow is rejected before conversion");
    const Band silent{{1}, {}, 0};
    const std::array bands{silent, silent, silent, silent};
    Require(rejected({44100, 1, {}, {}, {}, {{0, UINT32_MAX, bands}}}, 1), "Contact plus tail overflow is rejected");
    Require(rejected({44100, 1, no_tail, {}, {}, {{UINT32_MAX, 1, bands}}}, 1), "Contact endpoint overflow is rejected");
}
void Bank() {
    std::vector<double> signal(2048);
    uint32_t state = 41;
    for (auto &x : signal) {
        state = 1664525 * state + 1013904223;
        x = double(state) / 4294967296. - .5;
    }
    const auto filters = MakeFilterBank();
    const auto output = MergeBands(SplitBands(signal, filters), filters, signal.size(), true);
    double energy = 0, error = 0;
    for (size_t i = 0; i < signal.size(); ++i) {
        energy += signal[i] * signal[i];
        error += std::pow(signal[i] - output[i], 2);
    }
    Require(std::sqrt(error / energy) < .001, "Four-band delay-compensated QMF reconstruction");
}
void Notches() {
    std::vector<double> impulse(8192);
    impulse[0] = 1;
    const std::array notches{Notch{.7, .02}, Notch{1.9, .04}};
    const auto filtered = FilterNotches(impulse, notches, false);
    const auto inverse = FilterNotches(filtered, notches, true);
    double error = 0;
    for (size_t n = 0; n < impulse.size(); ++n) error = std::max(error, std::abs(impulse[n] - inverse[n]));
    Require(error < 1e-12, "Notch and inverse cancel including full tails");
    auto spectrum = FourierTransform(filtered, 8192);
    for (size_t k = 0; k < spectrum.size(); k += 17) {
        const auto z = std::polar(1., -2 * std::numbers::pi * k / spectrum.size());
        std::complex<double> expected = 1;
        for (auto notch : notches) {
            const double r = std::exp(-notch.Bandwidth / 2);
            expected *= (1. - 2 * r * std::cos(notch.Frequency) * z + r * r * z * z) /
                (1. - 1.9 * r * std::cos(notch.Frequency) * z + .9025 * r * r * z * z);
        }
        Require(std::abs(expected - spectrum[k]) < 2e-12, "Independent Eq9-11 complex transfer response");
    }
    const auto recovered = EstimateNotches(filtered, 3, 2);
    Require(recovered.size() == 2, "Recover two spectral notches");
    Require(std::abs(recovered[0].Frequency - .7) < .001 && std::abs(recovered[1].Frequency - 1.9) < .001, "Parabolic notch frequency");
}
void DetectorBoundaries() {
    std::vector<float> cropped(4096, .25f);
    cropped[2048] += .02f;
    std::vector<double> envelope;
    const Settings settings;
    const auto onsets = DetectContacts(cropped, 44100, settings, envelope);
    Require(onsets.size() == 1 && std::abs(int(onsets[0]) - 2048) <= 10, "Crop discontinuities do not create or suppress an interior contact");
    const size_t margin = settings.HighpassTaps / 2 + settings.EnvelopeFrames / 2;
    Require(std::ranges::all_of(std::span(envelope).first(margin), [](double x) { return x == 0; }), "Detector excludes incomplete filter support");
    cropped[2048] -= .02f;
    Require(DetectContacts(cropped, 44100, settings, envelope).empty(), "A cropped constant has no contact events");
}
void NotchIsolation() {
    std::vector<double> impulse(8192);
    impulse[0] = 1;
    const std::array unresolved{Notch{.7, .01}, Notch{.72, .01}, Notch{std::numbers::pi - .01, .001}};
    const auto input = FilterNotches(impulse, unresolved, false);
    const auto notches = EstimateNotches(input, 3, 12);
    Require(notches.size() == 1, "Overlapping valleys form one feature and the Nyquist-edge valley is excluded");
    Require(notches[0].Frequency > .69 && notches[0].Frequency < .73, "Retain the resolved interior valley cluster");
    const auto whitened = FilterNotches(input, notches, true);
    double input_energy = 0, whitened_energy = 0;
    for (size_t n = 0; n < input.size(); ++n) {
        input_energy += input[n] * input[n];
        whitened_energy += whitened[n] * whitened[n];
    }
    Require(whitened_energy < 2 * input_energy, "Unresolved edge valleys cannot create an overlapping inverse-notch cascade");
}
void Pipeline() {
    std::vector<float> input(8192);
    for (uint32_t onset : {512u, 2048u, 4096u, 6144u}) {
        input[onset] += .2f;
        for (size_t n = 0; n < 1800 && onset + n < input.size(); ++n)
            input[onset + n] += float(.03 * std::exp(-double(n) / 400) * std::sin(.11 * n));
    }
    const Settings options{.TailSeconds = .02};
    const auto analysis = Analyze(input, 44100, options);
    Require(analysis.Onsets.size() == 4, "Four isolated contacts detected");
    for (size_t i = 0; i < 4; ++i)
        Require(std::abs(int(analysis.Onsets[i]) - int(std::array{512, 2048, 4096, 6144}[i])) <= 10, "Delay-compensated contact timing");
    const auto cpu = Synthesize(analysis);
    Require(std::ranges::all_of(cpu, [](float x) { return std::isfinite(x); }), "Full record finite");
    Require(cpu == Synthesize(analysis), "Deterministic full-record resynthesis");
    const auto slower = Synthesize(analysis, .5);
    Require(slower.size() > cpu.size(), "Rate controls contact timing independently of resonances");
    auto gpu = CreateGpu();
    const auto actual = SynthesizeGpu(gpu, analysis);
    const auto accelerated = AnalyzeGpu(gpu, input, 44100, options);
    Require(accelerated.Onsets == analysis.Onsets, "GPU detector agrees on contact times");
    const auto complete = SynthesizeGpu(gpu, accelerated);
    Require(Relative(cpu, complete) < 5e-4, "Complete GPU analysis and synthesis agrees with double reference");
    const double relative = Relative(cpu, actual);
    Require(relative < 2e-6, "GPU complete-record agreement");
    std::cout << "Lee GPU relative error " << relative << '\n';
    std::vector<float> silence(4096);
    Require(Analyze(silence, 44100).Contacts.empty(), "Silence yields no contacts");
}
}
int main() {
    try {
        FrameBounds();
        Bank();
        Notches();
        DetectorBoundaries();
        NotchIsolation();
        Pipeline();
        std::cout << "Lee tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
