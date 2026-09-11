#include "lagrange/Lagrange.h"
#include "core/SignalAnalysis.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::lagrange;
namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
void MeixnerEquation() {
    const auto window = Meixner();
    double norm = 0;
    for (size_t n = 0; n < window.size(); ++n) {
        const double expected = std::pow(1 - .89 * .89, 5) * std::exp(.5 * (std::lgamma(10 + n) - std::lgamma(10) - std::lgamma(n + 1))) * std::pow(.89, n);
        Require(std::abs(window[n] - expected) < 2e-14, "Meixner recurrence differs from independent gamma-function Eq6-8");
        norm += window[n] * window[n];
    }
    Require(std::abs(norm - 1) < 1e-10, "Meixner squared envelope normalizes to one");
    const std::array<float, 6> constant{2, 2, 2, 2, 2, 2};
    Require(PeakEnvelope(constant) == std::vector<float>(6, 2), "Spline preserves constant envelope");
}
void EnvelopeInverse() {
    const std::array<float, 5> shape{.2f, .7f, 1.f, .6f, .1f};
    const std::array planted{Trigger{20, .7}, Trigger{23, .3}, Trigger{200, 1.2}, Trigger{400, .03}, Trigger{510, .8}};
    std::vector<float> envelope(512);
    for (auto event : planted)
        for (size_t n = 0; n < shape.size() && event.Sample + n < envelope.size(); ++n)
            envelope[event.Sample + n] += float(event.Amplitude * shape[n]);
    const std::array candidates{Trigger{20, 1}, Trigger{23, 1}, Trigger{100, 1}, Trigger{200, 1}, Trigger{400, 1}, Trigger{510, 1}};
    const auto fitted = FitTriggerAmplitudes(envelope, shape, candidates);
    for (auto event : planted) {
        const auto recovered = std::ranges::find(fitted, event.Sample, &Trigger::Sample);
        Require(recovered != fitted.end() && std::abs(recovered->Amplitude - event.Amplitude) < 2e-6,
                "Nonnegative amplitudes fail overlapping, weak or right-truncated planted impacts");
    }
    const std::array noisy_candidates{Trigger{20, 1}, Trigger{21, 1}, Trigger{23, 1}, Trigger{100, 1}, Trigger{200, 1}, Trigger{400, 1}, Trigger{510, 1}};
    for (size_t n = 0; n < envelope.size(); ++n) envelope[n] += float(.005 * (1 + std::sin(.1 * n)));
    const auto noisy = TriggerSignal(FitTriggerAmplitudes(envelope, shape, noisy_candidates), uint32_t(envelope.size()));
    const std::array scipy_amplitudes{.7114096927542075, 0., .3098392377593684, .002132572348231912,
                                      1.2134696006691439, .04097539443914227, .8146483500594415};
    for (size_t j = 0; j < noisy_candidates.size(); ++j)
        Require(std::abs(noisy[noisy_candidates[j].Sample] - scipy_amplitudes[j]) < 3e-6, "Amplitude fit differs from independent SciPy dense NNLS fixture");
    for (auto candidate : noisy_candidates) {
        double gradient = 0;
        for (size_t n = candidate.Sample; n < std::min(envelope.size(), candidate.Sample + shape.size()); ++n) {
            double residual = -envelope[n];
            for (auto other : noisy_candidates)
                if (n >= other.Sample && n < other.Sample + shape.size()) residual += noisy[other.Sample] * shape[n - other.Sample];
            gradient += residual * shape[n - candidate.Sample];
        }
        Require(noisy[candidate.Sample] > 0 ? std::abs(gradient) < 2e-6 : gradient > -2e-6, "Nonnegative fit violates independently evaluated KKT conditions");
    }
    const std::array<float, 5> invertible_shape{.2f, .7f, 1.f, .6f, .17f};
    std::vector<float> complete(1024);
    for (auto event : planted)
        for (size_t n = 0; n < invertible_shape.size(); ++n) complete[event.Sample + n] += float(event.Amplitude * invertible_shape[n]);
    const auto inverse = DeconvolveEnvelope(complete, invertible_shape, 1e-6);
    const auto expected = TriggerSignal(planted, uint32_t(complete.size()));
    double error = 0, energy = 0;
    for (size_t n = 0; n < complete.size(); ++n) {
        error += std::pow(double(inverse[n]) - expected[n], 2);
        energy += double(expected[n]) * expected[n];
    }
    Require(std::sqrt(error / energy) < 2e-6, "Envelope inverse fails actual independent E=T*w construction");
    const auto detected = DetectTriggers(expected);
    Require(std::ranges::any_of(detected, [](auto event) { return event.Sample == 400; }), "Global maximum suppresses weak later contact");
}
void SplitEnvelopeInverse() {
    const std::array<float, 5> shape{.2f, .6f, 1, .5f, .1f};
    const std::array<float, 9> input{.1f, .2f, .4f, .8f, 1, .7f, .5f, .1f, .03f};
    const std::array expected{.11498162867777523, .15505510539987555, .28483467224786796, .72049596128950166,
                              .97601206170530574, .34571366575981594, .512233898806288, -.13876422999556789, .062386453538507185};
    const auto actual = DeconvolveEnvelopeSections(input, shape);
    for (size_t n = 0; n < input.size(); ++n)
        Require(std::abs(actual[n] - expected[n]) < 5e-8, "Split envelope inverse differs from independent SciPy two-pass IIR");
    std::vector<float> signal(128);
    const std::array<float, 4> product{.2f, .7f, .32f, .06f};
    for (size_t n = 0; n < product.size(); ++n) signal[50 + n] = product[n];
    const auto recovered = DeconvolveEnvelopeSections(signal, shape);
    for (size_t n = 0; n < recovered.size(); ++n)
        Require(std::abs(recovered[n] - (n == 51 ? 1 : 0)) < 2e-7, "Split inverse fails factorized-window impulse or attack-origin alignment");
    auto scaled_shape = shape;
    for (auto &sample : scaled_shape) sample *= 2;
    const auto scaled = DeconvolveEnvelopeSections(input, scaled_shape);
    for (size_t n = 0; n < actual.size(); ++n)
        Require(scaled[n] == actual[n] / 4, "Two section inverses do not apply both section gains");
    std::vector<float> peaks(32);
    peaks[10] = 1000;
    peaks[12] = .01f;
    peaks[20] = 300;
    Require(DetectTriggers(peaks).size() == 2, "Weak-peak fixture does not exercise indicator masking");
    const auto iterative = DetectTriggers(peaks, .2);
    Require(iterative.size() == 3 && iterative[1].Sample == 12 && iterative[1].Amplitude == peaks[12],
            "Dominant-peak pruning loses a masked event or changes its amplitude");
    Require(DetectTriggers(std::vector<float>(10, 1), .2).empty(), "Flat residual pruning does not terminate");
}

void ModalPhaseAdaptation() {
    constexpr double rate = 44100, pi = std::numbers::pi;
    const std::array pair{Resonance{1100, 4, .8, 0}, Resonance{1600, 9, .3, 0}};
    const auto adapted = AdaptModalPhases(pair, rate);
    Require(std::abs(adapted[1].RealGain - .014356550065806281) < 1e-12 &&
            std::abs(adapted[1].ImaginaryGain + .29965628555097584) < 1e-12, "IPA differs from independent SciPy equal-magnitude root");
    const auto pole_response = [&](Resonance mode, double radians) {
        return std::complex{mode.RealGain, mode.ImaginaryGain} /
               (1. - std::polar(std::exp(-mode.Damping / rate), 2 * pi * mode.Frequency / rate - radians));
    };
    const double valley = .20853416084422116;
    const auto first = pole_response(adapted[0], valley), second = pole_response(adapted[1], valley);
    Require(std::abs(std::arg(second / first) - pi / 2) < 1e-12, "Adjacent IPA responses fail Eq12 quadrature");
    Require(std::abs(std::norm(first + second) - std::norm(first) - std::norm(second)) < 1e-9, "IPA valley retains destructive interference");
    const std::array close{Resonance{1000, .01, 1, 0}, Resonance{1001, .01, 1, 0}};
    const auto improved = AdaptModalPhases(close, rate);
    const double middle = 2 * pi * 1000.5 / rate;
    Require(std::abs(Transfer(improved, middle, rate)) > 100 * std::abs(Transfer(close, middle, rate)), "IPA fails to reduce a planted inverse-filter antiresonance");
    const std::array unequal{Resonance{1000, .01, 1, 0}, Resonance{1001, .1, 1e-12, 0}};
    const auto boundary = AdaptModalPhases(unequal, rate);
    Require(std::abs(std::arg(std::complex{boundary[1].RealGain, boundary[1].ImaginaryGain}) - .0016627860131228633) < 1e-10,
            "IPA without an Eq13 crossing differs from independent Eq10 minimization");
    const std::array reversed{Resonance{1600, 9, -.6, 0}, Resonance{1400, 2, 0, 0}, Resonance{1100, 4, 0, 1.6}};
    const auto sorted = AdaptModalPhases(reversed, rate);
    Require(sorted[0].Frequency == 1100 && sorted[1].Frequency == 1400 && sorted[2].Frequency == 1600, "IPA requires sorted caller data");
    for (size_t k = 0; k < adapted.size(); ++k) {
        const auto mode = sorted[k * 2];
        Require(std::abs(mode.RealGain - 2 * adapted[k].RealGain) < 1e-12 &&
                std::abs(mode.ImaginaryGain - 2 * adapted[k].ImaginaryGain) < 1e-12, "IPA depends on common gain, input phase or zero-gain modes");
        Require(mode.Damping == pair[k].Damping && mode.Frequency == pair[k].Frequency, "IPA changes modal poles");
    }
}
void ModalEquations(Gpu &gpu) {
    const std::array<float, 7> shape{.1f, .3f, .6f, 1.f, .7f, .2f, .03f};
    std::vector<float> envelope(44117);
    for (size_t n = 0; n < envelope.size(); ++n) envelope[n] = float(.7 + .2 * std::sin(.01 * n) + .1 * std::cos(.003 * n));
    const auto envelope_cpu = DeconvolveEnvelope(envelope, shape), envelope_gpu = DeconvolveEnvelopeGpu(gpu, envelope, shape);
    double inverse_error = 0, inverse_energy = 0;
    for (size_t n = 0; n < envelope.size(); ++n) {
        inverse_error += std::pow(double(envelope_gpu[n]) - envelope_cpu[n], 2);
        inverse_energy += double(envelope_cpu[n]) * envelope_cpu[n];
    }
    Require(std::sqrt(inverse_error / inverse_energy) < 3e-6, "Full-record Metal regularized envelope inverse differs from FP64 FFT");

    const std::array modes{Resonance{631.17, 20, .3, -.2}, Resonance{4012.13, 47, -.4, .1}, Resonance{19001.137, .03, .01, .05}};
    constexpr uint32_t frames = 65536, rate = 44100;
    const auto cpu = ModalResponse(modes, frames, rate), actual = ModalResponseGpu(gpu, modes, frames, rate);
    double error = 0, energy = 0;
    for (size_t n = 0; n < cpu.size(); ++n) {
        error += std::pow(double(cpu[n]) - actual[n], 2);
        energy += double(cpu[n]) * cpu[n];
    }
    Require(std::sqrt(error / energy) < 2e-6, "Complete long phase-aware GPU response differs from independent FP64 sum");
    std::vector<float> signal(10001);
    for (size_t n = 0; n < signal.size(); ++n) signal[n] = float(std::sin(.23 * n) * std::exp(-.0003 * n) + .02 * std::cos(.91 * n));
    const auto recovered = InverseModalGpu(gpu, signal, modes, rate);
    const uint32_t size = std::bit_ceil(uint32_t(signal.size()) * 2);
    std::vector<std::complex<double>> spectrum(size);
    for (size_t n = 0; n < signal.size(); ++n) spectrum[n] = signal[n];
    FourierTransform(spectrum);
    for (uint32_t n = 0; n < size; ++n) spectrum[n] /= Transfer(modes, 2 * std::numbers::pi * n / size, rate);
    FourierTransform(spectrum, true);
    error = energy = 0;
    for (size_t n = 0; n < signal.size(); ++n) {
        error += std::pow(spectrum[n].real() - recovered[n], 2);
        energy += std::norm(spectrum[n]);
    }
    Require(std::sqrt(error / energy) < 2e-4, "Complete Eq4-5 GPU inverse differs from FP64 complex spectral division");
    std::vector<float> impulse(128), location(128, .25f);
    impulse[5] = 1;
    const auto comb = MovingCombGpu(gpu, impulse, location, 1, 100, 1000, .5f, -.25f);
    std::vector<float> comb_reference(128);
    comb_reference[5] = 1;
    comb_reference[7] = comb_reference[8] = .25f;
    comb_reference[12] = comb_reference[13] = -.125f;
    Require(comb == comb_reference, "Two-edge fractional delays differ from independent delayed impulse construction");
    for (size_t n = 0; n < location.size(); ++n) location[n] = float(n) / (location.size() - 1);
    const auto moving = MovingCombGpu(gpu, impulse, location, 1, 100, 1000, .5f, -.25f);
    for (size_t n = 0; n < moving.size(); ++n) {
        double expected = impulse[n];
        for (auto pair : std::array{std::pair{double(location[n]) * 10, .5}, std::pair{(1 - double(location[n])) * 10, -.25}}) {
            const double position = n - pair.first;
            const int low = int(std::floor(position));
            const auto value = [&](int index) { return index >= 0 && size_t(index) < impulse.size() ? impulse[size_t(index)] : 0; };
            expected += pair.second * std::lerp(value(low), value(low + 1), position - low);
        }
        Require(std::abs(moving[n] - expected) < 3e-7, "Moving comb violates controlled trajectory interpolation");
    }
}
void LargeEnvelopeInverse(Gpu &gpu) {
    const std::array<float, 1> shape{1};
    std::vector<float> envelope(1u << 21);
    for (size_t n = 0; n < envelope.size(); ++n) envelope[n] = float(.7 + .2 * std::sin(.01 * n));
    const auto result = DeconvolveEnvelopeGpu(gpu, envelope, shape);
    double error = 0, energy = 0;
    for (size_t n = 0; n < envelope.size(); ++n) {
        const double expected = envelope[n] / (1 + .03 * .03);
        error += std::pow(result[n] - expected, 2);
        energy += expected * expected;
    }
    Require(std::sqrt(error / energy) < 2e-6, "Large batched FFT inverse exceeds command parameter capacity or differs from scalar oracle");
}
void ModalDecay(Gpu &gpu) {
    constexpr uint32_t rate = 48000, frames = 12000, first = 100, second = first + 8008;
    const Resonance mode{3000, .08, .7, -.2};
    const std::complex<double> pole = std::exp(std::complex{-mode.Damping, 2 * std::numbers::pi * mode.Frequency} / double(rate));
    const std::array events{Trigger{first, 1}, Trigger{second, std::exp(-mode.Damping * (second - first) / rate)}};
    const Analysis analysis{.SampleRate = rate, .Frames = frames, .Modes = {mode}, .Impact = {1}};
    const auto actual = Synthesize(gpu, analysis, events), input = TriggerSignal(events, frames);
    const auto prefix = FilterModalGpu(gpu, input, analysis.Modes, frames, rate);
    Require(actual.size() == 2 * frames - 1, "Modal rendering changes the requested duration");
    std::complex<double> state{};
    double error = 0, energy = 0, tail_peak = 0;
    for (size_t n = 0; n < actual.size(); ++n) {
        state = pole * state + (n < input.size() ? double(input[n]) : 0);
        const double expected = (std::complex{mode.RealGain, mode.ImaginaryGain} * state).real();
        error += std::pow(actual[n] - expected, 2);
        energy += expected * expected;
        if (n >= frames) tail_peak = std::max(tail_peak, std::abs(double(actual[n])));
        else Require(std::abs(actual[n] - prefix[n]) < 2e-6, "Output horizon changes the causal prefix");
    }
    Require(std::sqrt(error / energy) < 2e-6, "Modal rendering differs from the independent causal recurrence");
    Require(tail_peak < 2e-6, "Truncated modal response reintroduces a cancelled resonance after the input ends");
}
void MovingCombNotches(Gpu &gpu) {
    constexpr uint32_t rate = 48000, frames = 48000;
    constexpr double pi = std::numbers::pi;
    std::vector<float> signal(frames), position(frames);
    for (uint32_t n = 0; n < frames; ++n) {
        signal[n] = float(std::cos(2 * pi * 1000 * n / rate) + std::cos(2 * pi * 2000 * n / rate));
        position[n] = n < frames / 2 ? .25f : .5f;
    }
    const auto output = MovingCombGpu(gpu, signal, position, 1, 1000, rate, 1, 0);
    const auto amplitude = [&](uint32_t begin, uint32_t end, double frequency) {
        std::complex<double> coefficient{};
        for (uint32_t n = begin; n < end; ++n) coefficient += double(output[n]) * std::polar(1., -2 * pi * frequency * n / rate);
        return 2 * std::abs(coefficient) / (end - begin);
    };
    // Stoelinga Eq4.10 moves the first notch from c/(2*l)=2000 Hz to 1000 Hz as the path doubles.
    Require(amplitude(4800, 19200, 2000) < 1e-6 && amplitude(28800, 43200, 1000) < 1e-6, "Moving reflection fails predicted spectral notch positions");
    Require(std::abs(amplitude(4800, 19200, 1000) - std::sqrt(2.)) < 1e-6 &&
            std::abs(amplitude(28800, 43200, 2000) - 2) < 1e-6, "Moving reflection fails Eq4.9 passband gains");
}
void PoleRecovery() {
    constexpr uint32_t rate = 44100;
    const std::array modes{Resonance{811.1, 30, .6, .1}, Resonance{1811.7, 70, -.4, .2}, Resonance{4501.9, 110, .2, -.3}};
    const auto signal = ModalResponse(modes, 8192, rate);
    const auto estimated = AnalyzeModes(signal, rate, {100, 0, 1114}, {.Bands = 8, .ModesPerBand = 6, .MaximumModes = 12, .WhiteningOrder = 0});
    for (auto mode : modes) {
        const auto nearest = std::ranges::min_element(estimated, [&](auto a, auto b) { return std::abs(a.Frequency - mode.Frequency) < std::abs(b.Frequency - mode.Frequency); });
        Require(nearest != estimated.end() && std::abs(nearest->Frequency - mode.Frequency) < 5, "Subband ESPRIT failed planted frequency recovery");
        Require(std::abs(nearest->Damping - mode.Damping) / mode.Damping < .25, "Subband ESPRIT failed planted damping recovery");
    }
}
void GainConventions(Gpu &gpu) {
    const std::array modes{Resonance{811.1, 30, .6, .1}, Resonance{1811.7, 70, -.04, .02}, Resonance{4501.9, 110, .002, -.003}};
    const auto signal = ModalResponse(modes, 4096, 44100);
    const auto analyze = [&](ModalGain gains, bool center = false) {
        return Analyze(gpu, signal, 44100, {.Bands = 8, .ModesPerBand = 6, .MaximumModes = 12, .WhiteningOrder = 0,
                       .Gains = gains, .ModalInterval = {100, 0, 1114}, .ImpactInterval = {100, 0, 300}, .CenterImpact = center});
    };
    const auto fitted = analyze(ModalGain::Fitted), unit = analyze(ModalGain::Unit), magnitude = analyze(ModalGain::Magnitude), causal = analyze(ModalGain::CausalMagnitude);
    Require(fitted.Modes.size() == magnitude.Modes.size() && fitted.Modes.size() == unit.Modes.size(), "Gain convention changes mode selection");
    const auto centered = analyze(ModalGain::CausalMagnitude, true);
    double mean = 0, centered_mean = 0;
    for (float sample : causal.Impact) mean += sample / double(causal.Impact.size());
    for (size_t n = 0; n < causal.Impact.size(); ++n) {
        Require(std::abs(centered.Impact[n] - (causal.Impact[n] - mean)) < 1e-7, "Impact centering alters waveform beyond its DC offset");
        centered_mean += centered.Impact[n] / double(centered.Impact.size());
    }
    Require(std::abs(centered_mean) < 1e-8, "Centered impact has a DC offset");
    for (size_t n = 0; n < fitted.Modes.size(); ++n) {
        const auto a = fitted.Modes[n], b = magnitude.Modes[n], c = unit.Modes[n];
        Require(a.Frequency == b.Frequency && a.Damping == b.Damping, "Magnitude gains change modal poles");
        Require(std::abs(b.RealGain - std::hypot(a.RealGain, a.ImaginaryGain)) < 1e-14 && b.ImaginaryGain == 0,
                "Magnitude convention fails to preserve fitted strength while removing phase");
        Require(c.RealGain == 1 && c.ImaginaryGain == 0, "Unit convention changes");
        const auto d = causal.Modes[n];
        const double theta = 2 * std::numbers::pi * d.Frequency / 44100, radius = std::exp(-d.Damping / 44100);
        const std::array selected{d};
        const auto response = ModalResponse(selected, 200, 44100);
        double previous = 0, older = 0;
        for (size_t k = 0; k < response.size(); ++k) {
            const double current = 2 * radius * std::cos(theta) * previous - radius * radius * older +
                                   (k == 0 ? std::hypot(a.RealGain, a.ImaginaryGain) * std::sin(theta) : 0);
            Require(std::abs(response[k] - current) < 1e-6 * std::max(1., b.RealGain), "Causal gains differ from JASS recurrence");
            older = previous;
            previous = current;
        }
    }
}
void InvalidInputs(Gpu &gpu) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const std::array<float, 2> finite{0, 1}, nonfinite{0, nan}, position{.2f, .3f};
    const std::array mode{Resonance{100, 1, 1, 0}}, invalid{Resonance{100, infinity, 1, 0}};
    const auto rejected = [](auto action) {
        try {
            action();
        } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    Require(rejected([&] { DeconvolveEnvelopeGpu(gpu, finite, nonfinite); }), "NaN envelope shape accepted");
    Require(rejected([&] { DeconvolveEnvelope(finite, finite, infinity); }), "Infinite relative envelope floor accepted");
    Require(rejected([&] { MovingCombGpu(gpu, finite, position, 1, infinity, 44100, 1, 1); }), "Infinite comb wave speed accepted");
    Require(rejected([&] { MovingCombGpu(gpu, nonfinite, position, 1, 100, 44100, 1, 1); }), "NaN comb waveform accepted");
    Require(rejected([&] { ModalResponseGpu(gpu, invalid, 10, 44100); }), "Infinite modal damping accepted");
    Require(rejected([&] { ModalResponseGpu(gpu, mode, 10, infinity); }), "Infinite sample rate accepted");
    Require(rejected([&] { InverseModalGpu(gpu, finite, mode, 44100, infinity); }), "Infinite inverse floor accepted");
    Require(rejected([&] { InverseModalGpu(gpu, nonfinite, mode, 44100); }), "NaN modal waveform accepted");
    Require(rejected([&] { AnalyzeModes(nonfinite, 44100, {0, 0, 2}); }), "NaN modal analysis accepted");
    Require(rejected([&] { AdaptModalPhases(mode, infinity); }), "Infinite IPA sample rate accepted");
    Require(rejected([&] { AdaptModalPhases(invalid, 44100); }), "Nonfinite IPA damping accepted");
    Require(rejected([&] { AdaptModalPhases(std::array{mode[0], mode[0]}, 44100); }), "Coincident IPA modes accepted");
    Require(rejected([&] { DeconvolveEnvelopeSections(finite, nonfinite); }), "Nonfinite split envelope accepted");
    Require(rejected([&] { DeconvolveEnvelopeSections(finite, std::array<float, 3>{0, 1, .5f}); }), "Zero attack inverse accepted");
    Require(rejected([&] { DeconvolveEnvelopeSections(finite, std::array<float, 5>{.2f, .6f, 1, .2f, .3f}); }), "Non-unimodal split shape accepted");
    Require(rejected([&] { AnalyzeModes(finite, 44100, {0, 0, 2}, {.Bands = 1u << 30}); }), "Oversized subband product accepted");
}
}
int main() {
    try {
        MeixnerEquation();
        EnvelopeInverse();
        SplitEnvelopeInverse();
        ModalPhaseAdaptation();
        PoleRecovery();
        auto gpu = CreateGpu();
        InvalidInputs(gpu);
        GainConventions(gpu);
        ModalEquations(gpu);
        ModalDecay(gpu);
        MovingCombNotches(gpu);
        LargeEnvelopeInverse(gpu);
        std::cout << "Lagrange equations, recovered poles, source recovery and full GPU waveforms passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
