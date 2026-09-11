#pragma once

#include "core/Gpu.h"
#include <complex>
#include <vector>

namespace surface_audio::lagrange {
struct Resonance {
    double Frequency{}, Damping{}, RealGain{}, ImaginaryGain{};
};
struct Interval {
    uint32_t Begin{}, Peak{}, End{};
};
struct Trigger {
    uint32_t Sample{};
    double Amplitude{};
};
enum class ModalGain { Fitted, Unit, Magnitude, CausalMagnitude, IncrementalPhase };
enum class TriggerFit { Envelope, Waveform, SplitEnvelope, JointWaveform };
struct AnalysisOptions {
    uint32_t Bands{8}, ModesPerBand{20}, MaximumModes{80}, WhiteningOrder{8};
    double WindowSeconds{.023}, InverseFloor{1e-8}, EnvelopeFloor{.03};
    ModalGain Gains{ModalGain::Fitted};
    Interval ModalInterval{}, ImpactInterval{};
    TriggerFit Fit{TriggerFit::Envelope};
    bool CenterImpact{};
    double AudioWeight{.01};
};
struct Analysis {
    uint32_t SampleRate{}, Frames{};
    Interval ModalInterval{}, ImpactInterval{};
    std::vector<Resonance> Modes{};
    std::vector<float> Excitation{}, Impact{}, Envelope{}, ImpactEnvelope{}, Deconvolved{};
    std::vector<Trigger> Triggers{};
    double SourceFitKkt{};
};

std::vector<double> Meixner(uint32_t frames = 200, double beta = 10, double gamma = .89);
std::vector<float> PeakEnvelope(std::span<const float>);
Interval SelectInterval(std::span<const float>, uint32_t sample_rate, uint32_t requested_frames = 0);
std::vector<float> FitImpactEnvelope(std::span<const float>);
std::vector<float> DeconvolveEnvelope(std::span<const float>, std::span<const float>, double relative_floor = .03);
std::vector<float> DeconvolveEnvelopeSections(std::span<const float>, std::span<const float>);
std::vector<float> DeconvolveEnvelopeGpu(Gpu &, std::span<const float>, std::span<const float>, double relative_floor = .03);
std::vector<Trigger> FitTriggerAmplitudes(std::span<const float>, std::span<const float>, std::span<const Trigger>, uint32_t iterations = 1500);
std::vector<Trigger> DetectTriggers(std::span<const float>, double minimum_peak_fraction = 0);
std::vector<float> TriggerSignal(std::span<const Trigger>, uint32_t frames);
std::complex<double> Transfer(std::span<const Resonance>, double radians, double sample_rate);
std::vector<Resonance> AdaptModalPhases(std::span<const Resonance>, double sample_rate);
std::vector<float> ModalResponse(std::span<const Resonance>, uint32_t frames, double sample_rate);
std::vector<float> ModalResponseGpu(Gpu &, std::span<const Resonance>, uint32_t frames, double sample_rate);
// Returns the requested prefix of causal modal filtering, with zero excitation beyond the input.
std::vector<float> FilterModalGpu(Gpu &, std::span<const float>, std::span<const Resonance>, uint32_t frames, double sample_rate);
std::vector<Resonance> AnalyzeModes(std::span<const float>, uint32_t sample_rate, Interval, const AnalysisOptions & = {});
Analysis Analyze(Gpu &, std::span<const float>, uint32_t sample_rate, const AnalysisOptions & = {});
std::vector<float> Synthesize(Gpu &, const Analysis &, std::span<const Trigger>);
std::vector<float> InverseModalGpu(Gpu &, std::span<const float>, std::span<const Resonance>, uint32_t sample_rate, double floor = 1e-8);
std::vector<float> MovingCombGpu(Gpu &, std::span<const float> signal, std::span<const float> position, double plate_length, double wave_speed, uint32_t sample_rate, float first_reflection, float second_reflection);
}
