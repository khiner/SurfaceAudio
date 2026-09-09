#pragma once

#include "core/Random.h"

#ifdef __METAL_VERSION__
#define CONAN_THREAD thread
#define CONAN_CONSTANT constant
#else
#include <cmath>
#include <span>
#include <vector>
#define CONAN_THREAD
#define CONAN_CONSTANT
#endif

namespace surface_audio::conan {

CONAN_CONSTANT constexpr unsigned QuantileCount = 65;
CONAN_CONSTANT constexpr unsigned PulseCapacity = 64;
CONAN_CONSTANT constexpr float Pi = 3.14159265358979323846f;

struct Controls {
    float Size{0.5f};
    float Velocity{0.5f};
    float Roughness{0.5f};
    float Asymmetry{0.3f};
};

struct Process {
    float Mean{};
    float Sigma{};
    float A1{};
    float B1{};
    unsigned Empirical{};
    float Quantiles[QuantileCount]{};
};

struct Parameters {
    float SampleRate{48000};
    Process Amplitude{};
    Process Interval{};
    float DurationScale{0.000394f};
    float DurationExponent{0.29f};
    float ModulationHz{3};
    float ModulationDepth{0.3f};
    float MinimumAmplitude{0.0001f};
    float MinimumInterval{0.00025f};
    float MaximumDuration{0.008f};
};

struct FilterState {
    float Input{};
    float Output{};
};

struct State {
    RandomState Random{};
    FilterState Amplitude{};
    FilterState Interval{};
    float UntilEventSamples{};
    float Phase{};
    float PhaseError{};
    float PulseAgeSamples[PulseCapacity]{};
    float PulseDurationSamples[PulseCapacity]{};
    float PulseAmplitude[PulseCapacity]{};
    unsigned PulseCount{};
    unsigned Events{};
    unsigned AmplitudeClamps{};
    unsigned IntervalClamps{};
    unsigned DurationClamps{};
    unsigned PulseOverflows{};
};

struct Event {
    float Amplitude{};
    float Interval{};
};

inline float Filter(CONAN_THREAD const Process &process, CONAN_THREAD FilterState &state, float input) {
    const float output = input + process.B1 * state.Input - process.A1 * state.Output;
    state = {input, output};
    return output;
}

inline float Whiten(CONAN_THREAD const Process &process, CONAN_THREAD FilterState &state, float input) {
    const float output = input + process.A1 * state.Input - process.B1 * state.Output;
    state = {input, output};
    return output;
}

// Quadratic probability spacing resolves rare residual tails without enlarging GPU parameters.
inline float QuantileProbability(unsigned index) {
    const float position = static_cast<float>(index) / (QuantileCount - 1);
    return position <= .5f ? 2 * position * position : 1 - 2 * (1 - position) * (1 - position);
}

inline float Quantile(CONAN_THREAD const Process &process, float uniform) {
    if (uniform <= 0) return process.Quantiles[0];
    if (uniform >= 1) return process.Quantiles[QuantileCount - 1];
#ifdef __METAL_VERSION__
    const float position = uniform <= .5f ? sqrt(.5f * uniform) : 1 - sqrt(.5f * (1 - uniform));
#else
    const float position = uniform <= .5f ? std::sqrt(.5f * uniform) : 1 - std::sqrt(.5f * (1 - uniform));
#endif
    unsigned index = static_cast<unsigned>(position * (QuantileCount - 1));
    if (index >= QuantileCount - 1) return process.Quantiles[QuantileCount - 1];
    const float lower = QuantileProbability(index), upper = QuantileProbability(index + 1);
    return process.Quantiles[index] + (uniform - lower) / (upper - lower) * (process.Quantiles[index + 1] - process.Quantiles[index]);
}

// Abramowitz-Stegun 26.2.17, maximum absolute CDF error 7.5e-8 before float rounding.
inline float NormalCdf(float value) {
#ifdef __METAL_VERSION__
    const float absolute = metal::abs(value);
    const float density = 0.3989422804014327f * metal::exp(-0.5f * value * value);
#else
    const float absolute = std::abs(value);
    const float density = 0.3989422804014327f * std::exp(-0.5f * value * value);
#endif
    const float t = 1 / (1 + 0.2316419f * absolute);
    const float tail = density * t * (0.319381530f + t * (-0.356563782f + t * (1.781477937f + t * (-1.821255978f + t * 1.330274429f))));
    return value >= 0 ? 1 - tail : tail;
}

inline Event NextEvent(CONAN_THREAD const Parameters &parameters, CONAN_THREAD State &state) {
    const float gaussian = Normal(state.Random);
    // One shared innovation is essential: independent noise destroys the amplitude/timing correlation.
    const float uniform = (parameters.Amplitude.Empirical || parameters.Interval.Empirical) ? NormalCdf(gaussian) : 0;
    const float amplitude_input = parameters.Amplitude.Empirical ? Quantile(parameters.Amplitude, uniform) : parameters.Amplitude.Sigma * gaussian;
    const float interval_input = parameters.Interval.Empirical ? Quantile(parameters.Interval, uniform) : parameters.Interval.Sigma * gaussian;
    return {parameters.Amplitude.Mean + Filter(parameters.Amplitude, state.Amplitude, amplitude_input), parameters.Interval.Mean + Filter(parameters.Interval, state.Interval, interval_input)};
}

inline float Duration(CONAN_THREAD const Parameters &parameters, float amplitude) {
#ifdef __METAL_VERSION__
    return parameters.DurationScale * pow(amplitude, -parameters.DurationExponent);
#else
    return parameters.DurationScale * std::pow(amplitude, -parameters.DurationExponent);
#endif
}

inline float Pulse(float centered_time, float duration) {
    if (centered_time < -duration * 0.5f || centered_time > duration * 0.5f || duration <= 0) return 0;
#ifdef __METAL_VERSION__
    return 0.5f * (1 + cos(2 * Pi * centered_time / duration));
#else
    return 0.5f * (1 + std::cos(2 * Pi * centered_time / duration));
#endif
}

inline float Step(CONAN_THREAD const Parameters &parameters, CONAN_THREAD State &state) {
    const float step = 1 / parameters.SampleRate;
    if (!state.Events) state.Phase -= 2 * Pi * parameters.ModulationHz * parameters.MaximumDuration * 0.5f;
    if (state.UntilEventSamples <= 0) {
        Event event = NextEvent(parameters, state);
        if (event.Amplitude < parameters.MinimumAmplitude) {
            event.Amplitude = parameters.MinimumAmplitude;
            ++state.AmplitudeClamps;
        }
        if (event.Interval < parameters.MinimumInterval) {
            event.Interval = parameters.MinimumInterval;
            ++state.IntervalClamps;
        }
        float duration = Duration(parameters, event.Amplitude);
        if (duration > parameters.MaximumDuration) {
            duration = parameters.MaximumDuration;
            ++state.DurationClamps;
        }
        // Fixed lookahead preserves the paper's peak-to-peak timing with symmetric pulses.
        if (state.PulseCount < PulseCapacity) {
            const unsigned index = state.PulseCount++;
            state.PulseAgeSamples[index] = -parameters.MaximumDuration * parameters.SampleRate * 0.5f - state.UntilEventSamples;
            state.PulseDurationSamples[index] = duration * parameters.SampleRate;
            state.PulseAmplitude[index] = event.Amplitude;
        } else ++state.PulseOverflows;
        state.UntilEventSamples += event.Interval * parameters.SampleRate;
        ++state.Events;
    }
    state.UntilEventSamples -= 1;
    float output = 0;
    for (unsigned index = 0; index < state.PulseCount;) {
        output += state.PulseAmplitude[index] * Pulse(state.PulseAgeSamples[index], state.PulseDurationSamples[index]);
        state.PulseAgeSamples[index] += 1;
        if (state.PulseAgeSamples[index] > state.PulseDurationSamples[index] * 0.5f) {
            --state.PulseCount;
            state.PulseAgeSamples[index] = state.PulseAgeSamples[state.PulseCount];
            state.PulseAmplitude[index] = state.PulseAmplitude[state.PulseCount];
            state.PulseDurationSamples[index] = state.PulseDurationSamples[state.PulseCount];
        } else ++index;
    }
#ifdef __METAL_VERSION__
    output *= 1 + parameters.ModulationDepth * sin(state.Phase);
#else
    output *= 1 + parameters.ModulationDepth * std::sin(state.Phase);
#endif
    const float increment = 2 * Pi * parameters.ModulationHz * step - state.PhaseError;
    const float next_phase = state.Phase + increment;
    state.PhaseError = (next_phase - state.Phase) - increment;
    state.Phase = next_phase;
    if (state.Phase >= 2 * Pi) state.Phase -= 2 * Pi;
    return output;
}

#ifndef __METAL_VERSION__
Parameters MakeParameters(Controls controls, float sample_rate = 48000);
State MakeState(uint64_t seed, uint64_t stream = 1);
void Validate(const Parameters &parameters);
void Render(const Parameters &parameters, State &state, std::span<float> output);

struct Impact {
    double Time{};
    double Amplitude{};
    double Duration{}; // Zero when the force peak has no measurable isolated FWHM.
};

struct Fit {
    Process Model{};
    double ResidualVariance{};
    double ResidualLagOne{};
};

struct DurationLaw {
    double Scale{};
    double Exponent{};
    double RSquared{};
};

// Fit exponent on isolated contacts, then hold it fixed while fitting rolling contacts.
DurationLaw FitDurationLaw(std::span<const Impact> impacts, double fixed_exponent = -1);
Fit FitProcess(std::span<const double> series, bool empirical = false);
std::vector<Impact> ExtractImpacts(std::span<const float> force, double sample_rate, double threshold = 0);
Parameters Calibrate(std::span<const Impact> impacts, float sample_rate, bool empirical = false, float duration_exponent = 0.29f);
#endif

} // namespace surface_audio::conan

#undef CONAN_THREAD

#undef CONAN_CONSTANT
