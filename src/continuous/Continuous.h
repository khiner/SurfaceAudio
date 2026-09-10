#pragma once

#include "conan/Conan.h"

#ifdef __METAL_VERSION__
#define CONTINUOUS_THREAD thread
#define CONTINUOUS_CONSTANT constant
#else
#include "core/Gpu.h"
#include <array>
#define CONTINUOUS_THREAD
#define CONTINUOUS_CONSTANT
#endif

namespace surface_audio::continuous {
CONTINUOUS_CONSTANT constexpr unsigned RingSize = 1024;

struct Parameters {
    conan::Process Amplitude, Interval;
    float SampleRate{44100}, DurationScale{}, DurationExponent{}, MaximumDuration{.008f};
    float ModulationHz{}, ModulationDepth{}, LowpassPole{}, Gain{1};
};
struct State {
    RandomState Random{};
    conan::FilterState Amplitude{}, Interval{};
    float UntilEventSamples{}, Phase{}, PhaseError{}, Lowpass{};
    float Pending[RingSize]{};
    unsigned Cursor{}, Events{}, IntervalClamps{}, DurationClamps{};
};

inline conan::Event NextEvent(CONTINUOUS_THREAD const Parameters &p, CONTINUOUS_THREAD State &s) {
    const float normal = Normal(s.Random), uniform = conan::NormalCdf(normal);
    const float a = p.Amplitude.Empirical ? conan::Quantile(p.Amplitude, uniform) : p.Amplitude.Sigma * normal;
    const float dt = p.Interval.Empirical ? conan::Quantile(p.Interval, uniform) : p.Interval.Sigma * normal;
    return {p.Amplitude.Mean + conan::Filter(p.Amplitude, s.Amplitude, a), p.Interval.Mean + conan::Filter(p.Interval, s.Interval, dt)};
}

inline float Step(CONTINUOUS_THREAD const Parameters &p, CONTINUOUS_THREAD State &s) {
#ifdef __METAL_VERSION__
    using namespace metal;
#else
    using std::abs;
    using std::ceil;
    using std::floor;
    using std::pow;
    using std::sin;
#endif
    if (s.UntilEventSamples <= 0) {
        const auto event = NextEvent(p, s);
        const float magnitude = abs(event.Amplitude) > .0001f ? abs(event.Amplitude) : .0001f;
        float duration = p.DurationScale * pow(magnitude, -p.DurationExponent);
        if (duration > p.MaximumDuration) {
            duration = p.MaximumDuration;
            ++s.DurationClamps;
        }
        const float half_duration = .5f * duration * p.SampleRate;
        // Fixed lookahead preserves symmetric pulse centers and the one-sample noise anchor.
        const float center = ceil(.5f * p.MaximumDuration * p.SampleRate) + s.UntilEventSamples;
        const int first = int(ceil(center - half_duration)), last = int(floor(center + half_duration));
        for (int lag = first < 0 ? 0 : first; lag <= last; ++lag)
            s.Pending[(s.Cursor + unsigned(lag)) & (RingSize - 1)] += event.Amplitude * conan::Pulse(float(lag) - center, 2 * half_duration);
        const float interval_samples = event.Interval * p.SampleRate;
        if (interval_samples < 1) ++s.IntervalClamps;
        s.UntilEventSamples += interval_samples < 1 ? 1 : interval_samples;
        ++s.Events;
    }
    s.UntilEventSamples -= 1;
    const float force = s.Pending[s.Cursor] * (1 + p.ModulationDepth * sin(s.Phase));
    s.Pending[s.Cursor] = 0;
    s.Cursor = (s.Cursor + 1) & (RingSize - 1);
    s.Lowpass = (1 - p.LowpassPole) * force + p.LowpassPole * s.Lowpass;
    const float increment = 2 * conan::Pi * p.ModulationHz / p.SampleRate - s.PhaseError;
    const float phase = s.Phase + increment;
    s.PhaseError = (phase - s.Phase) - increment;
    s.Phase = phase >= 2 * conan::Pi ? phase - 2 * conan::Pi : phase;
    return p.Gain * s.Lowpass;
}

#ifndef __METAL_VERSION__
struct Controls {
    double Angle{}, Radius{1};
    float Size{.5f}, Velocity{.5f}, Roughness{.5f}, Asymmetry{};
    float ScratchDensity{100}, FrictionSigma{1}, FrictionDurationSamples{1};
    float CutoffHz{20000}, Gain{1};
};
std::array<double, 3> ActionWeights(double angle, double radius);
Parameters MakeParameters(Controls controls, float sample_rate = 44100);
void Validate(const Parameters &);
State MakeState(uint64_t seed, uint64_t stream = 1);
void Render(const Parameters &, State &, std::span<float>);

struct GpuSource {
    uint32_t Voices{}, Frames{};
    GpuBuffer Parameters, States, Output, Block;
    GpuKernel Kernel;
};
GpuSource CreateGpuSource(Gpu &, std::span<const Parameters>, std::span<const State>, uint32_t frames);
void EncodeSource(Gpu &, const GpuSource &);
#endif
} // namespace surface_audio::continuous

#undef CONTINUOUS_THREAD
#undef CONTINUOUS_CONSTANT
