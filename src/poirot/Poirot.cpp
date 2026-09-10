#include "Poirot.h"
#include "SignalGpuTypes.h"
#include "core/FiniteDifference.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numbers>
#include <numeric>

namespace surface_audio::poirot {
constexpr double Pi = std::numbers::pi;
static void Validate(const StringParameters &p) {
    const std::array values{p.SampleRate, p.Length, p.WaveSpeed, p.Stiffness, p.Loss0, p.Loss1, p.Density, p.Area, p.ExcitationPosition, p.ExcitationForce, p.ExcitationDuration, p.ObstaclePosition, p.ObstacleHeight, p.ContactStiffness, p.ContactExponent, p.Activation, p.ReadoutPosition};
    for (const double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite string parameter");
    if (p.SampleRate <= 0 || p.Length <= 0 || p.WaveSpeed <= 0 || p.Stiffness < 0 || p.Loss0 < 0 || p.Loss1 < 0 || p.Density <= 0 || p.Area <= 0 || p.ExcitationDuration <= 0 || p.ContactStiffness < 0 || p.ContactExponent <= 1 || p.Activation < 0 || p.ExcitationPosition <= 0 || p.ExcitationPosition >= 1 || p.ObstaclePosition <= 0 || p.ObstaclePosition >= 1 || p.ReadoutPosition <= 0 || p.ReadoutPosition >= 1) throw std::invalid_argument("Invalid string parameter");
}
std::vector<Mode> StringModes(const StringParameters &p) {
    Validate(p);
    std::vector<Mode> modes;
    for (unsigned i = 1; i < 4096; ++i) {
        const double wavenumber = Pi * i / p.Length;
        const double frequency = std::sqrt(p.WaveSpeed * p.WaveSpeed * wavenumber * wavenumber + p.Stiffness * p.Stiffness * std::pow(wavenumber, 4)) / (2 * Pi);
        if (frequency >= p.SampleRate / 2) break;
        modes.push_back({float(frequency), float(p.Loss0 + p.Loss1 * wavenumber * wavenumber), 0, 0});
    }
    if (modes.empty() || modes.size() >= 4095) throw std::invalid_argument("Unsupported string mode count");
    return modes;
}
SignalState MakeSignal(const SignalParameters &p, std::span<const Mode> modes) {
    const std::array values{p.SampleRate, p.Activation, p.Lambda, p.Height, p.Position, p.SplitThreshold, p.SplitSlope, p.PowerScale, p.ReturnGain};
    for (const float value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite signal parameter");
    if (p.SampleRate <= 0 || p.Activation < 0 || p.Lambda < 0 || p.Lambda > 1 || p.Height < 0 || p.Position <= 0 || p.Position >= 1 || p.SplitThreshold < 0 || p.SplitSlope < 0 || p.PowerScale < 0 || p.ReturnGain < 0 || p.ReturnGain > 1 || modes.empty() || modes.size() > 1024) throw std::invalid_argument("Invalid signal parameter");
    SignalState s{.Parameters = p, .Modes = {modes.begin(), modes.end()}};
    for (auto *values : {&s.Power, &s.UpperPhase, &s.LowerPhase, &s.Threshold, &s.Weight, &s.Shape, &s.Loss, &s.Transfer}) values->resize(modes.size());
    for (size_t i = 0; i < modes.size(); ++i) {
        const auto &mode = modes[i];
        if (!std::isfinite(mode.Frequency) || !std::isfinite(mode.Damping) || !std::isfinite(mode.Amplitude) || !std::isfinite(mode.Phase) || mode.Frequency <= 0 || mode.Frequency >= p.SampleRate / 2 || mode.Damping < 0 || mode.Amplitude < 0) throw std::invalid_argument("Invalid signal mode");
        const float shape = std::abs(std::sin(Pi * (i + 1) * p.Position));
        const bool node = shape < 1e-6f;
        s.Shape[i] = node ? 0 : shape;
        s.Weight[i] = node ? 0 : shape;
        s.Threshold[i] = node ? INFINITY : .5f * std::pow(p.Height / shape, 2);
        s.Power[i] = .5f * mode.Amplitude * mode.Amplitude;
        s.UpperPhase[i] = mode.Phase;
        s.LowerPhase[i] = mode.Phase;
        s.Loss[i] = std::exp(-2 * mode.Damping / p.SampleRate);
    }
    const float total = std::accumulate(s.Weight.begin(), s.Weight.end(), 0.f);
    if (total > 0)
        for (auto &weight : s.Weight) weight /= total;
    return s;
}
float TransferPower(const SignalParameters &p, std::span<const float> power, std::span<const float> threshold, std::span<const float> weight, std::span<float> transfer) {
    if (power.size() != threshold.size() || power.size() != weight.size() || power.size() != transfer.size()) throw std::invalid_argument("Signal transfer dimensions");
    float total = 0;
    for (size_t i = 0; i < power.size(); ++i) {
        transfer[i] = p.Lambda * std::max(power[i] - threshold[i], 0.f);
        total += transfer[i];
    }
    for (size_t i = 0; i < power.size(); ++i) transfer[i] = p.ReturnGain * weight[i] * total - transfer[i];
    return total;
}
void RenderSignal(SignalState &s, std::span<float> output) {
    const auto &p = s.Parameters;
    const float step = 2 * Pi / p.SampleRate, split_hz = s.Modes[0].Frequency / 3;
    const double activation = std::floor(double(p.Activation) * p.SampleRate);
    for (auto &sample : output) {
        const bool active = double(s.Frame) >= activation;
        const bool reset = p.ResetPhaseAtActivation && s.Frame == uint64_t(activation);
        const float total = active ? TransferPower(p, s.Power, s.Threshold, s.Weight, s.Transfer) : 0;
        const float split = active ? -std::expm1(-p.SplitSlope * std::max(total * p.PowerScale - p.SplitThreshold, 0.f)) : 0;
        sample = 0;
        for (size_t i = 0; i < s.Modes.size(); ++i) {
            const float c = split * (p.ShapeWeightedSplit ? s.Shape[i] : s.Weight[i]);
            if (reset) {
                s.UpperPhase[i] = step * (s.Modes[i].Frequency + c * split_hz);
                s.LowerPhase[i] = step * (s.Modes[i].Frequency - split_hz);
            }
            const float b = std::sqrt(2 * s.Power[i] / (1 + c * c));
            sample += b * (std::sin(s.UpperPhase[i]) + c * std::sin(s.LowerPhase[i]));
            s.UpperPhase[i] = std::remainder(s.UpperPhase[i] + step * (s.Modes[i].Frequency + c * split_hz), float(2 * Pi));
            s.LowerPhase[i] = std::remainder(s.LowerPhase[i] + step * (s.Modes[i].Frequency - split_hz), float(2 * Pi));
            s.Power[i] = std::max(0.f, s.Power[i] + (active ? s.Transfer[i] : 0)) * s.Loss[i];
        }
        ++s.Frame;
    }
}
GpuSignal CreateGpuSignal(Gpu &gpu, std::span<const SignalState> states, uint32_t frames) {
    if (states.empty() || !frames) throw std::invalid_argument("Empty GPU signal");
    if (states[0].Frame > UINT32_MAX) throw std::invalid_argument("GPU initial frame exceeds 32 bits");
    const uint32_t count = states[0].Modes.size();
    if (!count || count > 1024 || uint64_t(states.size()) * count > UINT32_MAX / 3) throw std::invalid_argument("GPU signal mode count");
    const auto kernel = CreateKernel(gpu, "PoirotSignalSynthesize");
    const auto threads = std::bit_ceil(std::max(count, 32u));
    if (threads > kernel.MaxThreads || uint64_t(states.size()) * frames > UINT32_MAX) throw std::invalid_argument("GPU signal dimensions");
    std::vector<SignalGpuControls> parameters;
    std::vector<float> state;
    std::vector<SignalGpuMode> modes;
    parameters.reserve(states.size());
    state.reserve(states.size() * count * 3);
    modes.reserve(states.size() * count);
    for (const auto &s : states) {
        if (s.Modes.size() != count || s.Power.size() != count || s.UpperPhase.size() != count || s.LowerPhase.size() != count || s.Threshold.size() != count || s.Weight.size() != count || s.Shape.size() != count || s.Loss.size() != count || s.Frame != states[0].Frame) throw std::invalid_argument("GPU signal banks differ");
        const auto &p = s.Parameters;
        const double activation = std::floor(double(p.Activation) * p.SampleRate);
        if (!std::isfinite(activation) || activation < 0 || activation > UINT32_MAX) throw std::invalid_argument("GPU activation frame exceeds 32 bits");
        parameters.push_back({p.SampleRate, uint32_t(activation), p.Lambda, p.SplitThreshold, p.SplitSlope, p.PowerScale, p.ReturnGain, uint32_t(p.ShapeWeightedSplit), uint32_t(p.ResetPhaseAtActivation)});
        for (uint32_t i = 0; i < count; ++i) {
            modes.push_back({s.Modes[i].Frequency, s.Loss[i], s.Threshold[i], s.Weight[i], s.Shape[i]});
            state.insert(state.end(), {s.Power[i], s.UpperPhase[i], s.LowerPhase[i]});
        }
    }
    return {Upload<SignalGpuControls>(gpu, parameters), Upload<SignalGpuMode>(gpu, modes), Upload<float>(gpu, state), CreateBuffer(gpu, states.size() * frames * sizeof(float)), kernel, uint32_t(states.size()), count, frames, threads, states[0].Frame};
}
void EncodeSignal(Gpu &gpu, GpuSignal &s, uint32_t frames) {
    if (!frames || frames > s.Capacity || s.Frame > UINT32_MAX || frames > UINT32_MAX - s.Frame) throw std::invalid_argument("GPU signal frame count");
    const SignalGpuParameters header{s.Voices, s.ModeCount, frames, uint32_t(s.Frame)};
    const auto dispatch_parameters = BatchUpload(gpu, header);
    const std::array bindings{GpuBinding{s.Parameters, 0}, GpuBinding{s.Modes, 1}, GpuBinding{s.State, 2}, GpuBinding{s.Output, 3}, GpuBinding{dispatch_parameters, 4}};
    DispatchGroupsGpu(gpu, s.Kernel, bindings, {s.Voices}, {s.Threads});
    s.Frame += frames;
}
double ContactPotential(double penetration, double stiffness, double exponent) { return stiffness / (exponent + 1) * std::pow(std::max(penetration, 0.), exponent + 1); }
double ContactGradient(double a, double b, double stiffness, double exponent) {
    if (a <= 0 && b <= 0) return 0;
    if (std::abs(a - b) < 1e-7 * std::max({std::abs(a), std::abs(b), 1e-12})) return stiffness * std::pow(std::max(.5 * (a + b), 0.), exponent);
    return (ContactPotential(a, stiffness, exponent) - ContactPotential(b, stiffness, exponent)) / (a - b);
}
StringState MakeString(const StringParameters &p) {
    Validate(p);
    const double hmin = StiffStringMinimumSpacing(p.WaveSpeed, p.Stiffness * p.Stiffness, p.Loss1, 1 / p.SampleRate);
    const double segments = std::floor(p.Length / hmin);
    if (segments < 4 || segments > 100000) throw std::invalid_argument("Unsupported string grid");
    StringState s{.Parameters = p, .Segments = uint32_t(segments), .Spacing = p.Length / segments};
    s.Current.resize(s.Segments + 1);
    s.Previous = s.Current;
    s.Next = s.Current;
    return s;
}
void RenderString(StringState &s, std::span<float> output, std::span<float> contact) {
    if (!contact.empty() && contact.size() != output.size()) throw std::invalid_argument("String contact output size");
    const auto &p = s.Parameters;
    const double k = 1 / p.SampleRate, h = s.Spacing, denominator = 1 + p.Loss0 * k;
    const double a = (p.WaveSpeed * p.WaveSpeed * k * k + 2 * p.Loss1 * k) / (h * h), b = p.Stiffness * p.Stiffness * k * k / std::pow(h, 4), c = 2 * p.Loss1 * k / (h * h), force_scale = k * k / (p.Density * p.Area * h * denominator);
    const auto coordinate = [&](double x) { const double at = x * s.Segments; return std::pair{uint32_t(at), at - std::floor(at)}; };
    const auto [oi, of] = coordinate(p.ObstaclePosition);
    const auto [ei, ef] = coordinate(p.ExcitationPosition);
    const auto [ri, rf] = coordinate(p.ReadoutPosition);
    const auto interpolate = [](const auto &v, uint32_t i, double f) { return (1 - f) * v[i] + f * v[i + 1]; };
    const auto add = [&](uint32_t i, double f, double value) { if (i > 0) s.Next[i] += (1 - f) * value; if (i + 1 < s.Segments) s.Next[i + 1] += f * value; };
    const double obstacle_norm = (oi > 0 ? (1 - of) * (1 - of) : 0) + (oi + 1 < s.Segments ? of * of : 0);
    for (size_t frame = 0; frame < output.size(); ++frame, ++s.Frame) {
        for (uint32_t i = 1; i < s.Segments; ++i) {
            const double left2 = i > 1 ? s.Current[i - 2] : -s.Current[1];
            const double right2 = i + 2 <= s.Segments ? s.Current[i + 2] : -s.Current[s.Segments - 1];
            const double lap = SecondDifference(s.Current[i - 1], s.Current[i], s.Current[i + 1]);
            const double lap_prev = SecondDifference(s.Previous[i - 1], s.Previous[i], s.Previous[i + 1]);
            s.Next[i] = (2 * s.Current[i] + a * lap - b * (left2 - 4 * s.Current[i - 1] + 6 * s.Current[i] - 4 * s.Current[i + 1] + right2) + (p.Loss0 * k - 1) * s.Previous[i] - c * lap_prev) / denominator;
        }
        const double t = s.Frame * k;
        if (t < p.ExcitationDuration) add(ei, ef, -force_scale * .5 * p.ExcitationForce * (1 - std::cos(Pi * t / p.ExcitationDuration)));
        s.LastContactForce = 0;
        if (t >= p.Activation && p.ContactStiffness > 0) {
            const double old = interpolate(s.Previous, oi, of) - p.ObstacleHeight, free = interpolate(s.Next, oi, of) - p.ObstacleHeight, scale = force_scale * obstacle_norm;
            if (old > 0 || free > 0) {
                double lower = free - scale * p.ContactStiffness * std::pow(std::max({free, old, 0.}), p.ContactExponent), upper = free;
                for (unsigned j = 0; j < 64; ++j) {
                    const double mid = .5 * (lower + upper), residual = mid + scale * ContactGradient(mid, old, p.ContactStiffness, p.ContactExponent) - free;
                    if (residual > 0) upper = mid;
                    else lower = mid;
                }
                const double root = .5 * (lower + upper);
                s.LastContactForce = ContactGradient(root, old, p.ContactStiffness, p.ContactExponent);
                add(oi, of, -force_scale * s.LastContactForce);
                s.MaximumPenetration = std::max(s.MaximumPenetration, root);
            }
        }
        output[frame] = float(interpolate(s.Next, ri, rf));
        if (!contact.empty()) contact[frame] = float(s.LastContactForce);
        s.Previous.swap(s.Current);
        s.Current.swap(s.Next);
    }
}
} // namespace surface_audio::poirot
