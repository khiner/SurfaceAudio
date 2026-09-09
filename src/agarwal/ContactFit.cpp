#include "ContactFit.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>
#include <vector>

namespace surface_audio::agarwal {
namespace {
struct ContactFitBlock {
    uint32_t Modes, ForceFrames, Frames, Taps, SampleRate, Groups;
};
} // namespace

std::array<float, 2> ContactFitLogDecayBounds() {
    const double minimum = std::log(double(ContactFitMinimumDecay)), maximum = std::log(double(ContactFitMaximumDecay));
    const float lower = float(minimum), upper = float(maximum);
    return {double(lower) < minimum ? std::nextafter(lower, std::numeric_limits<float>::infinity()) : lower, double(upper) > maximum ? std::nextafter(upper, -std::numeric_limits<float>::infinity()) : upper};
}

ContactFitGpu CreateContactFitGpu(Gpu &gpu, std::span<const float> force, std::span<const ContactFitMode> modes, uint32_t sample_rate, uint32_t taps) {
    const size_t frames = force.size() + size_t(taps) - 1;
    if (force.empty() || modes.empty() || modes.size() > 50 || !sample_rate || !taps || frames > (1u << 22) || !std::ranges::all_of(force, [](float value) { return std::isfinite(value); })) throw std::invalid_argument("Invalid contact fit force or dimensions");
    for (const auto &mode : modes) {
        if (!std::isfinite(mode.Frequency) || mode.Frequency <= 0 || mode.Frequency >= sample_rate / 2.f || !std::isfinite(mode.Decay) || mode.Decay < ContactFitMinimumDecay || mode.Decay > ContactFitMaximumDecay || !std::isfinite(mode.Amplitude) || mode.Amplitude <= 0 || std::log(mode.Amplitude) < ContactFitMinimumLogAmplitude || std::log(mode.Amplitude) > ContactFitMaximumLogAmplitude) throw std::invalid_argument("Invalid contact fit mode");
    }
    const uint32_t count = uint32_t(modes.size()), groups = (uint32_t(frames) + 255) / 256;
    const ContactFitBlock block{count, uint32_t(force.size()), uint32_t(frames), taps, sample_rate, groups};
    const auto decay_bounds = ContactFitLogDecayBounds();
    const auto parameters = std::views::iota(0u, 2 * count) | std::views::transform([=](uint32_t index) { return index < count ? std::log(modes[index].Amplitude) : std::clamp(std::log(modes[index - count].Decay), decay_bounds[0], decay_bounds[1]); }) | std::ranges::to<std::vector>();
    const auto frequencies = modes | std::views::transform([=](const ContactFitMode &mode) {
                                 const double angle = 2 * std::numbers::pi * mode.Frequency / sample_rate;
                                 return std::array{float(std::cos(angle)), float(std::sin(angle))};
                             }) |
        std::ranges::to<std::vector>();
    const auto mix = CreateGpuMix(gpu, {count, uint32_t(frames), 1, 0, 1});
    return {
        .Modes = count, .ForceFrames = uint32_t(force.size()), .Frames = uint32_t(frames), .Taps = taps, .SampleRate = sample_rate, .GradientGroups = groups, .Parameters = Upload<float>(gpu, parameters), .Force = Upload<float>(gpu, force), .Frequencies = Upload<std::array<float, 2>>(gpu, frequencies), .Block = Upload(gpu, block), .ModeOutput = CreateBuffer(gpu, frames * count * sizeof(float)), .ModeDecayDerivative = CreateBuffer(gpu, frames * count * sizeof(float)), .PartialGradient = CreateBuffer(gpu, size_t(groups) * count * 2 * sizeof(float)), .Gradient = CreateBuffer(gpu, count * 2 * sizeof(float)), .Output = mix.Output, .Mix = mix, .Synthesize = CreateKernel(gpu, "ContactFitSynthesize"), .Differentiate = CreateKernel(gpu, "ContactFitDifferentiate"), .Reduce = CreateKernel(gpu, "ContactFitReduce")
    };
}

void EncodeContactFit(Gpu &gpu, const ContactFitGpu &state) {
    const std::array bindings{GpuBinding{state.Block, 0}, GpuBinding{state.Parameters, 1}, GpuBinding{state.Frequencies, 2}, GpuBinding{state.Force, 3}, GpuBinding{state.ModeOutput, 4}, GpuBinding{state.ModeDecayDerivative, 5}};
    DispatchGpu(gpu, state.Synthesize, bindings, {state.Modes}, {32});
    EncodeMix(gpu, state.Mix, state.ModeOutput);
}

void EncodeContactFitGradient(Gpu &gpu, const ContactFitGpu &state, GpuBuffer sample_adjoint) {
    if (sample_adjoint.Size != size_t(state.Frames) * sizeof(float)) throw std::invalid_argument("Contact fit sample adjoint extent mismatch");
    const std::array bindings{GpuBinding{state.Block, 0}, GpuBinding{state.ModeOutput, 1}, GpuBinding{state.ModeDecayDerivative, 2}, GpuBinding{sample_adjoint, 3}, GpuBinding{state.PartialGradient, 4}};
    DispatchGroupsGpu(gpu, state.Differentiate, bindings, {state.GradientGroups, state.Modes}, {256});
    const std::array reduce{GpuBinding{state.Block, 0}, GpuBinding{state.PartialGradient, 1}, GpuBinding{state.Gradient, 2}};
    DispatchGroupsGpu(gpu, state.Reduce, reduce, {state.Modes}, {256});
}
} // namespace surface_audio::agarwal
