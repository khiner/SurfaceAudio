#include "GpuEndpointMix.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace surface_audio {
namespace {
struct EndpointMixBlock {
    uint32_t Modes, Frames, Groups;
};
} // namespace

GpuEndpointMix CreateGpuEndpointMix(Gpu &gpu, std::span<const float> basis, std::span<const float> location, std::span<const float> log_parameters) {
    if (location.empty() || location.size() > (1u << 22) || log_parameters.empty() || log_parameters.size() % 2 || log_parameters.size() > 200 || basis.size() != location.size() * (log_parameters.size() / 2)) throw std::invalid_argument("Invalid endpoint mix dimensions");
    if (!std::ranges::all_of(basis, [](float value) { return std::isfinite(value); }) || !std::ranges::all_of(location, [](float value) { return std::isfinite(value) && value >= 0 && value <= 1; }) || !std::ranges::all_of(log_parameters, [](float value) { return std::isfinite(value) && value >= EndpointMixMinimumLogAmplitude && value <= EndpointMixMaximumLogAmplitude; })) throw std::invalid_argument("Invalid endpoint mix basis, location or log amplitudes");
    const uint32_t modes = uint32_t(log_parameters.size() / 2), frames = uint32_t(location.size()), groups = (frames + 255) / 256;
    const EndpointMixBlock block{modes, frames, groups};
    const auto mix = CreateGpuMix(gpu, {modes, frames, 1, 0, 1});
    return {
        .Modes = modes, .Frames = frames, .GradientGroups = groups, .Parameters = Upload<float>(gpu, log_parameters), .Basis = Upload<float>(gpu, basis), .Location = Upload<float>(gpu, location), .Block = Upload(gpu, block), .Components = CreateBuffer(gpu, basis.size_bytes()), .PartialGradient = CreateBuffer(gpu, size_t(groups) * modes * 2 * sizeof(float)), .Gradient = CreateBuffer(gpu, log_parameters.size_bytes()), .Output = mix.Output, .Mix = mix, .Synthesize = CreateKernel(gpu, "EndpointMixSynthesize"), .Differentiate = CreateKernel(gpu, "EndpointMixDifferentiate"), .Reduce = CreateKernel(gpu, "EndpointMixReduce")
    };
}

void EncodeEndpointMix(Gpu &gpu, const GpuEndpointMix &state) {
    const std::array bindings{GpuBinding{state.Block, 0}, GpuBinding{state.Parameters, 1}, GpuBinding{state.Basis, 2}, GpuBinding{state.Location, 3}, GpuBinding{state.Components, 4}};
    DispatchGpu(gpu, state.Synthesize, bindings, {state.Frames, state.Modes}, {256});
    EncodeMix(gpu, state.Mix, state.Components);
}

void EncodeEndpointMixGradient(Gpu &gpu, const GpuEndpointMix &state, GpuBuffer sample_adjoint) {
    if (sample_adjoint.Size != size_t(state.Frames) * sizeof(float)) throw std::invalid_argument("Endpoint mix sample adjoint extent mismatch");
    const std::array bindings{GpuBinding{state.Block, 0}, GpuBinding{state.Components, 1}, GpuBinding{state.Location, 2}, GpuBinding{sample_adjoint, 3}, GpuBinding{state.PartialGradient, 4}};
    DispatchGroupsGpu(gpu, state.Differentiate, bindings, {state.GradientGroups, state.Modes}, {256});
    const std::array reduce{GpuBinding{state.Block, 0}, GpuBinding{state.PartialGradient, 1}, GpuBinding{state.Gradient, 2}};
    DispatchGroupsGpu(gpu, state.Reduce, reduce, {state.Modes}, {256});
}
} // namespace surface_audio
