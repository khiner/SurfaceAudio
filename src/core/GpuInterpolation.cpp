#include "GpuInterpolation.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace surface_audio {
std::vector<float> InterpolateSignalsGpu(Gpu &gpu, std::span<const float> signals, std::span<const float> nodes, std::span<const float> positions) {
    if (nodes.empty() || positions.empty() || nodes.size() > UINT32_MAX || positions.size() > UINT32_MAX / nodes.size() || signals.size() != nodes.size() * positions.size()) throw std::invalid_argument("Invalid signal interpolation dimensions");
    for (const auto values : {signals, nodes, positions})
        if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); })) throw std::invalid_argument("Nonfinite signal interpolation input");
    for (size_t i = 1; i < nodes.size(); ++i)
        if (nodes[i] <= nodes[i - 1] || !std::isfinite(nodes[i] - nodes[i - 1])) throw std::invalid_argument("Signal interpolation nodes must increase with finite spacing");
    struct Parameters {
        uint32_t Nodes, Frames;
    };
    const auto block = Upload(gpu, Parameters{uint32_t(nodes.size()), uint32_t(positions.size())});
    const auto values = Upload<float>(gpu, signals), centers = Upload<float>(gpu, nodes), locations = Upload<float>(gpu, positions);
    const auto output = CreateBuffer(gpu, positions.size_bytes());
    const auto kernel = CreateKernel(gpu, "InterpolateSignals");
    const std::array bindings{GpuBinding{block, 0}, GpuBinding{values, 1}, GpuBinding{centers, 2}, GpuBinding{locations, 3}, GpuBinding{output, 4}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {uint32_t(positions.size())});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output);
    return {samples.begin(), samples.end()};
}
}
