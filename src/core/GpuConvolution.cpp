#include "GpuConvolution.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace surface_audio {
std::vector<float> ConvolveFixedGpu(Gpu &gpu, std::span<const float> excitation, std::span<const float> response) {
    if (excitation.empty() || response.empty() || excitation.size() + response.size() > UINT32_MAX) throw std::invalid_argument("Invalid fixed convolution dimensions");
    for (auto values : {excitation, response})
        for (float value : values)
            if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite convolution input");
    struct Parameters {
        uint32_t Inputs, Taps, Offset, Frames;
    };
    const auto count = uint32_t(excitation.size() + response.size() - 1);
    const auto input = Upload<float>(gpu, excitation), taps = Upload<float>(gpu, response), output = CreateBuffer(gpu, size_t(count) * sizeof(float));
    const auto parameters = Upload(gpu, Parameters{uint32_t(excitation.size()), uint32_t(response.size()), 0, 0});
    const auto kernel = CreateKernel(gpu, "FixedFirConvolve");
    const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{taps, 1}, GpuBinding{input, 2}, GpuBinding{output, 3}};
    for (uint32_t offset = 0; offset < count;) {
        const uint32_t frames = std::min(8192u, count - offset);
        BufferSpan<Parameters>(parameters)[0] = {uint32_t(excitation.size()), uint32_t(response.size()), offset, frames};
        BeginGpu(gpu);
        DispatchGroupsGpu(gpu, kernel, bindings, {frames}, {128});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        offset += frames;
    }
    const auto samples = BufferSpan<float>(output);
    return {samples.begin(), samples.end()};
}
} // namespace surface_audio
