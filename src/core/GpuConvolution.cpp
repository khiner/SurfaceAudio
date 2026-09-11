#include "GpuConvolution.h"
#include "GpuFft.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace surface_audio {
std::vector<float> ConvolveFftGpu(Gpu &gpu, std::span<const float> input, std::span<const float> taps) {
    if (input.empty() || taps.empty() || input.size() + taps.size() - 1 > (1u << 24)) throw std::invalid_argument("Invalid FFT convolution dimensions");
    for (const auto values : {input, taps})
        if (!std::ranges::all_of(values, [](float x) { return std::isfinite(x); })) throw std::invalid_argument("Nonfinite FFT convolution input");
    const uint32_t frames = uint32_t(input.size() + taps.size() - 1), size = std::bit_ceil(frames);
    std::vector<std::array<float, 2>> data(size_t(2) * size);
    for (size_t i = 0; i < input.size(); ++i) data[i][0] = input[i];
    for (size_t i = 0; i < taps.size(); ++i) data[size + i][0] = taps[i];
    const auto forward = CreateFftGpu(gpu, size, 2), inverse = CreateFftGpu(gpu, size);
    const auto source = Upload<std::array<float, 2>>(gpu, data), product = CreateBuffer(gpu, size_t(size) * sizeof(std::array<float, 2>));
    const auto kernel = CreateKernel(gpu, "FftMultiply");
    BeginGpu(gpu);
    EncodeFftGpu(gpu, forward, source);
    const auto parameters = BatchUpload(gpu, size);
    const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{forward.Output, 1}, GpuBinding{product, 2}};
    DispatchGpu(gpu, kernel, bindings, {size, 1, 1});
    EncodeFftGpu(gpu, inverse, product, true);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto transformed = BufferSpan<std::array<float, 2>>(inverse.Output);
    std::vector<float> output(frames);
    for (uint32_t i = 0; i < frames; ++i) output[i] = transformed[i][0];
    return output;
}
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
}
