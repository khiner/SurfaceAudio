#include "SpatialConvolution.h"
#include "AgarwalGpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace surface_audio::agarwal {
std::vector<float> ConvolveSpatialGpu(Gpu &gpu, std::span<const float> excitation, std::span<const float> morph, const ImpulseResponses &responses, uint32_t block_frames, float gain) {
    if (excitation.empty() || excitation.size() != morph.size() || !block_frames || !responses.TapCount || excitation.size() > UINT32_MAX - uint64_t(responses.TapCount) + 1 || !std::isfinite(gain)) throw std::invalid_argument("Invalid spatial convolution dimensions or gain");
    for (float value : excitation)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite spatial convolution excitation");
    for (float value : morph)
        if (!std::isfinite(value) || value < 0 || value > 1) throw std::invalid_argument("Invalid spatial convolution morph");
    const size_t count = excitation.size() + responses.TapCount - 1;
    const uint32_t capacity = uint32_t(std::min<size_t>(block_frames, count));
    auto prepared = PrepareGpuImpulseData(responses, capacity);
    struct FirParameters {
        uint32_t TapCount, FrameCount;
    };
    const auto ir_parameters = Upload(gpu, prepared.Parameters), fir_parameters = Upload(gpu, FirParameters{responses.TapCount, capacity});
    const auto frequencies = Upload<float>(gpu, prepared.Frequencies), amplitudes = Upload<float>(gpu, prepared.Amplitudes);
    const auto locations = CreateBuffer(gpu, size_t(capacity) * sizeof(float));
    const auto coefficients = CreateBuffer(gpu, size_t(capacity) * responses.TapCount * sizeof(float));
    const auto input = CreateBuffer(gpu, (size_t(responses.TapCount) - 1 + capacity) * sizeof(float)), output = CreateBuffer(gpu, size_t(capacity) * sizeof(float));
    const auto build = CreateKernel(gpu, "AgarwalBuildImpulseResponses"), convolve = CreateKernel(gpu, "FirConvolve");
    const std::array build_bindings{GpuBinding{ir_parameters, 0}, GpuBinding{frequencies, 1}, GpuBinding{amplitudes, 2}, GpuBinding{locations, 3}, GpuBinding{coefficients, 4}};
    const std::array convolve_bindings{GpuBinding{fir_parameters, 0}, GpuBinding{coefficients, 1}, GpuBinding{input, 2}, GpuBinding{output, 3}};
    const auto input_samples = BufferSpan<float>(input), locations_samples = BufferSpan<float>(locations), output_samples = BufferSpan<float>(output);
    std::vector<float> result(count);
    for (size_t offset = 0; offset < count;) {
        const uint32_t frames = uint32_t(std::min<size_t>(capacity, count - offset));
        const int64_t input_begin = int64_t(offset) - (responses.TapCount - 1);
        const size_t first = size_t(std::max<int64_t>(0, input_begin)), last = std::min(excitation.size(), offset + frames);
        std::ranges::fill(input_samples, 0);
        if (first < last) std::copy(excitation.begin() + first, excitation.begin() + last, input_samples.begin() + (int64_t(first) - input_begin));
        for (uint32_t frame = 0; frame < frames; ++frame) locations_samples[frame] = morph[std::min(offset + frame, morph.size() - 1)];
        BufferSpan<GpuImpulseParameters>(ir_parameters)[0].FrameCount = frames;
        BufferSpan<FirParameters>(fir_parameters)[0].FrameCount = frames;
        BeginGpu(gpu);
        DispatchGpu(gpu, build, build_bindings, {frames * responses.TapCount});
        DispatchGroupsGpu(gpu, convolve, convolve_bindings, {frames}, {128});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const float value = output_samples[frame] * gain;
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite spatial convolution output");
            result[offset + frame] = value;
        }
        offset += frames;
    }
    return result;
}
} // namespace surface_audio::agarwal
