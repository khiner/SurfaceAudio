#include "GpuSparseConvolution.h"
#include "GpuFft.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace surface_audio {
namespace {
struct SparseBlock {
    uint32_t Size, Frames, Observations;
    float Step, Penalty, Momentum;
};
float Peak(std::span<const float> signal) {
    for (float value : signal)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite sparse convolution input");
    float peak = 0;
    vDSP_maxmgv(signal.data(), 1, &peak, signal.size());
    return peak;
}
}
SparseConvolutionFit FitSparseConvolutionGpu(Gpu &gpu, std::span<const float> signal, std::span<const float> impulse, double relative_penalty, uint32_t iterations) {
    return FitSparseConvolutionGpu(gpu, std::array{ConvolutionObservation{signal, impulse}}, relative_penalty, iterations);
}
SparseConvolutionFit FitSparseConvolutionGpu(Gpu &gpu, std::span<const ConvolutionObservation> observations, double relative_penalty, uint32_t iterations) {
    if (observations.empty() || observations.size() > 16 || !(relative_penalty >= 0 && relative_penalty < 1) || !iterations)
        throw std::invalid_argument("Invalid sparse convolution observations");
    const size_t frames = observations.front().Signal.size();
    double signal_peak = 0, impulse_peak = 0;
    size_t taps = 0;
    for (const auto &[signal, impulse, weight] : observations) {
        if (signal.empty() || signal.size() != frames || frames > (1u << 23) || impulse.empty() || impulse.size() > (1u << 23) ||
            !(weight > 0) || !std::isfinite(weight)) throw std::invalid_argument("Invalid sparse convolution dimensions or weight");
        signal_peak = std::max(signal_peak, Peak(signal) * std::sqrt(weight));
        impulse_peak = std::max(impulse_peak, Peak(impulse) * std::sqrt(weight));
        taps = std::max(taps, impulse.size());
    }
    if (!(impulse_peak > 0)) throw std::invalid_argument("Zero sparse convolution impulse");
    if (signal_peak == 0) return {std::vector<float>(frames), 0};
    const uint32_t size = std::bit_ceil(uint32_t(frames + taps - 1)), count = uint32_t(observations.size());
    std::vector<std::array<float, 2>> packed(size_t(2) * count * size);
    for (size_t channel = 0; channel < observations.size(); ++channel) {
        const auto &[signal, impulse, weight] = observations[channel];
        const double scale = std::sqrt(weight);
        for (size_t n = 0; n < signal.size(); ++n) packed[2 * channel * size + n][0] = float(signal[n] * scale / signal_peak);
        for (size_t n = 0; n < impulse.size(); ++n) packed[(2 * channel + 1) * size + n][0] = float(impulse[n] * scale / impulse_peak);
    }
    const auto input = Upload<std::array<float, 2>>(gpu, packed);
    const auto initial = CreateFftGpu(gpu, size, 2 * count);
    const auto transform = CreateFftGpu(gpu, size);
    const auto model = CreateBuffer(gpu, size_t(size) * sizeof(std::array<float, 4>));
    const auto spectrum = CreateBuffer(gpu, size_t(size) * sizeof(std::array<float, 2>));
    const auto extrapolated = CreateBuffer(gpu, spectrum.Size), coefficients = CreateBuffer(gpu, size_t(size) * sizeof(float));
    const auto violation = CreateBuffer(gpu, coefficients.Size);
    const uint32_t groups = (size + 255) / 256;
    const auto power_max = CreateBuffer(gpu, groups * sizeof(float)), rhs_max = CreateBuffer(gpu, groups * sizeof(float));
    const auto prepare = CreateKernel(gpu, "SparseConvolutionPrepare"), maximum = CreateKernel(gpu, "SparseConvolutionMaximum");
    const auto gradient = CreateKernel(gpu, "SparseConvolutionGradient"), step = CreateKernel(gpu, "SparseConvolutionStep");
    const auto kkt = CreateKernel(gpu, "SparseConvolutionKkt");
    std::ranges::fill(BufferSpan<float>(extrapolated), 0);
    std::ranges::fill(BufferSpan<float>(coefficients), 0);
    const auto reduce = [&](GpuBuffer values, uint32_t stride, GpuBuffer output, uint32_t count) {
        const auto constants = BatchUpload(gpu, std::array<uint32_t, 2>{count, stride});
        const std::array bindings{GpuBinding{constants, 0}, GpuBinding{values, 1}, GpuBinding{output, 2}};
        DispatchGroupsGpu(gpu, maximum, bindings, {groups, 1, 1}, {256, 1, 1});
    };
    BeginGpu(gpu);
    EncodeFftGpu(gpu, initial, input);
    const auto constants = BatchUpload(gpu, SparseBlock{size, uint32_t(frames), count, 0, 0, 0});
    const std::array preparation{GpuBinding{constants, 0}, GpuBinding{initial.Output, 1}, GpuBinding{model, 2}, GpuBinding{spectrum, 3}};
    DispatchGpu(gpu, prepare, preparation, {size, 1, 1});
    reduce(model, 4, power_max, size);
    EncodeFftGpu(gpu, transform, spectrum, true);
    reduce(transform.Output, 2, rhs_max, frames);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const float lipschitz = std::ranges::max(BufferSpan<float>(power_max)), rhs_peak = std::ranges::max(BufferSpan<float>(rhs_max));
    if (!(rhs_peak > 0)) return {std::vector<float>(frames), 0};
    const float penalty = float(relative_penalty * rhs_peak);
    double acceleration = 1;
    for (uint32_t iteration = 0;; ++iteration) {
        const double next = .5 * (1 + std::sqrt(1 + 4 * acceleration * acceleration));
        const float momentum = iteration < iterations - 1 ? float((acceleration - 1) / next) : 0;
        BeginGpu(gpu);
        EncodeFftGpu(gpu, transform, extrapolated);
        const auto block = BatchUpload(gpu, SparseBlock{size, uint32_t(frames), count, 1 / lipschitz, penalty, momentum});
        const std::array derivatives{GpuBinding{block, 0}, GpuBinding{model, 1}, GpuBinding{transform.Output, 2}, GpuBinding{spectrum, 3}};
        DispatchGpu(gpu, gradient, derivatives, {size, 1, 1});
        EncodeFftGpu(gpu, transform, spectrum, true);
        const std::array update{GpuBinding{block, 0}, GpuBinding{transform.Output, 1}, GpuBinding{coefficients, 2}, GpuBinding{iteration < iterations ? extrapolated : violation, 3}};
        DispatchGpu(gpu, iteration < iterations ? step : kkt, update, {size, 1, 1});
        if (iteration == iterations) reduce(violation, 1, rhs_max, frames);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        if (iteration == iterations) break;
        acceleration = next;
    }
    const double error = std::ranges::max(BufferSpan<float>(rhs_max)) / rhs_peak, scale = double(signal_peak) / impulse_peak;
    std::vector<float> result(frames);
    const auto values = BufferSpan<float>(coefficients);
    for (uint32_t n = 0; n < frames; ++n) {
        result[n] = float(values[n] * scale);
        if (!std::isfinite(result[n])) throw std::runtime_error("Sparse convolution coefficients exceed FP32 range");
    }
    return {std::move(result), error};
}
}
