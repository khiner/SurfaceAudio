#include "core/GpuFft.h"
#include <array>
#include <bit>
#include <cmath>
#include <numbers>
#include <vector>

namespace surface_audio {
namespace {
struct FftBlock {
    uint32_t Size, Count, LogSize, Width;
    float Sign, Scale;
};
}
FftGpu CreateFftGpu(Gpu &gpu, uint32_t size, uint32_t count) {
    if (!std::has_single_bit(size) || size > (1u << 24) || !count || uint64_t(size) * count > UINT32_MAX) throw std::invalid_argument("Invalid GPU FFT dimensions");
    std::vector<std::array<float, 2>> twiddles(std::max(1u, size / 2));
    for (uint32_t i = 0; i < twiddles.size(); ++i) {
        const double angle = 2 * std::numbers::pi * i / size;
        twiddles[i] = {float(std::cos(angle)), float(std::sin(angle))};
    }
    const size_t bytes = size_t(size) * count * sizeof(std::array<float, 2>);
    return {.Size = size, .Count = count, .Output = CreateBuffer(gpu, bytes), .Scratch = CreateBuffer(gpu, bytes), .Twiddles = Upload<std::array<float, 2>>(gpu, twiddles), .Local = CreateKernel(gpu, "FftLocal"), .Stage = CreateKernel(gpu, "FftStage")};
}
void EncodeFftGpu(Gpu &gpu, const FftGpu &plan, GpuBuffer input, bool inverse) {
    if (input.Size < plan.Output.Size || input.Data == plan.Output.Data || input.Data == plan.Scratch.Data) throw std::invalid_argument("GPU FFT input overlaps transform storage or is too short");
    const uint32_t bits = std::countr_zero(plan.Size), stages = bits > 12 ? bits - 12 : 0;
    const float sign = inverse ? 1.f : -1.f, scale = inverse ? 1.f / plan.Size : 1.f;
    auto source = stages % 2 ? plan.Scratch : plan.Output;
    const auto parameters = BatchUpload(gpu, FftBlock{plan.Size, plan.Count, bits, 0, sign, stages ? 1.f : scale});
    const std::array local{GpuBinding{parameters, 0}, GpuBinding{input, 1}, GpuBinding{source, 2}, GpuBinding{plan.Twiddles, 3}};
    DispatchGroupsGpu(gpu, plan.Local, local, {std::max(1u, plan.Size / 4096) * plan.Count, 1, 1}, {256, 1, 1});
    for (uint32_t width = 8192; width <= plan.Size; width *= 2) {
        const auto target = source.Data == plan.Output.Data ? plan.Scratch : plan.Output;
        const auto constants = BatchUpload(gpu, FftBlock{plan.Size, plan.Count, bits, width, sign, width == plan.Size ? scale : 1.f});
        const std::array bindings{GpuBinding{constants, 0}, GpuBinding{source, 1}, GpuBinding{target, 2}, GpuBinding{plan.Twiddles, 3}};
        DispatchGpu(gpu, plan.Stage, bindings, {plan.Size / 2, plan.Count, 1}, {256, 1, 1});
        source = target;
    }
}
}
