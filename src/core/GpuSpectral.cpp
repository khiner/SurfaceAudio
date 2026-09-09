#include "GpuSpectral.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <ranges>
#include <vector>

namespace surface_audio {
namespace {
struct SpectralBlock {
    uint32_t Samples, Size, Hop, Frames, LogSize, Scale;
    float FloorSquared, Delta;
};
SpectralResolutionGpu CreateResolution(Gpu &gpu, uint32_t samples, uint32_t size, SpectralLossOptions options) {
    const uint32_t hop = size / options.HopDivisor, frames = 1 + samples / hop;
    const SpectralBlock block{samples, size, hop, frames, uint32_t(std::countr_zero(size)), uint32_t(options.Scale), options.MagnitudeFloor * options.MagnitudeFloor, options.HuberDelta};
    const auto split = [](double value) {
        const float high = float(value);
        return std::array{high, float(value - double(high))};
    };
    const auto window = std::views::iota(0u, size) | std::views::transform([=](uint32_t index) { return split(.5 - .5 * std::cos(2 * std::numbers::pi * index / size)); }) | std::ranges::to<std::vector>();
    const auto twiddles = std::views::iota(0u, size / 2) | std::views::transform([=](uint32_t index) {
                              const double angle = -2 * std::numbers::pi * index / size;
                              const auto real = split(std::cos(angle)), imaginary = split(std::sin(angle));
                              return std::array{real[0], real[1], imaginary[0], imaginary[1]};
                          }) |
        std::ranges::to<std::vector>();
    const size_t points = size_t(frames) * size;
    return {
        .Size = size, .Hop = hop, .Frames = frames, .Parameters = Upload(gpu, block), .Target = CreateBuffer(gpu, points * 2 * sizeof(float)), .Spectrum = CreateBuffer(gpu, points * 2 * sizeof(float)), .Adjoint = CreateBuffer(gpu, points * sizeof(float)), .FrameLoss = CreateBuffer(gpu, frames * sizeof(float)), .Window = Upload<std::array<float, 2>>(gpu, window), .Twiddles = Upload<std::array<float, 4>>(gpu, twiddles), .Transform = CreateBuffer(gpu, points * 4 * sizeof(float))
    };
}
void Forward(Gpu &gpu, const SpectralLossGpu &state, const SpectralResolutionGpu &resolution, GpuBuffer waveform, GpuBuffer spectrum) {
    const std::array bindings{GpuBinding{resolution.Parameters, 0}, GpuBinding{waveform, 1}, GpuBinding{resolution.Transform, 2}, GpuBinding{resolution.Window, 3}, GpuBinding{resolution.Twiddles, 4}};
    DispatchGroupsGpu(gpu, state.Forward, bindings, {resolution.Frames * (resolution.Size > 2048 ? 2u : 1u)}, {256});
    const std::array finish{GpuBinding{resolution.Parameters, 0}, GpuBinding{resolution.Transform, 1}, GpuBinding{spectrum, 2}, GpuBinding{resolution.Twiddles, 3}};
    DispatchGpu(gpu, state.FinishForward, finish, {resolution.Size, resolution.Frames});
}
} // namespace

SpectralLossGpu CreateSpectralLossGpu(Gpu &gpu, std::span<const float> target, uint32_t sample_rate, SpectralLossOptions options) {
    if (target.empty() || target.size() > (1u << 22) || !sample_rate || !std::isfinite(options.MagnitudeFloor) || options.MagnitudeFloor < 1e-12f || options.MagnitudeFloor > 1e6f || !std::isfinite(options.HuberDelta) || options.HuberDelta <= 0 || !std::has_single_bit(options.HopDivisor) || options.HopDivisor > 64 || uint32_t(options.Scale) > uint32_t(SpectralMagnitudeScale::Linear)) throw std::invalid_argument("Invalid spectral loss dimensions or options");
    if (!std::ranges::all_of(target, [](float value) { return std::isfinite(value); })) throw std::invalid_argument("Nonfinite spectral target");
    const uint32_t samples = uint32_t(target.size());
    const SpectralLossGpu state{
        .Samples = samples, .SampleRate = sample_rate, .Options = options, .Resolutions = {CreateResolution(gpu, samples, 4096, options), CreateResolution(gpu, samples, 1024, options), CreateResolution(gpu, samples, 256, options), CreateResolution(gpu, samples, 64, options)}, .Gradient = CreateBuffer(gpu, target.size_bytes()), .Loss = CreateBuffer(gpu, 5 * sizeof(float)), .Forward = CreateKernel(gpu, "SpectralForwardPrecise"), .FinishForward = CreateKernel(gpu, "SpectralFinishForward"), .CompareAdjoint = CreateKernel(gpu, "SpectralCompareAdjoint"), .Overlap = CreateKernel(gpu, "SpectralOverlap"), .Reduce = CreateKernel(gpu, "SpectralReduce")
    };
    const auto waveform = Upload<float>(gpu, target);
    BeginGpu(gpu);
    for (const auto &r : state.Resolutions) Forward(gpu, state, r, waveform, r.Target);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    return state;
}

void EncodeSpectralLoss(Gpu &gpu, const SpectralLossGpu &state, GpuBuffer waveform) {
    if (waveform.Size < size_t(state.Samples) * sizeof(float)) throw std::invalid_argument("Spectral waveform buffer is too small");
    for (const auto &r : state.Resolutions) {
        Forward(gpu, state, r, waveform, r.Spectrum);
        const std::array bindings{GpuBinding{r.Parameters, 0}, GpuBinding{r.Spectrum, 1}, GpuBinding{r.Target, 2}, GpuBinding{r.Adjoint, 3}, GpuBinding{r.FrameLoss, 4}};
        DispatchGroupsGpu(gpu, state.CompareAdjoint, bindings, {r.Frames}, {256});
    }
    const auto &[r0, r1, r2, r3] = state.Resolutions;
    const std::array overlap{GpuBinding{r0.Parameters, 0}, GpuBinding{r1.Parameters, 1}, GpuBinding{r2.Parameters, 2}, GpuBinding{r3.Parameters, 3}, GpuBinding{r0.Adjoint, 4}, GpuBinding{r1.Adjoint, 5}, GpuBinding{r2.Adjoint, 6}, GpuBinding{r3.Adjoint, 7}, GpuBinding{state.Gradient, 8}};
    DispatchGpu(gpu, state.Overlap, overlap, {state.Samples});
    const std::array reduce{GpuBinding{r0.Parameters, 0}, GpuBinding{r1.Parameters, 1}, GpuBinding{r2.Parameters, 2}, GpuBinding{r3.Parameters, 3}, GpuBinding{r0.FrameLoss, 4}, GpuBinding{r1.FrameLoss, 5}, GpuBinding{r2.FrameLoss, 6}, GpuBinding{r3.FrameLoss, 7}, GpuBinding{state.Loss, 8}};
    DispatchGroupsGpu(gpu, state.Reduce, reduce, {1}, {256});
}
} // namespace surface_audio
