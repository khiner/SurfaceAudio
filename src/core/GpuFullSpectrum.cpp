#include "GpuFullSpectrum.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace surface_audio {
namespace {
struct FullSpectrumBlock {
    uint32_t Samples, Size, Scale;
    float FloorSquared, Delta;
};
void Transform(Gpu &gpu, const FullSpectrumGpu &s, GpuBuffer waveform) {
    const std::array bindings{GpuBinding{s.Parameters, 0}, GpuBinding{waveform, 1}, GpuBinding{s.Input, 2}};
    DispatchGpu(gpu, s.Pack, bindings, {s.Size});
    EncodeFftGpu(gpu, s.Fft, s.Input);
}
}
FullSpectrumGpu CreateFullSpectrumGpu(Gpu &gpu, std::span<const float> target, SpectralLossOptions options) {
    if (target.empty() || target.size() > (1u << 22) || !std::ranges::all_of(target, [](float x) { return std::isfinite(x); }) || !std::isfinite(options.MagnitudeFloor) || options.MagnitudeFloor < 1e-12f || options.MagnitudeFloor > 1e6f || !std::isfinite(options.HuberDelta) || options.HuberDelta <= 0 || uint32_t(options.Scale) > 2) throw std::invalid_argument("Invalid full-spectrum loss");
    const uint32_t samples = uint32_t(target.size()), size = std::bit_ceil(samples);
    const size_t bytes = size_t(size) * 2 * sizeof(float);
    const FullSpectrumGpu s{.Samples = samples, .Size = size, .Fft = CreateFftGpu(gpu, size), .Parameters = Upload(gpu, FullSpectrumBlock{samples, size, uint32_t(options.Scale), options.MagnitudeFloor * options.MagnitudeFloor, options.HuberDelta}), .Input = CreateBuffer(gpu, bytes), .Target = CreateBuffer(gpu, bytes), .BinLoss = CreateBuffer(gpu, size_t(size / 2 + 1) * sizeof(float)), .Loss = CreateBuffer(gpu, sizeof(float)), .Pack = CreateKernel(gpu, "FullSpectrumPack"), .Compare = CreateKernel(gpu, "FullSpectrumCompare"), .Accumulate = CreateKernel(gpu, "FullSpectrumAccumulate"), .Reduce = CreateKernel(gpu, "FullSpectrumReduce")};
    const auto waveform = Upload<float>(gpu, target);
    BeginGpu(gpu);
    Transform(gpu, s, waveform);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    std::ranges::copy(BufferSpan<float>(s.Fft.Output), BufferSpan<float>(s.Target).begin());
    return s;
}
void EncodeFullSpectrumGpu(Gpu &gpu, const FullSpectrumGpu &s, GpuBuffer waveform, GpuBuffer gradient, GpuBuffer loss) {
    if (waveform.Size < size_t(s.Samples) * sizeof(float) || gradient.Size < size_t(s.Samples) * sizeof(float) || loss.Size < sizeof(float)) throw std::invalid_argument("Invalid full-spectrum buffers");
    Transform(gpu, s, waveform);
    const std::array compare{GpuBinding{s.Parameters, 0}, GpuBinding{s.Fft.Output, 1}, GpuBinding{s.Target, 2}, GpuBinding{s.Input, 3}, GpuBinding{s.BinLoss, 4}};
    DispatchGpu(gpu, s.Compare, compare, {s.Size});
    EncodeFftGpu(gpu, s.Fft, s.Input, true);
    const std::array accumulate{GpuBinding{s.Parameters, 0}, GpuBinding{s.Fft.Output, 1}, GpuBinding{gradient, 2}};
    DispatchGpu(gpu, s.Accumulate, accumulate, {s.Samples});
    const std::array reduce{GpuBinding{s.Parameters, 0}, GpuBinding{s.BinLoss, 1}, GpuBinding{s.Loss, 2}, GpuBinding{loss, 3}};
    DispatchGroupsGpu(gpu, s.Reduce, reduce, {1}, {256});
}
}
