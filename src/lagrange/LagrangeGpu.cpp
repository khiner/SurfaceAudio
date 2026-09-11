#include "Lagrange.h"
#include "core/GpuConvolution.h"
#include "core/GpuFft.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>

namespace surface_audio::lagrange {
namespace {
void ValidateModes(std::span<const Resonance> modes, double rate, bool inverse) {
    if (!std::isfinite(rate) || !(rate > 0) || modes.size() > 1024) throw std::invalid_argument("Invalid modal dimensions");
    for (auto mode : modes) {
        if (!(mode.Frequency >= 0 && mode.Frequency <= rate / 2) || !std::isfinite(mode.Damping) || mode.Damping < 0 ||
            !std::isfinite(float(mode.Damping / rate)) || !std::isfinite(float(mode.RealGain)) || !std::isfinite(float(mode.ImaginaryGain)) ||
            (inverse && !(float(mode.Damping / rate) > 0))) throw std::invalid_argument("Invalid modal parameters");
    }
}
void ValidateSignal(std::span<const float> signal) {
    if (signal.empty() || signal.size() > (1u << 23)) throw std::invalid_argument("Invalid signal dimensions");
    for (float value : signal)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite signal");
}
}
std::vector<float> MovingCombGpu(Gpu &gpu, std::span<const float> signal, std::span<const float> position, double length, double speed, uint32_t rate, float first, float second) {
    ValidateSignal(signal);
    if (signal.size() != position.size() || !std::isfinite(float(length)) || !std::isfinite(speed) || !(length > 0) || !(speed > 0) ||
        !rate || !std::isfinite(first) || !std::isfinite(second) || !std::isfinite(float(rate / speed)) || !(length * rate / speed < (1u << 30)))
        throw std::invalid_argument("Invalid moving comb dimensions");
    for (float location : position)
        if (!(location >= 0 && location <= length)) throw std::invalid_argument("Contact position outside plate");
    struct Block {
        uint32_t Frames;
        float Length, DelayScale, First, Second;
    };
    const auto input = Upload<float>(gpu, signal), locations = Upload<float>(gpu, position);
    const auto output = CreateBuffer(gpu, signal.size_bytes());
    const auto kernel = CreateKernel(gpu, "LagrangeMovingComb");
    BeginGpu(gpu);
    const auto constants = BatchUpload(gpu, Block{uint32_t(signal.size()), float(length), float(rate / speed), first, second});
    const std::array bindings{GpuBinding{constants, 0}, GpuBinding{input, 1}, GpuBinding{locations, 2}, GpuBinding{output, 3}};
    DispatchGpu(gpu, kernel, bindings, {uint32_t(signal.size()), 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto result = BufferSpan<float>(output);
    return {result.begin(), result.end()};
}
std::vector<float> ModalResponseGpu(Gpu &gpu, std::span<const Resonance> modes, uint32_t frames, double rate) {
    ValidateModes(modes, rate, false);
    if (!frames || frames > (1u << 23)) throw std::invalid_argument("Invalid modal response dimensions");
    std::vector<std::array<float, 8>> parameters;
    for (auto mode : modes) {
        const double frequency = mode.Frequency / rate;
        parameters.push_back({float(frequency), float(frequency - float(frequency)), float(mode.Damping / rate), float(mode.RealGain), float(mode.ImaginaryGain), 0, 0, 0});
    }
    if (parameters.empty()) return std::vector<float>(frames);
    const auto poles = Upload<std::array<float, 8>>(gpu, parameters), output = CreateBuffer(gpu, size_t(frames) * sizeof(float));
    const auto kernel = CreateKernel(gpu, "LagrangeResponse");
    BeginGpu(gpu);
    const auto constants = BatchUpload(gpu, std::array<uint32_t, 2>{frames, uint32_t(modes.size())});
    const std::array bindings{GpuBinding{constants, 0}, GpuBinding{poles, 1}, GpuBinding{output, 2}};
    DispatchGpu(gpu, kernel, bindings, {frames, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto result = BufferSpan<float>(output);
    return {result.begin(), result.end()};
}
std::vector<float> FilterModalGpu(Gpu &gpu, std::span<const float> signal, std::span<const Resonance> modes, uint32_t frames, double rate) {
    ValidateSignal(signal);
    const auto response = ModalResponseGpu(gpu, modes, frames, rate);
    auto result = ConvolveFftGpu(gpu, signal.first(std::min(signal.size(), size_t(frames))), response);
    result.resize(frames);
    return result;
}
std::vector<float> InverseModalGpu(Gpu &gpu, std::span<const float> signal, std::span<const Resonance> modes, uint32_t rate, double floor) {
    ValidateSignal(signal);
    ValidateModes(modes, rate, true);
    if (modes.empty() || !(floor >= 0 && floor <= std::sqrt(std::numeric_limits<float>::max())))
        throw std::invalid_argument("Invalid modal inverse dimensions");
    const uint32_t size = std::bit_ceil(uint32_t(signal.size()) * 2);
    std::vector<std::array<float, 2>> samples(size);
    for (size_t n = 0; n < signal.size(); ++n) samples[n] = {signal[n], 0};
    std::vector<std::array<float, 4>> parameters;
    for (auto mode : modes) {
        if (!(mode.Damping > 0)) throw std::invalid_argument("Undamped modal inverse pole");
        parameters.push_back({float(2 * std::numbers::pi * mode.Frequency / rate), float(mode.Damping / rate), float(mode.RealGain), float(mode.ImaginaryGain)});
    }
    const auto input = Upload<std::array<float, 2>>(gpu, samples), poles = Upload<std::array<float, 4>>(gpu, parameters);
    const auto transform = CreateFftGpu(gpu, size);
    const auto divided = CreateBuffer(gpu, size_t(size) * sizeof(std::array<float, 2>));
    const auto kernel = CreateKernel(gpu, "LagrangeInverse");
    struct Block {
        uint32_t Size, Modes;
        float Floor, Padding;
    };
    BeginGpu(gpu);
    EncodeFftGpu(gpu, transform, input);
    const auto constants = BatchUpload(gpu, Block{size, uint32_t(modes.size()), float(floor), 0});
    const std::array bindings{GpuBinding{constants, 0}, GpuBinding{poles, 1}, GpuBinding{transform.Output, 2}, GpuBinding{divided, 3}};
    DispatchGpu(gpu, kernel, bindings, {size, 1, 1});
    EncodeFftGpu(gpu, transform, divided, true);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto transformed = BufferSpan<std::array<float, 2>>(transform.Output);
    std::vector<float> result(signal.size());
    for (size_t n = 0; n < result.size(); ++n) result[n] = transformed[n][0];
    return result;
}
std::vector<float> DeconvolveEnvelopeGpu(Gpu &gpu, std::span<const float> envelope, std::span<const float> shape, double relative_floor) {
    ValidateSignal(envelope);
    ValidateSignal(shape);
    if (envelope.size() + shape.size() > (1u << 24) || !(relative_floor > 0 && relative_floor <= 1))
        throw std::invalid_argument("Invalid envelope inverse dimensions");
    double sum = 0;
    for (float value : shape) {
        if (!(value >= 0)) throw std::invalid_argument("Negative impact envelope");
        sum += value;
    }
    const float floor = float(sum * relative_floor);
    if (!(floor * floor > 0) || !std::isfinite(floor * floor)) throw std::invalid_argument("Invalid envelope inverse floor");
    const uint32_t size = std::bit_ceil(uint32_t(envelope.size() + shape.size()));
    std::vector<std::array<float, 2>> samples(size_t(2) * size);
    for (size_t n = 0; n < envelope.size(); ++n) samples[n] = {envelope[n], 0};
    for (size_t n = 0; n < shape.size(); ++n) samples[size + n] = {shape[n], 0};
    const auto input = Upload<std::array<float, 2>>(gpu, samples);
    const auto forward = CreateFftGpu(gpu, size, 2), inverse = CreateFftGpu(gpu, size);
    const auto divided = CreateBuffer(gpu, size_t(size) * sizeof(std::array<float, 2>));
    const auto kernel = CreateKernel(gpu, "LagrangeEnvelopeInverse");
    struct Block { uint32_t Size, Modes; float Floor, Padding; };
    BeginGpu(gpu);
    EncodeFftGpu(gpu, forward, input);
    const auto constants = BatchUpload(gpu, Block{size, 0, floor, 0});
    const std::array bindings{GpuBinding{constants, 0}, GpuBinding{forward.Output, 1}, GpuBinding{divided, 2}};
    DispatchGpu(gpu, kernel, bindings, {size, 1, 1});
    EncodeFftGpu(gpu, inverse, divided, true);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto transformed = BufferSpan<std::array<float, 2>>(inverse.Output);
    std::vector<float> result(envelope.size());
    for (size_t n = 0; n < result.size(); ++n) result[n] = transformed[n][0];
    return result;
}
}
