#include "GpuFiniteModes.h"
#include "GpuMix.h"
#include <array>
#include <cmath>
#include <complex>
#include <numbers>

namespace surface_audio {
std::vector<float> ConvolveFiniteModesGpu(Gpu &gpu, std::span<const float> excitation, std::span<const float> morph, std::span<const FiniteMode> modes, uint32_t sample_rate, uint32_t taps, float gain) {
    const size_t count = excitation.size() + size_t(taps) - 1;
    if (excitation.empty() || morph.size() != excitation.size() || modes.empty() || !sample_rate || !taps || count > UINT32_MAX || modes.size() > UINT32_MAX / count || !std::isfinite(gain)) throw std::invalid_argument("Invalid finite modal convolution dimensions");
    for (float value : excitation)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite modal excitation");
    for (float value : morph)
        if (!std::isfinite(value) || value < 0 || value > 1) throw std::invalid_argument("Invalid modal interpolation weight");
    std::vector<float> coefficients(6 * modes.size());
    for (size_t index = 0; index < modes.size(); ++index) {
        const auto &mode = modes[index];
        if (!(mode.Frequency > 0 && mode.Frequency < sample_rate / 2. && mode.Decay > 0 && mode.Amplitude0 > 0 && mode.Amplitude1 > 0) || !std::isfinite(mode.Decay) || !std::isfinite(mode.Amplitude0) || !std::isfinite(mode.Amplitude1)) throw std::invalid_argument("Invalid finite modal parameters");
        const auto pole = std::polar(std::exp(-1 / (sample_rate * mode.Decay)), 2 * std::numbers::pi * mode.Frequency / sample_rate);
        const std::complex<double> rounded{float(pole.real()), float(pole.imag())};
        if (std::norm(rounded) >= 1 || !std::isfinite(float(mode.Amplitude0)) || !std::isfinite(float(mode.Amplitude1)) || float(mode.Amplitude0) <= 0 || float(mode.Amplitude1) <= 0) throw std::invalid_argument("Finite modal parameters lose float stability or amplitude range");
        const auto delayed = std::pow(rounded, taps);
        coefficients[index] = float(rounded.real());
        coefficients[modes.size() + index] = float(rounded.imag());
        coefficients[2 * modes.size() + index] = float(delayed.real());
        coefficients[3 * modes.size() + index] = float(delayed.imag());
        coefficients[4 * modes.size() + index] = float(mode.Amplitude0);
        coefficients[5 * modes.size() + index] = float(mode.Amplitude1);
    }
    if (taps == 1) return std::vector<float>(count, 0);
    const std::array block{uint32_t(count), uint32_t(excitation.size()), uint32_t(modes.size()), taps};
    const auto parameters = Upload<uint32_t>(gpu, block), coefficient_buffer = Upload<float>(gpu, coefficients), input = Upload<float>(gpu, excitation), location_buffer = Upload<float>(gpu, morph), output = CreateBuffer(gpu, count * modes.size() * sizeof(float));
    const auto kernel = CreateKernel(gpu, "FiniteModalConvolve");
    const auto mix = CreateGpuMix(gpu, {uint32_t(modes.size()), uint32_t(count), 1, 0, gain});
    const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{coefficient_buffer, 1}, GpuBinding{input, 2}, GpuBinding{location_buffer, 3}, GpuBinding{output, 4}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {uint32_t(modes.size())});
    EncodeMix(gpu, mix, output);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(mix.Output);
    return {samples.begin(), samples.end()};
}
} // namespace surface_audio
