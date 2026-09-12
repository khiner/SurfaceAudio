#include "core/GpuFullSpectrum.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>
using namespace surface_audio;
namespace {
void Require(bool x, const char *message) {
    if (!x) throw std::runtime_error(message);
}
struct Evaluation {
    double Loss{};
    std::vector<double> Gradient;
};
Evaluation Reference(std::span<const float> input, std::span<const float> target, SpectralLossOptions options) {
    const uint32_t n = std::bit_ceil(uint32_t(input.size()));
    Evaluation result{.Gradient = std::vector<double>(input.size())};
    for (uint32_t k = 0; k <= n / 2; ++k) {
        std::complex<double> x{}, y{};
        for (uint32_t j = 0; j < input.size(); ++j) {
            const auto basis = std::polar(1., -2 * std::numbers::pi * k * j / n);
            x += double(input[j]) * basis;
            y += double(target[j]) * basis;
        }
        const double a = std::norm(x) + double(options.MagnitudeFloor) * options.MagnitudeFloor, b = std::norm(y) + double(options.MagnitudeFloor) * options.MagnitudeFloor;
        const auto magnitude = [&](double power) { return options.Scale == SpectralMagnitudeScale::Linear ? std::sqrt(power) : options.Scale == SpectralMagnitudeScale::NaturalLog ? .5 * std::log(power) :
                                                                                                                                                                                     10 / std::log(10.) * std::log(power); };
        const double difference = magnitude(a) - magnitude(b), absolute = std::abs(difference), delta = options.HuberDelta, weight = 1. / (n / 2 + 1);
        result.Loss += weight * (absolute <= delta ? .5 * difference * difference : delta * (absolute - .5 * delta));
        const double factor = options.Scale == SpectralMagnitudeScale::Linear ? 1 / std::sqrt(a) : options.Scale == SpectralMagnitudeScale::NaturalLog ? 1 / a :
                                                                                                                                                         20 / std::log(10.) / a;
        const auto adjoint = std::clamp(difference, -delta, delta) * weight * factor * x;
        for (uint32_t j = 0; j < input.size(); ++j) result.Gradient[j] += (adjoint * std::polar(1., 2 * std::numbers::pi * k * j / n)).real();
    }
    return result;
}
}
int main() {
    try {
        auto gpu = CreateGpu();
        std::array<float, 101> input{}, target{};
        for (uint32_t i = 0; i < input.size(); ++i) {
            input[i] = float(.3 * std::sin(.4 * i) + .1 * std::cos(.123 * i));
            target[i] = float(.2 * std::sin(.31 * i));
        }
        for (auto scale : {SpectralMagnitudeScale::Linear, SpectralMagnitudeScale::NaturalLog, SpectralMagnitudeScale::Decibels}) {
            const SpectralLossOptions options{.MagnitudeFloor = .01f, .HuberDelta = 3, .Scale = scale};
            const auto reference = Reference(input, target, options);
            const auto s = CreateFullSpectrumGpu(gpu, target, options);
            const auto waveform = Upload<float>(gpu, input), gradient = CreateBuffer(gpu, sizeof(input)), loss = Upload(gpu, .25f);
            constexpr uint32_t repeats = 7;
            std::ranges::fill(BufferSpan<float>(gradient), .125f);
            BeginGpu(gpu);
            for (uint32_t i = 0; i < repeats; ++i) EncodeFullSpectrumGpu(gpu, s, waveform, gradient, loss);
            SubmitGpu(gpu);
            WaitGpu(gpu);
            Require(std::abs(BufferSpan<float>(s.Loss)[0] - reference.Loss) < 1e-5 * std::max(1., reference.Loss), "Full-spectrum loss against independent DFT");
            Require(std::abs(BufferSpan<float>(loss)[0] - repeats * reference.Loss - .25) < repeats * 1e-5 * std::max(1., reference.Loss), "Additive loss across a growing batch");
            for (uint32_t i = 0; i < input.size(); ++i)
                Require(std::abs(BufferSpan<float>(gradient)[i] - repeats * reference.Gradient[i] - .125) < repeats * 1e-4 * (1 + std::abs(reference.Gradient[i])), "Real sample adjoint including zero padding and FFT normalization");
        }
        std::vector<float> long_input(131073), long_target(long_input.size());
        long_input[0] = .3f;
        long_target[0] = .1f;
        const SpectralLossOptions options{.MagnitudeFloor = .01f, .HuberDelta = 3, .Scale = SpectralMagnitudeScale::Linear};
        const auto spectrum = CreateFullSpectrumGpu(gpu, long_target, options);
        const auto waveform = Upload<float>(gpu, long_input), gradient = Upload<float>(gpu, std::vector<float>(long_input.size())), loss = Upload(gpu, 0.f);
        BeginGpu(gpu);
        EncodeFullSpectrumGpu(gpu, spectrum, waveform, gradient, loss);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const double magnitude = std::hypot(double(long_input[0]), double(options.MagnitudeFloor));
        const double difference = magnitude - std::hypot(double(long_target[0]), double(options.MagnitudeFloor));
        const double derivative = difference * long_input[0] / magnitude;
        Require(std::abs(BufferSpan<float>(loss)[0] - .5 * difference * difference) < 1e-7, "Long zero-padded impulse spectrum");
        for (size_t i = 0; i < long_input.size(); ++i) {
            const double expected = i == 0 ? derivative : i % 2 ? 0 :
                                                                  derivative / (spectrum.Size / 2 + 1);
            Require(std::abs(BufferSpan<float>(gradient)[i] - expected) < 1e-7, "Long transform adjoint against analytic impulse spectrum");
        }
        std::cout << "Full-spectrum loss and adjoint passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
