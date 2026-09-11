#include "core/SignalAnalysis.h"
#include "core/GpuConvolution.h"
#include "core/GpuFft.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>

using namespace surface_audio;
namespace {
using Complex = std::complex<double>;
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void FourierTest(Gpu &gpu) {
    std::mt19937 random(72);
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (const uint32_t size : {1u, 32u, 4096u, 8192u, 262144u}) {
        const uint32_t count = size < 8192 ? 3 : 1;
        std::vector<std::complex<float>> input(size_t(size) * count);
        for (auto &value : input) value = {distribution(random), distribution(random)};
        std::vector<Complex> expected(input.begin(), input.end());
        for (uint32_t record = 0; record < count; ++record) {
            auto signal = std::span(expected).subspan(size_t(record) * size, size);
            FourierTransform(signal);
            if (size <= 32) {
                for (uint32_t bin = 0; bin < size; ++bin) {
                    Complex direct{};
                    for (uint32_t sample = 0; sample < size; ++sample) direct += Complex(input[size_t(record) * size + sample]) * std::polar(1., -2 * std::numbers::pi * bin * sample / size);
                    Require(std::abs(direct - signal[bin]) < 1e-11, "CPU FFT matches direct DFT");
                }
            }
        }
        const auto forward = CreateFftGpu(gpu, size, count), inverse = CreateFftGpu(gpu, size, count);
        const auto source = Upload<std::complex<float>>(gpu, input);
        BeginGpu(gpu);
        EncodeFftGpu(gpu, forward, source);
        EncodeFftGpu(gpu, inverse, forward.Output, true);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto actual = BufferSpan<std::complex<float>>(forward.Output), restored = BufferSpan<std::complex<float>>(inverse.Output);
        double error{}, energy{}, inverse_error{};
        for (size_t i = 0; i < expected.size(); ++i) {
            error += std::norm(expected[i] - Complex(actual[i]));
            energy += std::norm(expected[i]);
            inverse_error = std::max(inverse_error, std::abs(Complex(restored[i]) - Complex(input[i])));
        }
        Require(std::sqrt(error / energy) < 8e-7 && inverse_error < 2e-6, "GPU complex FFT preserves complete records and inverse normalization");
        for (uint32_t record = 0; record < count; ++record) FourierTransform(std::span(expected).subspan(size_t(record) * size, size), true);
        for (size_t i = 0; i < expected.size(); ++i) Require(std::abs(expected[i] - Complex(input[i])) < 1e-12, "CPU inverse FFT preserves complex samples");
        std::cout << "FFT " << size << " x " << count << " relative " << std::sqrt(error / energy) << " inverse max " << inverse_error << '\n';
    }
}
void PredictionTest() {
    const std::array signal{1., 2., 3., 4., 5.};
    const auto fit = FitLinearPrediction(signal, 2);
    Require(std::abs(fit.Denominator[1] + 232. / 285) < 1e-14 && std::abs(fit.Denominator[2] - 34. / 285) < 1e-14, "LPC solves independently derived Toeplitz equations");
    Require(std::abs(fit.Error - (55 - 40 * 232. / 285 + 26 * 34. / 285)) < 1e-13, "LPC reports unnormalized prediction energy");
    const auto silent = FitLinearPrediction(std::array<double, 32>{}, 20);
    Require(silent.Error == 0 && silent.Denominator[0] == 1 && std::ranges::all_of(std::span(silent.Denominator).subspan(1), [](double x) { return x == 0; }), "Silent LPC remains finite");
}
void EspritTest() {
    const std::array<DampedSinusoid, 3> modes{{{std::polar(.993, .31), {.4, -.2}}, {std::polar(.982, -.63), {-.7, .3}}, {std::polar(.999, 1.3), {.2, .8}}}};
    std::vector<Complex> signal(127);
    for (const auto mode : modes) {
        Complex value = mode.Gain;
        for (auto &sample : signal) {
            sample += value;
            value *= mode.Pole;
        }
    }
    const auto fitted = EstimateEsprit(signal, modes.size());
    for (const auto mode : modes) {
        const auto nearest = std::ranges::min_element(fitted, {}, [&](auto candidate) { return std::abs(candidate.Pole - mode.Pole); });
        Require(std::abs(nearest->Pole - mode.Pole) < 1e-12 && std::abs(nearest->Gain - mode.Gain) < 1e-11, "ESPRIT recovers planted complex poles and gains");
    }
    double maximum{};
    for (size_t i = 0; i < signal.size(); ++i) {
        Complex expected{};
        for (const auto mode : fitted) expected += mode.Gain * std::pow(mode.Pole, double(i));
        maximum = std::max(maximum, std::abs(signal[i] - expected));
    }
    Require(maximum < 1e-11, "ESPRIT reconstructs the full independent signal");
    std::cout << "ESPRIT full-record maximum " << maximum << '\n';
}
void ConvolutionTest(Gpu &gpu) {
    std::vector<float> input(4099), taps(2051);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(std::sin(.073 * i) * std::exp(-.002 * i));
    for (size_t i = 0; i < taps.size(); ++i) taps[i] = float(std::cos(.053 * i) * std::exp(-.003 * i));
    const auto actual = ConvolveFftGpu(gpu, input, taps);
    std::vector<double> reference(input.size() + taps.size() - 1);
    for (size_t i = 0; i < input.size(); ++i)
        for (size_t j = 0; j < taps.size(); ++j) reference[i + j] += double(input[i]) * taps[j];
    double error{}, energy{}, maximum{};
    Require(actual.size() == reference.size(), "FFT convolution retains the complete tail");
    for (size_t i = 0; i < reference.size(); ++i) {
        error += std::pow(reference[i] - actual[i], 2);
        energy += reference[i] * reference[i];
        maximum = std::max(maximum, std::abs(reference[i] - actual[i]));
    }
    Require(std::sqrt(error / energy) < 1e-6 && maximum < 1e-5, "FFT convolution matches independent full-record direct sum");
    const auto identity = ConvolveFftGpu(gpu, std::array{2.f}, std::array{3.f});
    Require(identity.size() == 1 && identity[0] == 6, "Scalar FFT convolution preserves gain");
    std::cout << "FFT convolution relative " << std::sqrt(error / energy) << " maximum " << maximum << '\n';
}
}
int main() {
    try {
        PredictionTest();
        EspritTest();
        auto gpu = CreateGpu();
        FourierTest(gpu);
        ConvolutionTest(gpu);
        std::cout << "Signal analysis passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
