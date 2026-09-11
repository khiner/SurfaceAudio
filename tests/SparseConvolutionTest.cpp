#include "core/GpuSparseConvolution.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void DenseOracle(Gpu &gpu) {
    const std::array<float, 5> impulse{.2f, -.7f, 1, -.6f, .1f};
    const std::array<std::pair<uint32_t, double>, 5> events{{{0, .25}, {7, .75}, {9, .5}, {24, .3}, {31, .8}}};
    std::vector<float> signal(32);
    for (const auto [sample, amplitude] : events)
        for (size_t k = 0; k < impulse.size() && sample + k < signal.size(); ++k) signal[sample + k] += amplitude * impulse[k];
    for (size_t n = 0; n < signal.size(); ++n) signal[n] += .005 * std::cos(.2 * n);
    // Independent dense SciPy NNLS solution with a zero target for the omitted convolution tail.
    std::vector<double> expected(32);
    expected[0] = .24062974474106533;
    expected[7] = .7432257725318417;
    expected[9] = .49320899123389345;
    expected[24] = .2897719773927386;
    expected[26] = .0020879284812234116;
    expected[29] = .07634391418984707;
    const auto fit = FitSparseConvolutionGpu(gpu, signal, impulse, .01);
    for (size_t n = 0; n < signal.size(); ++n)
        Require(std::abs(expected[n] - fit.Coefficients[n]) < 2e-5, "Sparse convolution differs from dense NNLS oracle");
    Require(fit.RelativeKkt < 2e-5, "Sparse convolution optimality residual");
    std::vector<double> residual(signal.size() + impulse.size() - 1);
    for (size_t n = 0; n < signal.size(); ++n) {
        residual[n] -= signal[n];
        for (size_t k = 0; k < impulse.size(); ++k) residual[n + k] += fit.Coefficients[n] * double(impulse[k]);
    }
    double peak = 0, violation = 0;
    for (size_t n = 0; n < signal.size(); ++n) {
        double rhs = 0;
        for (size_t k = 0; k < impulse.size() && n + k < signal.size(); ++k) rhs += impulse[k] * double(signal[n + k]);
        peak = std::max(peak, rhs);
    }
    for (size_t n = 0; n < signal.size(); ++n) {
        double gradient = .01 * peak;
        for (size_t k = 0; k < impulse.size(); ++k) gradient += impulse[k] * residual[n + k];
        violation = std::max(violation, fit.Coefficients[n] > 0 ? std::abs(gradient) : std::max(0., -gradient));
    }
    Require(violation / peak < 2e-5, "Independent full-convolution KKT failed");
    for (auto &sample : signal) sample *= 8;
    const auto scaled = FitSparseConvolutionGpu(gpu, signal, impulse, .01);
    for (size_t n = 0; n < signal.size(); ++n)
        Require(std::abs(scaled.Coefficients[n] / 8 - fit.Coefficients[n]) < 2e-5, "Sparse convolution scale dependence");
}
void SweepingContacts(Gpu &gpu) {
    constexpr uint32_t rate = 44100, frames = 2 * rate;
    std::vector<float> impact(2640), signal(frames);
    for (size_t n = 0; n < impact.size(); ++n) {
        const double t = double(n) / rate;
        impact[n] = std::exp(-65 * t) * (std::sin(2 * std::numbers::pi * 137 * t) + .4 * std::sin(2 * std::numbers::pi * 301 * t));
    }
    std::vector<uint32_t> contacts;
    for (double time = .1; time < 1.85; time += 1 / (20 + 30 * time)) contacts.push_back(uint32_t(std::round(time * rate)));
    for (auto sample : contacts)
        for (size_t n = 0; n < impact.size(); ++n) signal[sample + n] += impact[n];
    const auto fit = FitSparseConvolutionGpu(gpu, signal, impact);
    double near = 0, total = 0, error = 0, energy = 0;
    std::vector<double> reconstructed(frames + impact.size() - 1);
    for (size_t n = 0; n < frames; ++n) {
        total += fit.Coefficients[n];
        if (std::ranges::any_of(contacts, [&](auto sample) { return std::abs(double(sample) - n) <= rate * .001; })) near += fit.Coefficients[n];
        if (fit.Coefficients[n] > 0)
            for (size_t k = 0; k < impact.size(); ++k) reconstructed[n + k] += fit.Coefficients[n] * double(impact[k]);
    }
    for (size_t n = 0; n < reconstructed.size(); ++n) {
        const double reference = n < signal.size() ? signal[n] : 0;
        error += std::pow(reference - reconstructed[n], 2);
        energy += reference * reference;
    }
    std::cout << "Sweep contact mass within 1 ms " << near / total << ", waveform error " << std::sqrt(error / energy) << '\n';
    Require(near / total > .99, "Overlapping oscillatory impacts lose the contact sweep");
    Require(std::sqrt(error / energy) < .02, "Sweeping contact waveform mismatch");
    Require(fit.RelativeKkt < .003, "Sweep optimality residual");
}
void JointObservations(Gpu &gpu) {
    const std::array<float, 8> first{.2f, -.1f, .7f, .3f, -.4f, .6f, .1f, 0}, second{.1f, .5f, .2f, -.3f, .4f, .8f, -.1f, .2f};
    const std::array<float, 3> first_impulse{1, -.4f, .2f};
    const std::array<float, 4> second_impulse{.3f, .8f, -.2f, .1f};
    const std::array observations{ConvolutionObservation{first, first_impulse, .5}, ConvolutionObservation{second, second_impulse, 3}};
    // Independent dense SciPy NNLS solution with both complete convolution tails and lambda=0.01*max(sum(w*I.T*s)).
    const std::array expected{.4280761721321968, .3377870375955054, 0., .12025830903644416,
                              .6938817322432762, .21910076261331374, .10605054612293417, .023114815239913305};
    const auto fit = FitSparseConvolutionGpu(gpu, observations, .01);
    for (size_t n = 0; n < expected.size(); ++n)
        Require(std::abs(fit.Coefficients[n] - expected[n]) < 2e-5, "Joint convolution differs from dense NNLS oracle");
    Require(fit.RelativeKkt < 2e-5, "Joint convolution optimality residual");
    const std::array scaled{ConvolutionObservation{second, second_impulse, 24}, ConvolutionObservation{first, first_impulse, 4}};
    const auto repeated = FitSparseConvolutionGpu(gpu, scaled, .01);
    for (size_t n = 0; n < expected.size(); ++n)
        Require(std::abs(repeated.Coefficients[n] - fit.Coefficients[n]) < 2e-5, "Joint fit depends on observation order or common weight scale");
}
void Boundaries(Gpu &gpu) {
    const std::array<float, 2> silent{}, impulse{1, -.5}, nonfinite{0, std::numeric_limits<float>::quiet_NaN()};
    const auto fit = FitSparseConvolutionGpu(gpu, silent, impulse);
    Require(fit.Coefficients == std::vector<float>(2) && fit.RelativeKkt == 0, "Silent source fit");
    const auto reject = [](auto action) { try { action(); } catch (const std::invalid_argument &) { return true; } return false; };
    Require(reject([&] { FitSparseConvolutionGpu(gpu, impulse, silent); }), "Zero impulse accepted");
    Require(reject([&] { FitSparseConvolutionGpu(gpu, nonfinite, impulse); }), "Nonfinite signal accepted");
    Require(reject([&] { FitSparseConvolutionGpu(gpu, impulse, impulse, 1); }), "Invalid penalty accepted");
    Require(reject([&] { FitSparseConvolutionGpu(gpu, impulse, impulse, .001, 0); }), "Zero iterations accepted");
    Require(reject([&] { FitSparseConvolutionGpu(gpu, std::array{ConvolutionObservation{impulse, impulse, -1}}); }), "Negative observation weight accepted");
    Require(reject([&] { FitSparseConvolutionGpu(gpu, std::array{ConvolutionObservation{impulse, impulse}, ConvolutionObservation{{}, impulse}}); }),
            "Unequal observation lengths accepted");
}
}
int main() {
    try {
        auto gpu = CreateGpu();
        DenseOracle(gpu);
        SweepingContacts(gpu);
        JointObservations(gpu);
        Boundaries(gpu);
        std::cout << "Sparse convolution waveform inversion passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
