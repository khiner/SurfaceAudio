#include "core/Adam.h"
#include "core/GpuEndpointMix.h"
#include "core/GpuSpectral.h"
#include "core/Random.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

using namespace surface_audio;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename F> void Reject(F &&function, const char *message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    Require(rejected, message);
}

std::vector<double> Reference(std::span<const float> basis, std::span<const float> location, std::span<const double> parameters) {
    const size_t modes = parameters.size() / 2;
    std::vector<double> output(location.size(), 0);
    for (size_t frame = 0; frame < location.size(); ++frame)
        for (size_t mode = 0; mode < modes; ++mode) {
            const double x = location[frame];
            output[frame] += basis[mode * location.size() + frame] * std::exp((1 - x) * parameters[mode] + x * parameters[modes + mode]);
        }
    return output;
}

std::vector<double> ReferenceGradient(std::span<const float> basis, std::span<const float> location, std::span<const double> parameters, std::span<const float> adjoint) {
    const size_t modes = parameters.size() / 2;
    std::vector<double> gradient(parameters.size(), 0);
    for (size_t frame = 0; frame < location.size(); ++frame)
        for (size_t mode = 0; mode < modes; ++mode) {
            const double x = location[frame];
            const double component = adjoint[frame] * double(basis[mode * location.size() + frame]) * std::exp((1 - x) * parameters[mode] + x * parameters[modes + mode]);
            gradient[mode] += component * (1 - x);
            gradient[modes + mode] += component * x;
        }
    return gradient;
}

void FormulaTest(Gpu &gpu, uint32_t modes, uint32_t frames, bool tail_only) {
    auto random = MakeRandom(131 + modes + frames);
    const auto basis = std::views::iota(0u, modes * frames) | std::views::transform([&](uint32_t) { return .2f * (Uniform(random) - .5f); }) | std::ranges::to<std::vector>();
    const auto location = std::views::iota(0u, frames) | std::views::transform([&](uint32_t frame) { return frame % 17 == 0 ? 0.f : frame % 19 == 0 ? 1.f :
                                                                                                                                                      Uniform(random); }) | std::ranges::to<std::vector>();
    const auto initial = std::views::iota(0u, 2 * modes) | std::views::transform([&](uint32_t) { return 2 * Uniform(random) - 1; }) | std::ranges::to<std::vector>();
    const auto adjoint = std::views::iota(0u, frames) | std::views::transform([&](uint32_t frame) { return tail_only && frame < frames * 4 / 5 ? 0.f : Uniform(random) - .5f; }) | std::ranges::to<std::vector>();
    const auto state = CreateGpuEndpointMix(gpu, basis, location, initial);
    const auto input = Upload<float>(gpu, adjoint);
    BeginGpu(gpu);
    EncodeEndpointMix(gpu, state);
    EncodeEndpointMixGradient(gpu, state, input);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const std::vector<double> parameters(initial.begin(), initial.end());
    const auto expected = Reference(basis, location, parameters), expected_gradient = ReferenceGradient(basis, location, parameters, adjoint);
    const auto output = BufferSpan<float>(state.Output), gradient = BufferSpan<float>(state.Gradient);
    double error = 0, energy = 0, peak = 0, maximum = 0, gradient_error = 0;
    for (size_t frame = 0; frame < frames; ++frame) {
        const double difference = output[frame] - expected[frame];
        error += difference * difference;
        energy += expected[frame] * expected[frame];
        peak = std::max(peak, std::abs(expected[frame]));
        maximum = std::max(maximum, std::abs(difference));
    }
    for (size_t index = 0; index < gradient.size(); ++index) gradient_error = std::max(gradient_error, std::abs(gradient[index] - expected_gradient[index]) / std::max(.01, std::abs(expected_gradient[index])));
    Require(maximum < 4e-6 * std::max(1., peak) && std::sqrt(error / energy) < 3e-6, "GPU endpoint waveform versus independent double formula");
    Require(gradient_error < 5e-5, "GPU endpoint adjoint versus independent double derivative");
    for (uint32_t endpoint = 0; endpoint < 2; ++endpoint) {
        const auto direction = std::views::iota(0u, 2 * modes) | std::views::transform([&](uint32_t index) { return index / modes == endpoint ? double(Uniform(random) - .5f) : 0.; }) | std::ranges::to<std::vector>();
        const double analytic = std::inner_product(gradient.begin(), gradient.end(), direction.begin(), 0.);
        constexpr double step = 1e-4;
        const auto perturbed = [&](double sign) { return std::views::iota(size_t(0), parameters.size()) | std::views::transform([&](size_t index) { return parameters[index] + sign * step * direction[index]; }) | std::ranges::to<std::vector>(); };
        const auto plus = Reference(basis, location, perturbed(1)), minus = Reference(basis, location, perturbed(-1));
        double numeric = 0;
        for (size_t frame = 0; frame < frames; ++frame) numeric += adjoint[frame] * (plus[frame] - minus[frame]) / (2 * step);
        Require(std::abs(analytic - numeric) < 5e-5 * std::max(.01, std::abs(numeric)), "Endpoint adjoint versus independent central finite difference");
    }
    const std::vector<float> saved(output.begin(), output.end()), saved_gradient(gradient.begin(), gradient.end());
    BeginGpu(gpu);
    EncodeEndpointMix(gpu, state);
    EncodeEndpointMixGradient(gpu, state, input);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    Require(std::equal(saved.begin(), saved.end(), output.begin()) && std::equal(saved_gradient.begin(), saved_gradient.end(), gradient.begin()), "Endpoint workspace reuse is deterministic");
    std::cout << "Endpoint " << modes << " modes, " << frames << " frames, tail adjoint " << tail_only << ": peak error " << maximum << ", gradient error " << gradient_error << '\n';
}

void EndpointLocationTest(Gpu &gpu) {
    const std::array basis{2.f, -1.f, .5f}, adjoint{1.f, -.3f, .7f};
    const std::array parameters{-.7f, 1.2f};
    const auto input = Upload<float>(gpu, adjoint);
    for (uint32_t endpoint = 0; endpoint < 2; ++endpoint) {
        const std::array location{float(endpoint), float(endpoint), float(endpoint)};
        const auto state = CreateGpuEndpointMix(gpu, basis, location, parameters);
        BeginGpu(gpu);
        EncodeEndpointMix(gpu, state);
        EncodeEndpointMixGradient(gpu, state, input);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto output = BufferSpan<float>(state.Output), gradient = BufferSpan<float>(state.Gradient);
        for (size_t frame = 0; frame < output.size(); ++frame) Require(std::abs(output[frame] - basis[frame] * std::exp(double(parameters[endpoint]))) < 2e-6, "Exact endpoint location selects its endpoint amplitude");
        Require(gradient[1 - endpoint] == 0, "Unused endpoint has exactly zero adjoint");
    }
}

void ValidationTest(Gpu &gpu) {
    const std::array basis{1.f, 2.f}, location{0.f, 1.f}, parameters{0.f, 0.f};
    const float infinity = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    Reject([&] { CreateGpuEndpointMix(gpu, {}, location, parameters); }, "Reject missing basis");
    Reject([&] { CreateGpuEndpointMix(gpu, basis, {}, parameters); }, "Reject missing location");
    Reject([&] { CreateGpuEndpointMix(gpu, basis, location, {}); }, "Reject missing parameters");
    Reject([&] { CreateGpuEndpointMix(gpu, basis, location, std::array{0.f}); }, "Reject odd parameter count");
    const std::vector<float> too_many(202, 0), large_basis(202, 1);
    Reject([&] { CreateGpuEndpointMix(gpu, large_basis, location, too_many); }, "Reject more than one hundred modes");
    for (float invalid : {-1.f, 1.01f, nan, infinity}) Reject([&] { CreateGpuEndpointMix(gpu, basis, std::array{0.f, invalid}, parameters); }, "Reject invalid location");
    for (float invalid : {-30.01f, 20.01f, nan, infinity}) Reject([&] { CreateGpuEndpointMix(gpu, basis, location, std::array{0.f, invalid}); }, "Reject invalid endpoint amplitude");
    Reject([&] { CreateGpuEndpointMix(gpu, std::array{1.f, nan}, location, parameters); }, "Reject nonfinite basis");
    const auto state = CreateGpuEndpointMix(gpu, basis, location, parameters);
    const auto wrong_adjoint = Upload<float>(gpu, std::array{1.f});
    Reject([&] { EncodeEndpointMixGradient(gpu, state, wrong_adjoint); }, "Reject mismatched sample adjoint extent");
}

void SpectralChainTest(Gpu &gpu, SpectralMagnitudeScale scale) {
    constexpr uint32_t modes = 4, frames = 1001, rate = 44100;
    auto random = MakeRandom(193);
    const auto basis = std::views::iota(0u, modes * frames) | std::views::transform([&](uint32_t) { return .12f * (Uniform(random) - .5f); }) | std::ranges::to<std::vector>();
    const auto location = std::views::iota(0u, frames) | std::views::transform([](uint32_t frame) { return float(frame) / (frames - 1); }) | std::ranges::to<std::vector>();
    const std::array initial{-.4f, .1f, -.3f, .2f, .3f, -.2f, .1f, -.1f};
    const std::array target_parameters{-.1, -.15, 0., -.05, .55, -.4, .35, -.35};
    const auto target_double = Reference(basis, location, target_parameters);
    const std::vector<float> target(target_double.begin(), target_double.end());
    const auto state = CreateGpuEndpointMix(gpu, basis, location, initial);
    const auto spectral = CreateSpectralLossGpu(gpu, target, rate, {.Scale = scale});
    const auto parameters = BufferSpan<float>(state.Parameters);
    const auto evaluate = [&] {
        BeginGpu(gpu);
        EncodeEndpointMix(gpu, state);
        EncodeSpectralLoss(gpu, spectral, state.Output);
        EncodeEndpointMixGradient(gpu, state, spectral.Gradient);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        return double(BufferSpan<float>(spectral.Loss)[0]);
    };
    const double initial_loss = evaluate();
    const std::vector<float> gradient(BufferSpan<float>(state.Gradient).begin(), BufferSpan<float>(state.Gradient).end());
    // Larger perturbations cross a near-zero real Nyquist bin in the log losses;
    // smaller ones lose precision when subtracting float loss values.
    const auto steps = scale == SpectralMagnitudeScale::Linear ? std::array{.003, .001} : std::array{.0005, .0003};
    double worst_difference = 0;
    for (uint32_t endpoint = 0; endpoint < 2; ++endpoint) {
        const auto direction = std::views::iota(0u, 2 * modes) | std::views::transform([&](uint32_t index) { return index / modes == endpoint ? double(Uniform(random) - .5f) : 0.; }) | std::ranges::to<std::vector>();
        for (double step : steps) {
            const auto perturbed = [&](double sign) { return std::views::iota(size_t(0), initial.size()) | std::views::transform([&](size_t index) { return float(initial[index] + sign * step * direction[index]); }) | std::ranges::to<std::vector>(); };
            const auto plus = perturbed(1), minus = perturbed(-1);
            std::ranges::copy(plus, parameters.begin());
            const double upper = evaluate();
            std::ranges::copy(minus, parameters.begin());
            const double lower = evaluate();
            double analytic = 0;
            for (size_t index = 0; index < initial.size(); ++index) analytic += gradient[index] * (double(plus[index]) - minus[index]) / (2 * step);
            const double numeric = (upper - lower) / (2 * step);
            const double relative = std::abs(analytic - numeric) / std::max(.002, std::abs(analytic));
            worst_difference = std::max(worst_difference, relative);
            Require(relative < .01, "Composed endpoint spectral loss versus quantization-aware central finite difference");
        }
    }
    std::ranges::copy(initial, parameters.begin());
    if (scale == SpectralMagnitudeScale::Linear) {
        const std::vector bounds(parameters.size(), std::array{double(EndpointMixMinimumLogAmplitude), double(EndpointMixMaximumLogAmplitude)});
        auto optimizer = CreateAdam(parameters, bounds);
        for (uint32_t step = 0; step < 50; ++step) {
            evaluate();
            UpdateAdam(optimizer, BufferSpan<float>(state.Gradient), parameters, .03);
        }
        const double fitted_loss = evaluate();
        Require(fitted_loss < .05 * initial_loss, "Shared Adam recovers endpoint amplitudes from full-time spectral loss");
        std::cout << "Endpoint synthetic fitting loss " << initial_loss << " -> " << fitted_loss << '\n';
    }
    std::cout << "Endpoint composed spectral scale " << uint32_t(scale) << ": worst directional error " << worst_difference << '\n';
}

void AdamTest() {
    const std::array initial{0.f, 0.f};
    const std::array bounds{std::array{-.01, .01}, std::array{-1., 1.}};
    auto state = CreateAdam(initial, bounds);
    auto parameters = initial;
    const std::array gradient{1.f, -2.f};
    Require(UpdateAdam(state, gradient, parameters, .1) == 1, "Shared Adam reports clipped coordinates");
    Require(state.Master[0] == -.01 && std::abs(state.Master[1] - .1 * 2 / (2 + 1e-8)) < 1e-15 && state.Step == 1, "Adam first update uses bias correction and epsilon");
    const auto saved = state.Master;
    Reject([&] { UpdateAdam(state, std::array{1.f, std::numeric_limits<float>::quiet_NaN()}, parameters, .1); }, "Reject nonfinite Adam gradient");
    Require(state.Master == saved && state.Step == 1, "Invalid Adam update does not partially change state");
    Reject([&] { CreateAdam(initial, std::array{std::array{.1, 1.}, std::array{-1., 1.}}); }, "Reject Adam initial value outside bounds");
}
} // namespace

int main() {
    try {
        AdamTest();
        auto gpu = CreateGpu();
        ValidationTest(gpu);
        EndpointLocationTest(gpu);
        FormulaTest(gpu, 1, 17, false);
        FormulaTest(gpu, 7, 777, false);
        FormulaTest(gpu, 3, 1093, true);
        FormulaTest(gpu, 100, 269, false);
        FormulaTest(gpu, 1, 65793, true);
        for (auto scale : {SpectralMagnitudeScale::Linear, SpectralMagnitudeScale::NaturalLog, SpectralMagnitudeScale::Decibels}) SpectralChainTest(gpu, scale);
        std::cout << "Endpoint mix tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
