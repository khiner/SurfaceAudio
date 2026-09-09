#include "agarwal/ContactFit.h"
#include "core/GpuSpectral.h"
#include "core/Random.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace surface_audio;
using namespace surface_audio::agarwal;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void DecayEndpointRoundTripTest(Gpu &gpu) {
    constexpr uint32_t rate = 44100;
    const auto bounds = ContactFitLogDecayBounds();
    Require(double(bounds[0]) >= std::log(double(ContactFitMinimumDecay)) && double(bounds[1]) <= std::log(double(ContactFitMaximumDecay)), "Decay log bounds round inward");
    Require(double(std::nextafter(bounds[0], -std::numeric_limits<float>::infinity())) < std::log(double(ContactFitMinimumDecay)) && double(std::nextafter(bounds[1], std::numeric_limits<float>::infinity())) > std::log(double(ContactFitMaximumDecay)), "Decay log bounds are the closest inward floats");
    const std::array force{1.f};
    const std::array endpoints{ContactFitMinimumDecay, ContactFitMaximumDecay};
    for (uint32_t endpoint = 0; endpoint < endpoints.size(); ++endpoint) {
        const std::array modes{ContactFitMode{1000, endpoints[endpoint], .1f}};
        const auto state = CreateContactFitGpu(gpu, force, modes, rate, 17);
        const auto parameters = BufferSpan<float>(state.Parameters);
        Require(parameters[1] == bounds[endpoint], "Creation uses the optimizer's inward log endpoint");
        std::stringstream text;
        text << std::setprecision(10) << modes[0].Frequency << ' ' << std::exp(parameters[1]) << ' ' << std::exp(parameters[0]);
        ContactFitMode reloaded{};
        text >> reloaded.Frequency >> reloaded.Decay >> reloaded.Amplitude;
        Require(bool(text) && reloaded.Decay >= ContactFitMinimumDecay && reloaded.Decay <= ContactFitMaximumDecay, "Exported endpoint decay remains inside physical bounds");
        const std::array imported{reloaded};
        const auto restored = CreateContactFitGpu(gpu, force, imported, rate, 17);
        Require(BufferSpan<float>(restored.Parameters)[1] == parameters[1], "Text export and reimport preserve endpoint log decay");
    }
}

std::vector<double> Reference(std::span<const float> force, std::span<const ContactFitMode> modes, std::span<const double> parameters, uint32_t rate, uint32_t taps) {
    std::vector<double> impulse(taps, 0), output(force.size() + taps - 1, 0);
    for (uint32_t lag = 0; lag < taps; ++lag)
        for (size_t mode = 0; mode < modes.size(); ++mode)
            impulse[lag] += std::exp(parameters[mode] - double(lag) / (rate * std::exp(parameters[modes.size() + mode]))) * std::sin(2 * std::numbers::pi * modes[mode].Frequency * lag / rate);
    for (size_t frame = 0; frame < force.size(); ++frame)
        for (uint32_t lag = 0; lag < taps; ++lag) output[frame + lag] += force[frame] * impulse[lag];
    return output;
}

std::vector<double> ReferenceGradient(std::span<const float> force, std::span<const ContactFitMode> modes, std::span<const double> parameters, std::span<const float> adjoint, uint32_t rate, uint32_t taps) {
    std::vector<double> gradient(2 * modes.size(), 0);
    for (size_t mode = 0; mode < modes.size(); ++mode) {
        const double tau = std::exp(parameters[modes.size() + mode]);
        for (uint32_t lag = 0; lag < taps; ++lag) {
            const double derivative = std::exp(parameters[mode] - double(lag) / (rate * tau)) * std::sin(2 * std::numbers::pi * modes[mode].Frequency * lag / rate);
            double correlation = 0;
            for (size_t frame = 0; frame < force.size(); ++frame) correlation += force[frame] * double(adjoint[frame + lag]);
            gradient[mode] += correlation * derivative;
            gradient[modes.size() + mode] += correlation * derivative * lag / (rate * tau);
        }
    }
    return gradient;
}

void FiniteFirTest(Gpu &gpu, uint32_t force_frames, uint32_t taps, uint32_t mode_count, bool tail_adjoint) {
    constexpr uint32_t rate = 44100;
    auto random = MakeRandom(61);
    const auto modes = std::views::iota(0u, mode_count) | std::views::transform([](uint32_t mode) { return ContactFitMode{173.f + 191.f * mode, .003f + .002f * mode, .03f / (1 + mode)}; }) | std::ranges::to<std::vector>();
    const auto force = std::views::iota(0u, force_frames) | std::views::transform([&](uint32_t) { return .1f * (Uniform(random) - .5f); }) | std::ranges::to<std::vector>();
    const auto state = CreateContactFitGpu(gpu, force, modes, rate, taps);
    const auto parameters = BufferSpan<float>(state.Parameters);
    const std::vector<double> reference_parameters(parameters.begin(), parameters.end());
    const auto adjoint = std::views::iota(0u, state.Frames) | std::views::transform([&](uint32_t frame) { return tail_adjoint && frame < force_frames ? 0.f : Uniform(random) - .5f; }) | std::ranges::to<std::vector>();
    const auto sample_adjoint = Upload<float>(gpu, adjoint);
    BeginGpu(gpu);
    EncodeContactFit(gpu, state);
    EncodeContactFitGradient(gpu, state, sample_adjoint);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto expected = Reference(force, modes, reference_parameters, rate, taps);
    const auto expected_gradient = ReferenceGradient(force, modes, reference_parameters, adjoint, rate, taps);
    const auto output = BufferSpan<float>(state.Output), gradient = BufferSpan<float>(state.Gradient);
    double maximum_error = 0, maximum_gradient_error = 0;
    for (size_t index = 0; index < output.size(); ++index) maximum_error = std::max(maximum_error, std::abs(output[index] - expected[index]));
    for (size_t index = 0; index < gradient.size(); ++index) maximum_gradient_error = std::max(maximum_gradient_error, std::abs(gradient[index] - expected_gradient[index]) / std::max(.001, std::abs(expected_gradient[index])));
    Require(maximum_error < 1e-5, "Contact recurrence versus independent complete finite FIR");
    Require(maximum_gradient_error < .003, "Contact log-parameter adjoint versus complete finite FIR derivative");
    for (uint32_t trial = 0; trial < 2; ++trial) {
        const auto direction = std::views::iota(0u, uint32_t(gradient.size())) | std::views::transform([&](uint32_t index) { return (index / mode_count == trial) ? double(Uniform(random) - .5f) : 0.; }) | std::ranges::to<std::vector>();
        const double analytic = std::inner_product(gradient.begin(), gradient.end(), direction.begin(), 0.);
        for (double step : {1e-4, 3e-5}) {
            const auto perturbed = [&](double sign) { return std::views::iota(size_t(0), direction.size()) | std::views::transform([&](size_t index) { return reference_parameters[index] + sign * step * direction[index]; }) | std::ranges::to<std::vector>(); };
            const auto plus = Reference(force, modes, perturbed(1), rate, taps), minus = Reference(force, modes, perturbed(-1), rate, taps);
            double numeric = 0;
            for (size_t index = 0; index < expected.size(); ++index) numeric += adjoint[index] * (plus[index] - minus[index]) / (2 * step);
            Require(std::abs(analytic - numeric) < .003 * std::max(.001, std::abs(numeric)), "Contact parameter gradient versus central finite difference");
        }
    }
    const std::vector<float> saved_output(output.begin(), output.end()), saved_gradient(gradient.begin(), gradient.end());
    BeginGpu(gpu);
    EncodeContactFit(gpu, state);
    EncodeContactFitGradient(gpu, state, sample_adjoint);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    Require(std::equal(output.begin(), output.end(), saved_output.begin()) && std::equal(gradient.begin(), gradient.end(), saved_gradient.begin()), "Contact workspace reuse has no persistent recurrence history");
    if (taps == 1) Require(std::ranges::all_of(output, [](float value) { return value == 0; }) && std::ranges::all_of(gradient, [](float value) { return value == 0; }), "One-tap sine response and gradient are exactly zero");
    std::cout << "Contact " << mode_count << " modes, " << force_frames << "+" << taps << " frames, tail adjoint " << tail_adjoint << ": sample error " << maximum_error << ", gradient error " << maximum_gradient_error << '\n';
}

void LongTailTest(Gpu &gpu) {
    constexpr uint32_t rate = 44100, frames = 4099, taps = 11025;
    const std::array modes{ContactFitMode{80, .25f, .02f}, ContactFitMode{11917, .2f, .01f}};
    const auto force = std::views::iota(0u, frames) | std::views::transform([](uint32_t frame) { return frame == 0 ? .5f : frame == frames - 1 ? -.3f :
                                                                                                                                                 0.f; }) | std::ranges::to<std::vector>();
    const auto state = CreateContactFitGpu(gpu, force, modes, rate, taps);
    const auto parameters = BufferSpan<float>(state.Parameters);
    const auto adjoint = std::views::iota(0u, state.Frames) | std::views::transform([](uint32_t frame) { return frame >= frames ? std::sin(.017f * frame) : 0.f; }) | std::ranges::to<std::vector>();
    const auto input = Upload<float>(gpu, adjoint);
    BeginGpu(gpu);
    EncodeContactFit(gpu, state);
    EncodeContactFitGradient(gpu, state, input);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const std::vector<double> reference_parameters(parameters.begin(), parameters.end());
    const auto expected = Reference(force, modes, reference_parameters, rate, taps);
    const auto expected_gradient = ReferenceGradient(force, modes, reference_parameters, adjoint, rate, taps);
    const auto output = BufferSpan<float>(state.Output), gradient = BufferSpan<float>(state.Gradient);
    double error = 0, energy = 0, maximum = 0, gradient_error = 0;
    for (size_t index = 0; index < output.size(); ++index) {
        const double difference = output[index] - expected[index];
        error += difference * difference;
        energy += expected[index] * expected[index];
        maximum = std::max(maximum, std::abs(difference));
    }
    for (size_t index = 0; index < gradient.size(); ++index) gradient_error = std::max(gradient_error, std::abs(gradient[index] - expected_gradient[index]) / std::max(.001, std::abs(expected_gradient[index])));
    std::cout << "Contact long finite tail relative RMS " << std::sqrt(error / energy) << ", peak " << maximum << ", gradient error " << gradient_error << '\n';
    Require(std::sqrt(error / energy) < .001 && maximum < 1e-5 && gradient_error < .003, "Long contact response and cutoff derivative versus independent finite FIR");
}

void SpectralChainTest(Gpu &gpu) {
    constexpr uint32_t rate = 44100, frames = 1001, taps = 257;
    const std::array modes{ContactFitMode{357, .008f, .13f}, ContactFitMode{1337, .017f, .1f}, ContactFitMode{4721, .013f, .07f}};
    auto random = MakeRandom(71);
    const auto force = std::views::iota(0u, frames) | std::views::transform([&](uint32_t) { return Uniform(random) - .5f; }) | std::ranges::to<std::vector>();
    const auto contact = CreateContactFitGpu(gpu, force, modes, rate, taps);
    const auto parameters = BufferSpan<float>(contact.Parameters);
    const std::vector<float> initial(parameters.begin(), parameters.end());
    const auto target_parameters = std::views::iota(size_t(0), initial.size()) | std::views::transform([&](size_t index) { return initial[index] + (index < modes.size() ? .2 : .3); }) | std::ranges::to<std::vector>();
    const auto target_double = Reference(force, modes, target_parameters, rate, taps);
    const std::vector<float> target(target_double.begin(), target_double.end());
    const auto spectral = CreateSpectralLossGpu(gpu, target, rate, {.Scale = SpectralMagnitudeScale::Linear});
    const auto evaluate = [&] {
        BeginGpu(gpu);
        EncodeContactFit(gpu, contact);
        EncodeSpectralLoss(gpu, spectral, contact.Output);
        EncodeContactFitGradient(gpu, contact, spectral.Gradient);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        return double(BufferSpan<float>(spectral.Loss)[0]);
    };
    const double initial_loss = evaluate();
    const auto gradient = BufferSpan<float>(contact.Gradient);
    const std::vector<float> initial_gradient(gradient.begin(), gradient.end());
    const std::vector<double> initial_double(initial.begin(), initial.end());
    const auto oracle = ReferenceGradient(force, modes, initial_double, BufferSpan<float>(spectral.Gradient), rate, taps);
    for (size_t index = 0; index < gradient.size(); ++index) Require(std::abs(gradient[index] - oracle[index]) < .003 * std::max(.001, std::abs(oracle[index])), "Composed contact spectral gradient agrees with independent sample-adjoint contraction");
    bool decreased = false;
    for (double step : {.1, .03, .01}) {
        const double norm = std::sqrt(std::inner_product(initial_gradient.begin(), initial_gradient.end(), initial_gradient.begin(), 0.));
        for (size_t index = 0; index < parameters.size(); ++index) parameters[index] = float(initial[index] - step * initial_gradient[index] / norm);
        decreased |= evaluate() < initial_loss;
    }
    Require(decreased, "Full-time spectral loss decreases along contact analytic gradient");
}
} // namespace

int main() {
    try {
        auto gpu = CreateGpu();
        DecayEndpointRoundTripTest(gpu);
        FiniteFirTest(gpu, 733, 197, 5, false);
        FiniteFirTest(gpu, 73, 257, 3, true);
        FiniteFirTest(gpu, 193, 97, 50, false);
        FiniteFirTest(gpu, 17, 1, 2, false);
        LongTailTest(gpu);
        SpectralChainTest(gpu);
        std::cout << "Contact fit tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
