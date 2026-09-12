#include "agarwal/Impact.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
using namespace surface_audio;
using namespace surface_audio::agarwal;
namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
void Near(double a, double b, double tolerance, const char *message) { Require(std::abs(a - b) <= tolerance, message); }
void Equations() {
    Near(EquivalentStiffness(6, 3), 2, 1e-15, "Series stiffness");
    Near(EquivalentStiffness(6, 3, StiffnessCombination::Sum), 9, 1e-15, "Poster stiffness");
    Near(EquivalentStiffness(1e300, 1e300), 5e299, 1e285, "Stable series stiffness");
    const double plate = PlateStiffness(70e9, .2, .04, .005);
    Near(plate, 30381944.44444445, 1e-7, "Printed plate stiffness");
    Near(PublishedStiffness(ImpactMaterial::Glass, 0), 3.04e7, 0, "Tabulated stiffness");
    Require(std::abs(PlateStiffness(135e9, .2, .04, .005) / PublishedStiffness(ImpactMaterial::Metal, 0) - 1) > .1, "Table/formula discrepancy remains explicit");
    const ImpactSettings s{.Mass = .5, .Velocity = 2, .StiffnessA = 100, .StiffnessB = 100, .Scale = 3};
    const double duration = ImpactDuration(s);
    Near(duration, std::numbers::pi / 10, 1e-15, "Half-cycle duration");
    Near(ImpactForce(duration / 2, s), 3, 1e-15, "Momentum scales peak force");
    Near(ImpactForce(-1, s) + ImpactForce(duration, s) + ImpactForce(duration + 1, s), 0, 0, "Finite causal support");
    const ImpactSettings clipped{.Mass = .5, .Velocity = 2, .StiffnessA = 100, .StiffnessB = 100, .Scale = 3, .ForceLimit = .7};
    Near(ImpactForce(duration / 2, clipped), .7 * std::tanh(3 / .7), 1e-15, "Nonlinear force law");
}
void GpuChecks(Gpu &gpu) {
    constexpr double rate = 44100;
    std::array<ImpactSettings, 4> controls{{{.Mass = .05, .Velocity = 2}, {.Mass = .00001, .Velocity = 2}, {.Mass = .5, .Velocity = 2, .ForceLimit = .3}, {.Mass = .1, .Velocity = 0}}};
    constexpr uint32_t frames = 256;
    const auto point = RenderImpactForcesGpu(gpu, controls, frames, rate, ImpactSampling::Point);
    const auto average = RenderImpactForcesGpu(gpu, controls, frames, rate);
    for (uint32_t voice = 0; voice < controls.size(); ++voice) {
        const auto &s = controls[voice];
        const double duration = ImpactDuration(s), amplitude = s.Scale * s.Mass * s.Velocity;
        double integral = 0;
        for (uint32_t frame = 0; frame < frames; ++frame) {
            Near(point[voice * frames + frame], ImpactForce(frame / rate, s), 2e-6, "GPU point force against FP64 equations");
            Require(average[voice * frames + frame] >= -1e-7 && std::isfinite(average[voice * frames + frame]), "Finite nonnegative average force");
            integral += average[voice * frames + frame] / rate;
        }
        constexpr uint32_t steps = 100000;
        double reference = 0;
        for (uint32_t i = 0; i < steps; ++i) reference += ImpactForce((i + .5) * duration / steps, s) * duration / steps;
        Near(integral, reference, 1e-6 * std::max(reference, 1e-15), "Cell averages preserve continuous force integral");
        if (std::isinf(s.ForceLimit)) Near(integral, 2 * amplitude * duration / std::numbers::pi, 1e-6 * std::max(reference, 1e-15), "Analytic linear impulse");
    }
    Require(average[frames] > 0, "Subsample micro-impact survives cell averaging");
    const ImpactSettings micro{.Mass = .05, .Velocity = 2, .ForceLimit = .001};
    const auto pulse = RenderMicroImpactGpu(gpu, micro, rate);
    const auto linear = RenderMicroImpactGpu(gpu, ImpactSettings{.Mass = .05, .Velocity = 2}, rate);
    Require(pulse == linear, "Thesis micro-impacts omit nonlinear saturation");
    const std::array<float, 3> excitation{1, -.3f, .1f}, a{.5f, .2f, -.1f};
    const std::array<float, 2> b{.3f, -.1f};
    const auto mixed = MixContactForcesGpu(gpu, excitation, a, excitation, 2, .25);
    for (size_t i = 0; i < mixed.size(); ++i) Near(mixed[i], 1.25 * excitation[i] + 2 * a[i], 1e-7, "Signed contact components retain their independent weights");
    const auto actual = RenderContactGpu(gpu, excitation, micro, rate, a, b);
    std::vector<double> expected(excitation.size() + pulse.size() + a.size() - 2);
    for (size_t i = 0; i < excitation.size(); ++i)
        for (size_t j = 0; j < pulse.size(); ++j)
            for (size_t k = 0; k < a.size(); ++k) expected[i + j + k] += double(excitation[i]) * pulse[j] * (double(a[k]) + (k < b.size() ? b[k] : 0));
    Require(actual.size() == expected.size(), "Complete micro-impact and response tails");
    for (size_t i = 0; i < actual.size(); ++i) Near(actual[i], expected[i], 2e-7, "Independent triple convolution");
    bool rejected = false;
    try {
        RenderImpactForcesGpu(gpu, controls, 1, rate);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Incomplete pulse batches are rejected");
}
}
int main() {
    try {
        Equations();
        auto gpu = CreateGpu();
        GpuChecks(gpu);
        std::cout << "Agarwal impact and micro-impact checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
