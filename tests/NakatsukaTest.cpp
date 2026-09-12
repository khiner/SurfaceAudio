#include "nakatsuka/Nakatsuka.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <simd/simd.h>

using namespace surface_audio;
namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
}
int main() {
    try {
        namespace n = nakatsuka;
        constexpr double pi = std::numbers::pi;
        const n::Material material{1e9, .3, 1, 1e-6, .001, .002, 1e-5, 1e-5};
        const double expected = pi * pi * (1e6 + 250000) * std::sqrt(1e-9 / (12 * .91));
        Require(std::abs(n::AngularFrequency(material, .001, .002, 1, 1) / expected - 1) < 1e-14, "Simply supported plate equation");
        Require(std::abs(n::AngularFrequency(material, .0005, .001, 1, 1) / expected - 4) < 1e-13, "Velocity-squared frequency scaling");
        Require(n::AdhesionPotential(.1, 2, .2) == 2 && n::AdhesionPotential(.4, 2, .2) == 5, "Piecewise adhesion potential");
        const double r = .4, h = 1e-6, c = n::AdhesionPotential(r, 2, .2), gradient = (n::AdhesionPotential(r + h, 2, .2) - n::AdhesionPotential(r - h, 2, .2)) / (2 * h);
        Require(std::abs(n::AdhesionCorrection(r, 2, .2, false) + c / gradient) < 1e-9, "Printed PBD projection from numerical derivative");
        Require(n::AdhesionCorrection(r, 2, .2, true) < 0 && n::AdhesionCorrection(.1, 2, .2, true) == 0, "Shifted adhesion boundary");
        const auto patches = n::MakePatches(material, 1000, 42), repeated = n::MakePatches(material, 1000, 42);
        double mean = 0;
        for (size_t i = 0; i < patches.size(); ++i) {
            Require(patches[i].Width == repeated[i].Width, "Repeatable patch distribution");
            mean += patches[i].Width;
        }
        Require(std::abs(mean / patches.size() / .001 - 1) < .002, "Gaussian patch widths");
        constexpr uint32_t frames = 2048;
        const std::array<n::Patch, 1> patch{{{.001f, .002f}}};
        const n::AcousticSettings settings{.OddModes = 1};
        const std::vector<n::Sample> samples(frames, {.1f, 1e-9f, .5f, 1});
        const auto reference = n::Render(material, settings, patch, samples);
        const double omega = n::AngularFrequency(material, patch[0].Width, patch[0].Height, 1, 1) * std::pow(double(samples[0].Speed) / settings.ReferenceSpeed, 2);
        for (uint32_t i = 0; i < frames; ++i) {
            const double analytic = -settings.AirDensity * samples[0].Displacement * omega * omega / (4 * pi * .5) * std::cos(omega * (i + 1) / 44100 - omega * .5 / 343);
            Require(std::abs(reference[i] - analytic) < 1e-13, "Independent monopole phase and pressure");
        }
        auto gpu = CreateGpu();
        const auto actual = n::RenderGpu(gpu, material, settings, patch, samples);
        double error = 0, power = 0;
        for (size_t i = 0; i < frames; ++i) {
            error += std::pow(actual[i] - reference[i], 2);
            power += reference[i] * reference[i];
        }
        Require(std::sqrt(error / power) < .001, "Microrectangle GPU pressure");
        for (double factor : {1 - 1e-8, 1 + 1e-8}) {
            const n::Material boundary{material.Modulus * std::pow(factor * pi * settings.SampleRate / omega, 2), material.Poisson, material.ArealDensity, material.Thickness};
            const auto cpu = n::Render(boundary, settings, patch, samples);
            const auto rendered = n::RenderGpu(gpu, boundary, settings, patch, samples);
            double difference = 0, energy = 0;
            for (uint32_t i = 0; i < frames; ++i) {
                difference += std::pow(rendered[i] - cpu[i], 2);
                energy += cpu[i] * cpu[i];
            }
            Require(factor < 1 ? energy > 0 && std::sqrt(difference / energy) < .0001 : energy == 0 && difference == 0, "Extended-precision Nyquist boundary");
        }
        const n::Scene scene{.Columns = 4, .Rows = 4, .Speed = 0, .Gravity = 0};
        const auto rest = n::SimulateGpu(gpu, scene, 16);
        for (const auto &sample : rest.Samples) Require(sample.Speed < .002f && sample.Contact == 0, "Stationary separated sheet");
        const auto falling = n::SimulateGpu(gpu, {.Columns = 4, .Rows = 4, .Speed = 0}, 1);
        for (size_t i = 4; i < 16; ++i) Require(falling.FinalPositions[4 * i + 2] <= scene.InitialHeight, "Gravity displaces free vertices downward");
        const n::Scene small{.Columns = 4, .Rows = 4, .ShiftedAdhesion = 1, .SampleRate = 1000, .Speed = .03f};
        constexpr uint32_t steps = 80;
        const auto evolved = n::SimulateGpu(gpu, small, steps);
        std::array<simd_double3, 16> positions, velocity{}, predicted, anchors{};
        std::array<bool, 16> attached{};
        std::array<double, 16> weights;
        const double dt = 1 / double(small.SampleRate);
        for (size_t i = 0; i < 16; ++i) {
            positions[i] = {(double(i % 4) - 1.5) * small.Spacing, (double(i / 4) - 1.5) * small.Spacing, small.InitialHeight};
            weights[i] = i < 4 ? 0 : small.InverseMass;
        }
        const auto initial = positions;
        for (uint32_t frame = 0; frame < steps; ++frame) {
            for (size_t i = 0; i < 16; ++i) {
                predicted[i] = weights[i] == 0 ? initial[i] - simd_double3{0, small.Speed * (frame + 1) * dt, 0} : positions[i] + dt * (velocity[i] + simd_double3{0, 0, -small.Gravity * dt});
                if (weights[i] == 0) continue;
                const double distance = simd_length(predicted[i]) - small.SphereRadius;
                if (attached[i] && (simd_length(predicted[i] - anchors[i]) > 4 * small.AdhesionDistance || distance > 4 * small.AdhesionDistance)) attached[i] = false;
                if (!attached[i] && distance <= small.AdhesionDistance) {
                    anchors[i] = predicted[i] - distance * simd_normalize(predicted[i]);
                    attached[i] = true;
                }
            }
            for (uint32_t iteration = 0; iteration < small.Iterations; ++iteration) {
                const auto previous = predicted;
                for (int i = 0; i < 16; ++i) {
                    const std::array adjacent{i % 4 > 0 ? i - 1 : -1, i % 4 < 3 ? i + 1 : -1, i >= 4 ? i - 4 : -1, i < 12 ? i + 4 : -1};
                    simd_double3 correction{};
                    for (int j : adjacent)
                        if (j >= 0) {
                            const auto delta = previous[i] - previous[j];
                            const double distance = simd_length(delta), weight = weights[i] + weights[j];
                            if (distance > 0 && weight > 0) correction -= weights[i] / weight * (distance - small.Spacing) * delta / distance;
                        }
                    predicted[i] += .25 * small.Stretch * correction;
                    if (weights[i] > 0) {
                        if (attached[i]) {
                            const auto delta = predicted[i] - anchors[i];
                            const double r = simd_length(delta);
                            if (r > small.AdhesionDistance && r < 4 * small.AdhesionDistance)
                                predicted[i] += small.Adhesion * std::max(n::AdhesionCorrection(r, 1, small.AdhesionDistance, small.ShiftedAdhesion), -r) * delta / r;
                        }
                        const double distance = simd_length(predicted[i]) - small.SphereRadius;
                        const auto normal = simd_normalize(predicted[i]);
                        if (distance < 0) predicted[i] -= distance * normal;
                    }
                }
            }
            for (size_t i = 0; i < 16; ++i) {
                velocity[i] = (predicted[i] - positions[i]) / dt * std::exp(-small.Damping * dt);
                positions[i] = predicted[i];
            }
        }
        for (size_t i = 0; i < 16; ++i)
            for (size_t axis = 0; axis < 3; ++axis)
                Require(std::abs(positions[i][axis] - evolved.FinalPositions[4 * i + axis]) < 5e-5, "GPU PBD against double precision Jacobi constraints");
        const auto plane = [&](float adhesion) {
            return n::SimulateGpu(gpu, {.Columns = 2, .Rows = 2, .ShiftedAdhesion = 1, .SampleRate = 1000, .Speed = .002f, .Gravity = 0, .Adhesion = adhesion, .SphereRadius = 0, .InitialHeight = 0}, 250);
        };
        const auto bonded = plane(.5f), sliding = plane(0);
        Require(bonded.FinalPositions[9] > sliding.FinalPositions[9] + 1e-5, "Adhesion resists tangential sliding on a plane");
        const n::Scene delayed{.Columns = 2, .Rows = 2, .SampleRate = 1000, .Speed = .002f, .Gravity = 0, .SphereRadius = 0, .InitialHeight = 0, .MotionStart = .01f, .MotionDuration = .02f, .SettlingTime = .02f, .MotionRise = .015f, .MotionFall = .005f};
        const auto finite = n::SimulateGpu(gpu, delayed, 50);
        for (uint32_t i = 0; i < 9; ++i) Require(finite.Samples[i].Displacement == 0, "Settling precedes the delayed motion window");
        Require(std::abs(finite.FinalPositions[1] - (-.002f - .002f * .01f)) < 1e-8, "Driven motion stops at the declared duration");
        std::cout << "Nakatsuka plate, adhesion, radiation and GPU checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
