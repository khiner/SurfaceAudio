#include "matusiak2024/Matusiak2024.h"
#include "matusiak2024/Matusiak2024Gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
using namespace surface_audio;
using namespace surface_audio::matusiak2024;
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Friction() {
    for (unsigned curve : {0u, 1u}) {
        FrictionParameters<double> p;
        p.Exponential = curve;
        for (double v : {-.19, .19}) {
            const double z = std::copysign(.0000062, v), ev = 1e-7, ez = 1e-11;
            const auto f = Evaluate(p, 1.6403, z, v);
            const double dv = (Evaluate(p, 1.6403, z, v + ev).Rate - Evaluate(p, 1.6403, z, v - ev).Rate) / (2 * ev);
            const double dz = (Evaluate(p, 1.6403, z + ez, v).Rate - Evaluate(p, 1.6403, z - ez, v).Rate) / (2 * ez);
            Check(std::abs(dv - f.RateVelocity) < 1e-8 && std::abs(dz - f.RateBristle) < 1e-3, "Independent friction Jacobian");
            Check(f.Damping == p.Damping && f.DampingVelocity == 0, "2024 constant damping");
            const auto elastic = Evaluate(p, 1.6403, 0., v);
            Check(elastic.Rate == v && elastic.Adhesion == 0, "Elastic regime");
            Check(std::abs(Evaluate(p, 1.6403, elastic.Steady, v).Rate) < 1e-14, "Steady slip");
        }
    }
    FrictionParameters<double> p;
    p.Damping = 100;
    Check(Evaluate(p, 1., .0001, 3.).Dissipation < 0, "2024 law does not have 2025 global passivity guarantee");
}
void Equations(NumericalConvention convention) {
    auto s = MakeString({.Convention = convention});
    const bool archive = convention == NumericalConvention::AuthorArchive2025;
    for (size_t j = 0; j < s.Z.size(); ++j) {
        const double position = s.Config.Length * s.Config.BowPosition + s.Config.BowWidth * (double(j) / (s.Z.size() - 1) - .5);
        double transverse{}, torsion{}, transverse_sum{}, torsion_sum{};
        for (size_t i = 0; i < s.U.size(); ++i) {
            const double weight = s.Interpolation[j * s.U.size() + i];
            transverse += weight * (i + 1) * s.Spacing;
            transverse_sum += weight;
        }
        for (size_t i = 0; i < s.W.size(); ++i) {
            const double weight = s.TorsionInterpolation[j * s.W.size() + i];
            torsion += weight * (i + 1) * s.TorsionSpacing;
            torsion_sum += weight;
        }
        Check(std::abs(transverse - position - (archive ? s.Spacing : 0)) < 1e-14 && std::abs(torsion - position - (archive ? s.TorsionSpacing : 0)) < 1e-14 && std::abs(transverse_sum - 1) < 1e-14 && std::abs(torsion_sum - 1) < 1e-14, "Convention interpolation coordinates and partition of unity");
    }
    for (size_t i = 0; i < s.U.size(); ++i) s.U[i] = s.PreviousU[i] = 1e-5 * std::sin(3.141592653589793 * (i + 1) / (s.U.size() + 1));
    for (size_t i = 0; i < s.W.size(); ++i) s.W[i] = s.PreviousW[i] = .0001 * std::sin(3.141592653589793 * (i + 1) / (s.W.size() + 1));
    double error{};
    for (unsigned i = 0; i < 22050; ++i) {
        const auto previous_hair = s.PreviousHair, previous_u = s.PreviousU, previous_w = s.PreviousW;
        const auto result = Step(s, .8722 * i / 44100, 2.3433);
        error = std::max(error, std::abs(result.EnergyError));
        for (size_t j = 0; j < s.Z.size(); ++j) {
            const double vh = (s.Hair[j] - previous_hair[j]) / (2 * s.Step);
            Check(std::abs(vh + s.HairStiffness / s.HairDamping * (s.Hair[j] + previous_hair[j]) / 2 + s.Force[j] / (s.Z.size() * s.HairDamping)) < 1e-12, "Massless hair Equation 23");
            double slip = vh - .8722 * i / 44100;
            for (size_t k = 0; k < s.U.size(); ++k) slip += s.Interpolation[j * s.U.size() + k] * (s.U[k] - previous_u[k]) / (2 * s.Step);
            for (size_t k = 0; k < s.W.size(); ++k) slip -= s.TorsionFeedback * s.Config.Radius * s.TorsionInterpolation[j * s.W.size() + k] * (s.W[k] - previous_w[k]) / (2 * s.Step);
            Check(std::abs(slip - s.Velocity[j]) < 1e-11, "Reconstructed slip with selected torsion feedback");
        }
    }
    std::cout << "Independent energy balance " << error << " J\n";
    Check(error < 1e-10, "Discrete full-system energy accounting");
}
void AuthorOscillation() {
    const Parameters defaults;
    const Parameters p{.MaterialDensity = defaults.Tension / (137.2 * 137.2 * std::numbers::pi * defaults.Radius * defaults.Radius), .Friction = {370600, .0082, .0513, .7727, 1.0902, .7, 0}, .Convention = NumericalConvention::AuthorArchive2025};
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() / "repros/matusiak2024/red.f32";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Check(bool(file), "Open published red waveform fixture");
    std::vector<float> rows(static_cast<size_t>(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(rows.data()), rows.size() * sizeof(float));
    Check(bool(file) && rows.size() % 3 == 0, "Read complete published waveform fixture");
    const size_t frames = rows.size() / 3, start = 3 * frames / 5;
    std::vector<double> waveform(frames);
    auto state = MakeString(p);
    for (size_t i = 1; i < frames; ++i) waveform[i] = Step(state, .3473 * i / p.SampleRate, .7992).BridgeForce;
    double reference_mean{}, native_mean{};
    for (size_t i = start; i < frames; ++i) {
        reference_mean += rows[3 * i + 1];
        native_mean += waveform[i];
    }
    reference_mean /= frames - start;
    native_mean /= frames - start;
    double reference_power{}, native_power{};
    for (size_t i = start; i < frames; ++i) {
        reference_power += std::pow(rows[3 * i + 1] - reference_mean, 2);
        native_power += std::pow(waveform[i] - native_mean, 2);
    }
    const auto band_power = [&](double low, double high) {
        double result{};
        for (double frequency = low; frequency <= high; frequency += 5) {
            double real{}, imaginary{};
            for (size_t i = start; i < frames; ++i) {
                const double angle = 2 * std::numbers::pi * frequency * i / p.SampleRate, value = waveform[i] - native_mean;
                real += value * std::cos(angle);
                imaginary += value * std::sin(angle);
            }
            result += real * real + imaginary * imaginary;
        }
        return result;
    };
    const double ratio = std::sqrt(native_power / reference_power);
    const double fundamental = band_power(175, 215);
    const double torsion = band_power(580, 625);
    std::cout << "Red author/native sustained AC RMS ratio " << ratio << ", 196/603 Hz band ratio " << fundamental / torsion << '\n';
    Check(ratio > .85 && ratio < 1.15 && fundamental > 4 * torsion, "Published red double-slip oscillation retains amplitude and pitch");
}
void GpuCheck(NumericalConvention convention) {
    const Parameters defaults;
    const bool archive = convention == NumericalConvention::AuthorArchive2025;
    const Parameters parameters{.MaterialDensity = archive ? defaults.Tension / (137.2 * 137.2 * std::numbers::pi * defaults.Radius * defaults.Radius) : defaults.MaterialDensity, .Convention = convention};
    auto gpu = CreateGpu();
    const std::array drives{BowDrive{}, archive ? BowDrive{2.3435f, .3439f, .8722f} : BowDrive{2.0791f, .5f, 1.0481f}};
    const unsigned frames = archive ? 17387 : 2646;
    const auto batch = RenderStringsGpu(gpu, parameters, drives, frames);
    for (size_t j = 0; j < drives.size(); ++j) {
        auto s = MakeString(parameters);
        double error{}, norm{}, energy_error{}, energy_norm{}, balance{};
        for (unsigned i = 1; i < frames; ++i) {
            const auto sample = Step(s, std::min(double(drives[j].Velocity), double(drives[j].Acceleration) * i / 44100), drives[j].NormalForce);
            const double delta = sample.BridgeForce - batch.BridgeForce[j * frames + i];
            error += delta * delta;
            norm += sample.BridgeForce * sample.BridgeForce;
            const double ed = batch.StoredEnergy[j * frames + i] - TotalEnergy(sample.Stored);
            energy_error += ed * ed;
            energy_norm += std::pow(TotalEnergy(sample.Stored), 2);
            balance = std::max(balance, std::abs(double(batch.EnergyError[j * frames + i])));
        }
        std::cout << "GPU voice " << j << " waveform relative L2 " << std::sqrt(error / norm) << '\n';
        Check(batch.FailedSteps[j] == 0 && std::sqrt(error / norm) < .003, "GPU waveform convergence over selected record");
        std::cout << "GPU energy relative L2 " << std::sqrt(energy_error / energy_norm) << ", balance " << balance << " J\n";
        Check(std::sqrt(energy_error / energy_norm) < .01 && balance < (archive ? 2e-5 : 1e-7), "GPU full energy trajectory and accounting");
        size_t offset = j * (2 * (s.U.size() + s.W.size()) + 5 * s.Z.size());
        for (const auto *v : {&s.U, &s.PreviousU, &s.W, &s.PreviousW, &s.Hair, &s.PreviousHair, &s.Z, &s.Velocity, &s.MidpointZ}) {
            double e{}, n{};
            for (double x : *v) {
                const double d = batch.FinalState[offset++] - x;
                e += d * d;
                n += x * x;
            }
            Check(std::sqrt(e / std::max(n, 1e-100)) < .1, "GPU complete final state");
        }
    }
}
}
int main() {
    try {
        Friction();
        AuthorOscillation();
        for (auto convention : {NumericalConvention::Paper2024, NumericalConvention::AuthorArchive2025}) {
            Equations(convention);
            GpuCheck(convention);
        }
        std::cout << "Matusiak2024Test passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
