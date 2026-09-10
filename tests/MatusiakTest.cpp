#include "matusiak/Matusiak.h"
#include "matusiak/MatusiakGpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
using namespace surface_audio;
using namespace surface_audio::matusiak;
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Friction() {
    FrictionParameters<double> p{100000, 100, .228, .5071, 1.0207, .7};
    for (int i = -100; i <= 100; ++i)
        for (int j = -100; j <= 100; ++j) {
            const double v = i * .03, z = j * 1.0207 * 1.6403 / (100 * p.Stiffness);
            const auto f = Evaluate(p, 1.6403, z, v);
            Check(f.Dissipation >= -1e-12, "Refined constitutive law passivity");
            Check(f.Adhesion >= 0 && f.Adhesion <= 1, "Adhesion bounds");
            Check(f.Damping <= p.Damping && f.Damping >= 0, "Refined damping bounds");
        }
    const double v = .19, z = .0000062, epsv = 1e-7, epsz = 1e-11;
    const auto f = Evaluate(p, 1.6403, z, v);
    const double dv = (Evaluate(p, 1.6403, z, v + epsv).Rate - Evaluate(p, 1.6403, z, v - epsv).Rate) / (2 * epsv);
    const double dz = (Evaluate(p, 1.6403, z + epsz, v).Rate - Evaluate(p, 1.6403, z - epsz, v).Rate) / (2 * epsz);
    Check(std::abs(dv - f.RateVelocity) < 1e-8 && std::abs(dz - f.RateBristle) < 1e-4, "Independent finite difference friction Jacobian");
    const auto stick = Evaluate(p, 1.6403, 0., .1);
    Check(stick.Rate == .1 && stick.Adhesion == 0, "Elastic regime");
    const auto steady = Evaluate(p, 1.6403, stick.Steady, .1);
    Check(std::abs(steady.Rate) < 1e-14, "Steady sliding equilibrium");
}
std::vector<double> Reference() {
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() / "repros/matusiak/Reference.f64";
    std::ifstream file(path, std::ios::binary);
    std::vector<double> reference(22050 * 7);
    Check(bool(file.read(reinterpret_cast<char *>(reference.data()), reference.size() * sizeof(double))), "Read actual Octave reference");
    auto s = MakeString();
    double maximum{}, norm{}, error{}, energy_error{}, residual{};
    const double ramp = std::ceil(.3439 / .8722 * 44100) - 1;
    for (size_t i = 1; i < 22050; ++i) {
        const auto before = StoredEnergy(s);
        const auto sample = Step(s, std::min(.3439, .3439 * i / ramp), 2.3433);
        const double delta = sample.BridgeForce - reference[i * 7 + 1];
        maximum = std::max(maximum, std::abs(delta));
        norm += reference[i * 7 + 1] * reference[i * 7 + 1];
        error += delta * delta;
        Check(std::abs(TotalEnergy(before) - reference[i * 7 + 5]) < 2e-10, "Author pre-step energy oracle");
        Check(std::abs(sample.RelativeVelocity - reference[i * 7 + 6]) < 2e-8, "Author relative velocity oracle");
        Check(sample.BristleDissipation >= -1e-12, "Discrete bristle passivity");
        Check(surface_audio::matusiak::TotalEnergy(sample.Stored) >= -1e-12, "Nonnegative stored energy");
        energy_error = std::max(energy_error, std::abs(sample.EnergyError));
        residual = std::max(residual, sample.Residual);
    }
    std::cout << "Author diagonal full-record: max " << maximum << " N, relative L2 " << std::sqrt(error / norm) << ", energy error " << energy_error << " J\n";
    Check(maximum < 2e-8 && std::sqrt(error / norm) < 1e-8, "Actual author full-record bridge oracle");
    Check(energy_error < 1e-10 && residual < 3e-13, "Independent full-system energy and nonlinear residual");
    return reference;
}
void GpuValidation() {
    Gpu unavailable;
    const std::array drives{BowDrive{}};
    const auto reject = [](auto &&operation) {
        try {
            operation();
        } catch (const std::invalid_argument &) { return; }
        throw std::runtime_error("Invalid GPU inputs were not rejected before allocation");
    };
    reject([&] { RenderStringsGpu(unavailable, Parameters{.HairStiffness = 1e100}, drives, 1); });
    reject([&] { RenderStringsGpu(unavailable, Parameters{.Friction = {.Damping = 1e-100}}, drives, 1); });
    const Parameters p{.SampleRate = 1e7};
    const auto state = MakeString(p);
    const size_t stride = 3 * (state.U.size() + state.W.size()) + 5 * state.Z.size();
    const std::vector<BowDrive> oversized(UINT32_MAX / stride + 1);
    reject([&] { RenderStringsGpu(unavailable, p, oversized, 1); });
}
void GpuChecks(const std::vector<double> &reference) {
    auto gpu = CreateGpu();
    constexpr unsigned frames = 4096;
    const std::array drives{BowDrive{}, BowDrive{1.17165f, .3439f, .8722f}};
    const auto batch = RenderStringsGpu(gpu, {}, drives, frames);
    double error{}, norm{}, maximum{};
    for (size_t i = 0; i < frames; ++i) {
        const double r = reference[i * 7 + 1], d = batch.BridgeForce[i] - r;
        error += d * d;
        norm += r * r;
        maximum = std::max(maximum, std::abs(d));
    }
    std::cout << "Distributed Metal FP32 first4096: max " << maximum << " N, relative L2 " << std::sqrt(error / norm) << ", residual " << batch.MaximumResidual[0] << ", failed " << batch.FailedSteps[0] << '\n';
    Check(std::sqrt(error / norm) < .003, "Distributed GPU FP32 vs independent author reference");
    for (auto f : batch.FailedSteps) Check(f == 0, "Distributed GPU Newton convergence");
    for (float x : batch.BridgeForce) Check(std::isfinite(x), "Finite GPU distributed trace");
    std::array<LumpedParameters<float>, 2> params{};
    params[1].NormalForce = .9f;
    const auto lumped = RenderLumpedGpu(gpu, params, frames);
    for (size_t voice = 0; voice < params.size(); ++voice) {
        LumpedState<float> cpu;
        double e{}, denom{};
        for (unsigned i = 0; i < frames; ++i) {
            const auto sample = StepLumped(params[voice], cpu, std::min(params[voice].BowVelocity, params[voice].Acceleration * i / params[voice].SampleRate), 1e-6f);
            const double delta = sample.Displacement - lumped.Displacement[voice * frames + i];
            e += delta * delta;
            denom += sample.Displacement * sample.Displacement;
        }
        Check(std::sqrt(e / denom) < .003, "Lumped CPU/GPU trajectories");
        Check(lumped.FailedSteps[voice] == 0, "Lumped GPU convergence");
    }
}
void Stability() {
    for (double rate : {22050., 44100., 96000.}) {
        const auto s = MakeString(Parameters{.SampleRate = rate});
        const double k = s.Step, h = s.Spacing, ht = s.TorsionSpacing;
        const double transverse = std::pow(s.WaveSpeed * k / h, 2) + 4 * s.Config.Damping1 * k / (h * h) + 4 * s.Bending / s.Density * k * k / std::pow(h, 4);
        Check(transverse <= 1, "Independent transverse Nyquist stability bound");
        Check(s.TorsionSpeed * k / ht <= 1, "Independent torsional Courant stability bound");
    }
}
void InitialEnergy() {
    auto state = MakeString();
    for (size_t i = 0; i < state.U.size(); ++i) state.U[i] = state.PreviousU[i] = 1e-5 * std::sin(3.141592653589793 * (i + 1) / (state.U.size() + 1));
    const double initial = TotalEnergy(StoredEnergy(state));
    Check(initial > 0, "Nonzero initial string energy");
    for (int i = 0; i < 1000; ++i) {
        const auto sample = Step(state, 0., 2.3433);
        Check(std::abs(sample.EnergyError) < 1e-12, "Unforced nonzero-initial-state energy balance");
        Check(sample.BristleDissipation >= -1e-13, "Unforced bristle passivity");
    }
}
void LumpedEnergy() {
    LumpedParameters<double> p;
    LumpedState<double> s;
    double max_energy{};
    for (int i = 0; i < 22050; ++i) {
        const auto a = StepLumped(p, s, std::min(p.BowVelocity, p.Acceleration * i / p.SampleRate), 1e-13);
        Check(a.Iterations < 100 && a.Residual < 1.1e-13, "Lumped nonlinear convergence");
        Check(a.BristleDissipation >= -1e-13, "Lumped discrete passivity");
        max_energy = std::max(max_energy, std::abs(a.EnergyError));
    }
    Check(max_energy < 1e-10, "Lumped discrete energy balance");
}
}
int main(int argc, char **argv) {
    try {
        const bool cpu_only = argc == 2 && std::string_view(argv[1]) == "--cpu-only";
        if (argc > 1 && !cpu_only) throw std::invalid_argument("Expected --cpu-only or no arguments");
        Friction();
        Stability();
        InitialEnergy();
        const auto reference = Reference();
        LumpedEnergy();
        GpuValidation();
        if (!cpu_only) GpuChecks(reference);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
