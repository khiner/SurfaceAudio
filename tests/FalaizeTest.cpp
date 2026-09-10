#include "falaize/Falaize.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
using namespace surface_audio;
using namespace surface_audio::falaize;
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Constitutive() {
    Parameters<double> p;
    p.SampleRate = 1e100;
    bool rejected = false;
    try {
        MakeFloatModel(p);
    } catch (const std::invalid_argument &) { rejected = true; }
    Check(rejected, "Reject GPU coefficient overflow");
    p = {};
    for (const double v : {-.3, -.06, .01, .06, .3})
        for (const double z : {-6e-5, -3e-5, 0., 3e-5, 6e-5}) {
            const auto f = Evaluate(p, z, z, v);
            const double plastic = f.Adhesion * std::abs(v) / (p.NormalForce * (p.Dynamic + (p.Static - p.Dynamic) * std::exp(-v * v / (p.StribeckVelocity * p.StribeckVelocity)))) * p.Stiffness * z;
            Check(std::abs(f.Rate - (v - plastic)) < 1e-14, "Paper Eq10 bristle rate");
            Check(std::abs(f.Force - (p.Stiffness * z + p.ComplianceDamping * f.Rate + p.FluidDamping * v)) < 1e-14, "Paper Eq11 force");
            Check(std::abs(f.Dissipation - ((p.ComplianceDamping + p.FluidDamping) * v * v - p.ComplianceDamping * plastic * v + plastic * p.Stiffness * z)) < 1e-14, "Independent Eq17 quadratic power");
            const double dv = 1e-7, dz = 1e-11;
            Check(std::abs(f.ResistanceVelocity - (Evaluate(p, z, z, v + dv).Resistance - Evaluate(p, z, z, v - dv).Resistance) / (2 * dv)) < 1e-6, "Resistance velocity derivative");
            Check(std::abs(f.ResistanceState - (Evaluate(p, z + dz, z, v).Resistance - Evaluate(p, z - dz, z, v).Resistance) / (2 * dz)) < .01, "Resistance state derivative");
        }
    p.ComplianceDamping = 100;
    const auto active = Evaluate(p, .00005, .00005, 1.);
    Check(active.Dissipation < 0 && active.Determinant < 0, "Literal 2024 law admits negative dissipation");
    p.ComplianceDamping = 0;
    for (int i = -100; i <= 100; ++i)
        for (int j = -50; j <= 50; ++j) Check(Evaluate(p, j * 1e-5, j * 1e-5, i * .1).Dissipation >= -1e-12, "Zero compliance damping passive domain");
    p = {};
    for (double q0 : {-0.001, 0., .001, .015, .03})
        for (double dq : {-1e-3, -1e-7, 1e-12, 1e-7, 1e-3}) {
            const double q1 = q0 + dq;
            Check(std::abs(HammerGradient(p, q0, q1) * dq - (HammerEnergy(p, q1) - HammerEnergy(p, q0))) < 2e-16, "Independent discrete hammer chain rule");
        }
}
void String() {
    Parameters<double> p;
    p.SampleRate = 48000;
    p.StringDamping = 0;
    for (const unsigned elements : {0u, 20u}) {
        p.Elements = elements;
        const auto m = MakeModel(p);
        State<double> s;
        s.Displacement[0] = .001;
        const double initial = Energy(m, s, false), dt = 1 / p.SampleRate;
        double error{};
        for (unsigned n = 0; n < 48000; ++n) {
            const auto a = StepString(m, s);
            error = std::max(error, std::abs(a.Energy + s.Loss - initial));
            Check(a.Failed == 0, "Nonzero-state unforced solver convergence");
        }
        Check(error < 2e-12, "Nonzero initial energy accounting");
        const double angle = std::numbers::pi / (elements ? elements : 1);
        if (elements) Check(std::abs(m.OmegaSquared[0] - p.Tension / p.Density * (2 - 2 * std::cos(angle)) / (std::pow(p.Length / elements, 2) / 3 * (2 + std::cos(angle)))) < 1e-8, "Consistent FEM spectrum");
        const double inv = 1 / (1 + dt * dt / 4 * m.OmegaSquared[0]);
        double q = .001, v = 0;
        for (unsigned n = 0; n < 48000; ++n) {
            const double vm = (v - dt / 2 * m.OmegaSquared[0] * q) * inv;
            q += dt * vm;
            v = 2 * vm - v;
        }
        const double phase = 48000 * 2 * std::atan(dt * std::sqrt(m.OmegaSquared[0]) / 2);
        Check(std::abs(q - .001 * std::cos(phase)) < 1e-13 && std::abs(s.Displacement[0] - q) < 1e-13, "Independent midpoint modal dispersion");
    }
}
void Trajectories() {
    for (unsigned elements : {0u, 20u})
        for (double theta : {0., .5})
            for (bool hammer : {false, true}) {
                Parameters<double> p;
                p.Elements = elements;
                p.Theta = theta;
                const auto m = MakeModel(p);
                State<double> s;
                if (hammer) s.HammerVelocity = 1;
                const double initial = Energy(m, s, hammer);
                double error{}, residual{};
                for (unsigned i = 0; i < (hammer ? 9600 : 96000); ++i) {
                    const auto a = Step(m, s, .1, hammer, 1e-12);
                    error = std::max(error, std::abs(a.Energy + s.Loss - initial));
                    residual = std::max(residual, a.Residual);
                    Check(!a.Failed, "Paper trajectory convergence");
                    Check(a.Energy >= 0, "Stored energy nonnegative");
                }
                std::cout << "elements=" << elements << " theta=" << theta << " hammer=" << hammer << " energy=" << error << " residual=" << residual << '\n';
                Check(error < 1e-10, "Independent cumulative energy balance");
            }
}
void Oracle() {
    for (unsigned variant : {0u, 1u, 2u}) {
        const bool hammer = variant != 0;
        const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() / "repros/falaize" / (variant == 2 ? "hammer_figure_oracle.f64" : hammer ? "hammer_modal_oracle.f64" :
                                                                                                                                                                 "bow_modal_oracle.f64");
        std::ifstream file(path, std::ios::binary);
        Check(bool(file), "Read independent equation fixture");
        Parameters<double> p;
        if (variant == 2) p.HammerStiffness /= p.HammerThickness;
        const auto m = MakeModel(p);
        State<double> s;
        if (hammer) s.HammerVelocity = 1;
        double error[4]{}, norm[4]{};
        for (unsigned i = 0; i < 4096; ++i) {
            double reference[4];
            Check(bool(file.read(reinterpret_cast<char *>(reference), sizeof(reference))), "Complete equation fixture");
            const auto a = Step(m, s, .1, hammer, 1e-12);
            const double native[]{a.Displacement, a.Velocity, a.Force, s.Elastic};
            for (unsigned j = 0; j < 4; ++j) {
                const double d = native[j] - reference[j];
                error[j] += d * d;
                norm[j] += reference[j] * reference[j];
            }
        }
        for (unsigned j = 0; j < 4; ++j) Check(std::sqrt(error[j] / norm[j]) < 1e-7, "Independent full coupled numerical Jacobian oracle");
    }
}
void GpuComparison() {
    auto gpu = CreateGpu();
    for (bool hammer : {false, true}) {
        Parameters<double> p;
        const std::vector<float> drives{hammer ? 1.f : .1f, hammer ? .5f : .07f};
        constexpr unsigned frames = 9600;
        const auto batch = RenderGpu(gpu, p, drives, frames, hammer);
        double error{}, norm{}, state_error{}, state_norm{};
        for (unsigned voice = 0; voice < drives.size(); ++voice) {
            const auto m = MakeModel(p);
            State<double> s;
            if (hammer) s.HammerVelocity = drives[voice];
            for (unsigned i = 0; i < frames; ++i) {
                const auto a = Step(m, s, double(drives[voice]), hammer, 1e-12);
                const double d = batch.Velocity[voice * frames + i] - a.Velocity;
                error += d * d;
                norm += a.Velocity * a.Velocity;
            }
            unsigned offset = voice * (2 * p.Modes + 4);
            const auto compare = [&](double value) {
                const double d = batch.FinalState[offset++] - value;
                state_error += d * d;
                state_norm += value * value;
            };
            for (unsigned i = 0; i < p.Modes; ++i) compare(s.Displacement[i]);
            for (unsigned i = 0; i < p.Modes; ++i) compare(s.Velocity[i]);
            for (const double v : {s.Elastic, s.HammerVelocity, s.ContactVelocity, s.Force}) compare(v);
            Check(!batch.FailedSteps[voice], "GPU convergence");
        }
        std::cout << "gpu hammer=" << hammer << " wave_l2=" << std::sqrt(error / norm) << " state_l2=" << std::sqrt(state_error / state_norm) << '\n';
        Check(std::sqrt(error / norm) < .003, "Full waveform FP64 versus Metal");
        Check(std::sqrt(state_error / state_norm) < .003, "Full final state FP64 versus Metal");
    }
}
}
int main() {
    try {
        Constitutive();
        String();
        Trajectories();
        Oracle();
        GpuComparison();
        std::cout << "Falaize tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
