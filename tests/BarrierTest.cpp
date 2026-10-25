#include "barrier/Barrier.h"
#include "Require.h"

#include <array>
#include <iostream>
#include <numbers>

using namespace surface_audio;
using namespace surface_audio::barrier;
namespace {
void Mechanics() {
    for (bool implicit : {false, true}) {
        Controls c;
        c.Implicit = implicit;
        auto p = MakeParameters<double>(c);
        State<double> rest;
        for (unsigned n = 0; n < 100; ++n) {
            const auto a = Step(p, rest, Force<double>{});
            Require(a.Energy == 0 && a.FrictionForce == 0 && a.ContactLoss == 0, "Unforced rest remains finite and silent");
        }
        c.Kappa = 0;
        p = MakeParameters<double>(c);
        const double sigma = c.Damping / (2 * c.Mass), omega = std::sqrt(c.Stiffness / c.Mass - sigma * sigma);
        State<double> linear{.X = -.001, .DeltaX = -.001 * (1 - std::exp(sigma * p.Dt) * std::cos(omega * p.Dt))};
        for (unsigned n = 0; n < 4410; ++n) {
            const auto a = Step(p, linear, Force<double>{});
            const double exact = -.001 * std::exp(-sigma * n * p.Dt) * std::cos(omega * n * p.Dt);
            Require(std::abs(a.X - exact) < 1e-14, "Modal correction preserves the sampled linear oscillator");
        }
        c.Kappa = 1e9;
        c.Stiffness = c.Damping = 0;
        c.Friction = .2;
        p = MakeParameters<double>(c);
        State<double> impact{.X = -.0001, .DeltaX = .5 * p.Dt, .DeltaY = .2 * p.Dt};
        double last = .5 * c.Mass * (.5 * .5 + .2 * .2);
        bool touched = false;
        for (unsigned n = 0; n < 1000; ++n) {
            const auto a = Step(p, impact, Force<double>{});
            Require(a.Energy <= last + 1e-15 && a.ContactLoss >= 0 && a.Psi >= 0, "A free collision is passive");
            Require(std::abs(a.BalanceResidual) < 1e-15, "The collision closes the discrete energy balance");
            Require(a.TangentialNormalForce <= 0 && std::abs(a.FrictionForce) <= -c.Friction * a.TangentialNormalForce + 1e-12, "Contact friction obeys the Coulomb bound");
            touched |= a.X > c.Barrier;
            last = a.Energy;
        }
        Require(touched && impact.X < 0 && impact.DeltaX < 0, "The impact contacts and separates from the barrier");
        c = Controls{};
        c.Implicit = implicit;
        c.Barrier = .00002;
        c.Friction = .07;
        p = MakeParameters<double>(c);
        State<double> stick;
        unsigned stuck = 0, sliding = 0;
        for (unsigned n = 0; n < 2205; ++n) {
            const auto a = Step(p, stick, Force<double>{3, -.25 * std::cos(2 * std::numbers::pi * 100 * n * p.Dt)});
            Require(std::abs(a.BalanceResidual) < 1e-14 && a.ContactLoss >= 0, "Driven stick-slip closes the passive energy balance");
            if (n > 100) (a.Vy == 0 ? stuck : sliding)++;
        }
        Require(stuck > 100 && sliding > 100, "Subthreshold forcing sticks and sufficient forcing resumes slip");
        auto compressed = MakeState(p, c.Barrier + .00001, 0., 0., 0.);
        const double initial = Potential(p, compressed.X - c.Barrier) + p.Stiffness * compressed.X * compressed.X / 2;
        const auto release = Step(p, compressed, Force<double>{});
        Require(release.Energy <= initial + 1e-15 && std::abs(release.BalanceResidual) < 1e-15, "Preloaded contact initializes its stored energy consistently");
    }
    Controls invalid;
    invalid.Mass = 0;
    bool rejected = false;
    try { MakeParameters<float>(invalid); } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Reject zero mass");
}
void Streaming() {
    constexpr unsigned frames = 4096, voices = 4, block = 128;
    std::array<Parameters<float>, voices> parameters;
    std::vector<Force<float>> inputs(voices * frames);
    std::vector<Sample<float>> expected(inputs.size()), split(frames);
    for (unsigned v = 0; v < voices; ++v) {
        Controls c;
        c.Implicit = v & 1;
        c.Barrier = v >= 2 ? .00002 : 0;
        c.Friction = v >= 2 ? .07 : 1;
        parameters[v] = MakeParameters<float>(c);
        for (unsigned n = 0; n < frames; ++n) {
            const double t = n / c.SampleRate;
            inputs[v * frames + n] = v >= 2 ? Force<float>{3, float(-.25 * std::cos(2 * std::numbers::pi * 100 * t))} : Force<float>{float(-.1 * std::cos(1000 * t)), float(.1 * std::sin(1000 * t))};
        }
        State<float> full, chunks;
        const auto input = std::span(inputs).subspan(v * frames, frames);
        Render<float>(parameters[v], full, input, std::span(expected).subspan(v * frames, frames));
        for (unsigned n = 0; n < frames; n += block) Render<float>(parameters[v], chunks, input.subspan(n, block), std::span(split).subspan(n, block));
        Require(full == chunks, "CPU block boundaries preserve the complete state");
    }
    auto gpu = CreateGpu();
    auto source = CreateGpuSource(gpu, parameters, block);
    std::array<double, voices> error{}, norm{};
    for (unsigned n = 0; n < frames; n += block) {
        auto input = BufferSpan<Force<float>>(source.Input);
        for (unsigned v = 0; v < voices; ++v) std::copy_n(inputs.begin() + v * frames + n, block, input.begin() + v * block);
        BeginGpu(gpu);
        EncodeSource(gpu, source);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto output = BufferSpan<Sample<float>>(source.Output);
        for (unsigned v = 0; v < voices; ++v)
            for (unsigned j = 0; j < block; ++j) {
                const auto a = output[v * block + j], b = expected[v * frames + n + j];
                Require(std::isfinite(a.Energy) && a.ContactLoss >= 0 && a.Psi >= 0, "Metal preserves finite passive contact states");
                error[v] += std::pow(double(a.Vx - b.Vx), 2) + std::pow(double(a.Vy - b.Vy), 2);
                norm[v] += double(b.Vx) * b.Vx + double(b.Vy) * b.Vy;
            }
    }
    for (unsigned v = 0; v < voices; ++v) {
        std::cout << "voice " << v << " CPU/Metal velocity error " << std::sqrt(error[v] / norm[v]) << '\n';
        Require(std::sqrt(error[v] / norm[v]) < .002, "CPU and Metal agree across collisions and sticking intervals");
    }
}
}
int main() {
    try { Mechanics(); Streaming(); }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
