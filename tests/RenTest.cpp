#include "ren/Ren.h"
#include "Require.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

using namespace surface_audio;
using namespace surface_audio::ren;
namespace {
void Equations() {
    const Parameters p{.SampleRate = 1000, .Mass = 2, .RoughnessGain = .5, .RoughnessSpacing = .001, .Width = 2, .Height = 1, .TableSize = 4};
    const std::array normals{Normal{0, 0, 1}, Normal{-.5, 0, 1}};
    const std::array table{1.f, 2.f, -1.f, -2.f};
    State s;
    Contact c{.VelocityX = 1, .NormalVelocity = -.01, .NormalForce = 4, .MacroImpulse = 3, .Overlap = 1};
    auto a = Step(p, s, c, normals.data(), table.data());
    Require(a.Macro == 3 && a.Meso == 0 && std::abs(a.Micro - .002) < 1e-9, "Initial contact preserves supplied impulse and scales roughness force by dt");
    c.U = .6;
    c.MacroImpulse = 0;
    a = Step(p, s, c, normals.data(), table.data());
    Require(a.Meso == -1 && std::abs(a.Micro - .004) < 1e-9, "Normal-map transition follows the published old-minus-new momentum sign");
    a = Step(p, s, c, normals.data(), table.data());
    Require(a.Meso == 0, "Constant slope and speed produce no repeated meso impulse");
    c.U = -.75;
    a = Step(p, s, c, normals.data(), table.data());
    Require(a.Meso == 1 && s.Phase == 0, "UV and wavetable wrapping preserve momentum and phase");
    c.NormalVelocity = 0;
    c.MacroImpulse = 5;
    a = Step(p, s, c, normals.data(), table.data());
    Require(a.Total() == 0 && s.Lasting == 0, "Section 5.1 excludes non-approaching contacts");
    c.NormalVelocity = -1;
    c.VelocityX = 0;
    a = Step(p, s, c, normals.data(), table.data());
    Require(a.Macro == 5 && a.Micro == 0 && a.Meso == 0, "Transient contact uses macro impulse only");
    c.VelocityX = 1;
    c.U = .6;
    a = Step(p, s, c, normals.data(), table.data());
    Require(a.Meso == 0, "Contact resumption does not difference against stale momentum");
    c.Overlap = 0;
    Require(Step(p, s, c, normals.data(), table.data()).Total() == 0, "Separated objects are silent");
}
void Streaming() {
    constexpr unsigned frames = 8192, voices = 3;
    const Parameters p{.Width = 4, .Height = 2, .TableSize = 1024};
    std::array<Normal, 8> normals;
    for (unsigned i = 0; i < normals.size(); ++i) normals[i] = {float(.2 * std::sin(i)), float(.1 * std::cos(i)), 1};
    std::vector<float> table(p.TableSize);
    for (unsigned i = 0; i < table.size(); ++i) table[i] = float(std::sin(2 * std::numbers::pi * i / table.size()));
    std::vector<Contact> controls(frames * voices);
    for (unsigned v = 0; v < voices; ++v)
        for (unsigned n = 0; n < frames; ++n)
            controls[v * frames + n] = {.U = n * .00031f, .V = n * -.00013f, .VelocityX = .15f + v * .01f, .VelocityY = -.03f, .NormalVelocity = -.001, .NormalForce = .1f * (v + 1), .Overlap = unsigned(n % 317 > 9)};
    std::vector<Impulses> expected(controls.size()), split(frames);
    State full, chunks;
    Render(p, full, normals, table, std::span(controls).first(frames), std::span(expected).first(frames));
    for (unsigned offset = 0; offset < frames;) {
        const unsigned count = std::min(173u, frames - offset);
        Render(p, chunks, normals, table, std::span(controls).subspan(offset, count), std::span(split).subspan(offset, count));
        offset += count;
    }
    Require(full == chunks, "Block partition preserves complete source state");
    for (unsigned i = 0; i < frames; ++i) Require(expected[i].Total() == split[i].Total(), "Block partition preserves samples exactly");
    for (unsigned v = 1; v < voices; ++v) {
        State state;
        Render(p, state, normals, table, std::span(controls).subspan(v * frames, frames), std::span(expected).subspan(v * frames, frames));
    }
    auto gpu = CreateGpu();
    const auto source = CreateGpuSource(gpu, p, normals, table, voices, frames);
    std::ranges::copy(controls, BufferSpan<Contact>(source.Contacts).begin());
    BeginGpu(gpu);
    EncodeSource(gpu, source);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto actual = BufferSpan<Impulses>(source.Output);
    double square = 0, error = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        square += std::pow(expected[i].Total(), 2);
        error += std::pow(actual[i].Total() - expected[i].Total(), 2);
    }
    Require(std::sqrt(error / square) < 2e-5, "Complete CPU/Metal excitation agrees for multiple contacts");
    const auto state = BufferSpan<State>(source.States)[0];
    Require(std::abs(state.Phase - full.Phase) < .002 && std::abs(state.Momentum - full.Momentum) < 1e-8 && state.Lasting == full.Lasting, "CPU/Metal terminal state agrees");
}
void Validation() {
    const Parameters p{.Width = 1, .Height = 1, .TableSize = 2};
    const std::array normals{Normal{}};
    const std::array table{1.f, -1.f};
    const auto rejects = [](auto action) {
        try {
            action();
        } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    auto invalid = normals;
    invalid[0].Z = 0;
    Require(rejects([&] { Validate(p, invalid, table); }), "Reject tangent normals instead of clamping the equation denominator");
    State s;
    std::array c{Contact{.NormalForce = -1}};
    std::array<Impulses, 1> output{};
    Require(rejects([&] { Render(p, s, normals, table, c, output); }) && s == State{}, "Reject tensile load before changing state");
    c[0].NormalForce = std::numeric_limits<float>::quiet_NaN();
    Require(rejects([&] { ValidateContacts(c); }), "Reject nonfinite contact data");
}
}
int main() {
    try {
        Equations();
        Streaming();
        Validation();
        std::cout << "Ren tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
