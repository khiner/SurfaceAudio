#include "traer/Traer.h"
#include "core/ErbNoise.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::traer;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void Equations() {
    std::array<Resonance, 1> mode{{{5, -6, 60}}};
    std::vector<double> output(1001);
    EvaluateResponse(mode, {}, 1001, 1000, {}, output);
    Require(std::abs(output[0] - std::pow(10., -.3)) < 1e-14, "Paper uses cosine with nonzero onset");
    Require(std::abs(output[1000] / output[0] - .001) < 1e-15, "Mode decays by60dB per second");
    std::array<Transient, 1> band{{{-20, 120}}};
    std::vector<float> noise(1001, 2);
    EvaluateResponse({}, band, 1001, 1000, noise, output);
    Require(std::abs(output[0] - .2) < 1e-14 && std::abs(output[500] - .0002) < 1e-14, "Transient Eq3");
    const double duration = SpringContactDuration(.001, 4000);
    Require(std::abs(SpringContactDuration(.004, 4000) / duration - 2) < 1e-14, "Spring duration scales with square root of mass");
    const auto force = ImpactForce(.01, 2, 10000);
    double area = 0;
    for (float x : force) {
        Require(x >= 0, "Half sine nonnegative");
        area += x / 10000.;
    }
    Require(force.front() == 0 && force.back() == 0 && std::abs(force[50] - 2) < 1e-7, "Half sine endpoints and peak");
    Require(std::abs(area - .04 / std::numbers::pi) < 2e-6, "Independent analytic half sine area");
    Require(ScrapeForce(3, 4, 2, .5, .25, 1) == 9.5, "Eq8 curvature plus shear");
    Require(ScrapeForce(-3, 0, 2, 0, .25, 2) == 9, "Integer shear exponents retain literal sign behavior");
    Require(ScrapeForce(3, 4, 0, .5, .25, .5) == 0, "Rest produces zero force");
    Require(std::isfinite(ScrapeForce(-3, 4, 2, .5, .25, .5)), "Signed fractional shear remains real");
    std::vector<double> profile(100), positions(3, .25), velocities(3, 2);
    for (size_t i = 0; i < profile.size(); ++i) profile[i] = std::pow(.01 * i, 2);
    const auto excitation = ScrapeExcitation(profile, .01, positions, velocities, .5, .25, 1);
    Require(std::abs(excitation[0] - 4.25) < 1e-6, "Parabolic surface derivatives and force");
}
void Quilting() {
    auto random = MakeRandom(19), repeat = random;
    const std::array rows{1., 1., 1., 1., 1., 1., 1., 1.};
    const auto quilted = QuiltProfile(rows, 4, 31, 2, random);
    Require(quilted.size() == 31 && std::ranges::all_of(quilted, [](double x) { return x == 1; }), "Quilting preserves constant surface");
    Require(quilted == QuiltProfile(rows, 4, 31, 2, repeat), "Quilting seed repeatability");
}
void Statistics() {
    const std::array mean{1., 2.};
    const std::array covariance{4., 3., 3., 9.};
    auto gaussian = MakeGaussian(mean, covariance);
    auto random = MakeRandom(2019);
    std::array<double, 2> sum{};
    double cross = 0;
    constexpr int count = 100000;
    for (int i = 0; i < count; ++i) {
        auto x = SampleGaussian(gaussian, random);
        sum[0] += x[0];
        sum[1] += x[1];
        cross += (x[0] - 1) * (x[1] - 2);
    }
    Require(std::abs(sum[0] / count - 1) < .025 && std::abs(sum[1] / count - 2) < .035, "Gaussian empirical means");
    Require(std::abs(cross / count - 3) < .1, "Gaussian cross covariance retained");
    const std::array modal_mean{100., 200., 300., -10., -20., -30., 60., 120., 180.};
    std::array<double, 81> zero{};
    auto deterministic = MakeGaussian(modal_mean, zero);
    auto modes = SampleModes(deterministic, 100, 44100, random);
    Require(modes[2].Frequency == 300 && modes[1].DecayDbPerSecond == 120, "Joint modal parameter layout");
    bool rejected = false;
    try {
        SampleModes(deterministic, 120, 44100, random, 2);
    } catch (const std::runtime_error &) { rejected = true; }
    Require(rejected, "Spacing outside10percent rejects entire sample");
    PerturbOnsets(modes, random);
    Require(modes[2].Frequency == 300 && modes[1].DecayDbPerSecond == 120 && modes[0].OnsetDb != -10, "Perturb onset only");
    rejected = false;
    try {
        MakeGaussian(mean, std::array{1., 2., 2., 1.});
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Indefinite covariance rejected");
}
void GpuComparison() {
    auto gpu = CreateGpu();
    constexpr uint32_t frames = 12001, voices = 3;
    std::vector<Resonance> modes(voices * PaperModeCount);
    std::vector<Transient> bands(voices * PaperNoiseBandCount);
    const auto single_noise = CreateErbNoise(gpu, PaperNoiseBandCount, frames, 44100, {.TapCount = 65});
    std::vector<float> noise(voices * single_noise.size());
    for (uint32_t voice = 0; voice < voices; ++voice) {
        std::copy(single_noise.begin(), single_noise.end(), noise.begin() + voice * single_noise.size());
        for (uint32_t m = 0; m < PaperModeCount; ++m)
            modes[voice * PaperModeCount + m] = {float(101 + 431.17 * m + 10 * voice), -10.f - m, 30.f + m, m == 0 ? 301u : 0xffffffffu};
        for (uint32_t b = 0; b < PaperNoiseBandCount; ++b) bands[voice * PaperNoiseBandCount + b] = {-30.f - b, 120.f + b};
    }
    auto response = CreateResponseGpu(gpu, frames, voices, 44100, modes, bands, noise);
    BeginGpu(gpu);
    EncodeResponse(gpu, response);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto result = BufferSpan<float>(response.Output);
    std::vector<double> reference(frames);
    double error = 0, energy = 0;
    for (uint32_t voice = 0; voice < voices; ++voice) {
        EvaluateResponse(std::span(modes).subspan(voice * PaperModeCount, PaperModeCount), std::span(bands).subspan(voice * PaperNoiseBandCount, PaperNoiseBandCount), frames, 44100, std::span(noise).subspan(voice * single_noise.size(), single_noise.size()), reference);
        for (uint32_t i = 0; i < frames; ++i) {
            error += std::pow(result[voice * frames + i] - reference[i], 2);
            energy += reference[i] * reference[i];
        }
    }
    Require(std::sqrt(error / energy) < 2e-4, "Complete3voice15mode30band GPU output agrees withFP64");
    const std::vector<float> first(result.begin(), result.end());
    BeginGpu(gpu);
    EncodeResponse(gpu, response);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    Require(std::equal(first.begin(), first.end(), result.begin()), "Stateless response repeatability");
    const std::array high_mode{Resonance{19001.137f, 0, 6}};
    constexpr uint32_t long_frames = 441001;
    std::vector<double> long_reference(long_frames);
    EvaluateResponse(high_mode, {}, long_frames, 44100, {}, long_reference);
    auto long_response = CreateResponseGpu(gpu, long_frames, 1, 44100, high_mode, {}, {});
    BeginGpu(gpu);
    EncodeResponse(gpu, long_response);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto long_output = BufferSpan<float>(long_response.Output);
    double late_error = 0, late_energy = 0, maximum_error = 0;
    for (uint32_t i = 0; i < long_frames; ++i) {
        const double delta = double(long_output[i]) - long_reference[i];
        maximum_error = std::max(maximum_error, std::abs(delta));
        if (i >= 9 * 44100) {
            late_error += delta * delta;
            late_energy += long_reference[i] * long_reference[i];
        }
    }
    Require(std::sqrt(late_error / late_energy) < 2e-5 && maximum_error < 2e-5, "19kHz10second late tail retains accurate phase");
    std::cout << "19kHz10second tail relative error " << std::sqrt(late_error / late_energy) << " maximum " << maximum_error << '\n';
    const std::array force{1.f, 2.f, -1.f, .5f};
    const std::array responses{1.f, 2.f, 0.f, -2.f, 1.f, .5f, .25f, -1.f, 3.f};
    const std::array nodes{0.f, .5f, 1.f};
    const std::array positions{-.1f, .25f, .75f, 1.2f};
    const auto spatial = ConvolveSpatialGpu(gpu, force, responses, 3, nodes, positions);
    for (size_t frame = 0; frame < spatial.size(); ++frame) {
        const double x = std::clamp(double(positions[std::min(frame, positions.size() - 1)]), 0., 1.);
        const size_t region = x <= .5 ? 0 : 1;
        const double mix = (x - .5 * region) * 2;
        double expected = 0;
        for (size_t tap = 0; tap < 3; ++tap)
            if (frame >= tap && frame - tap < force.size())
                expected += force[frame - tap] * std::lerp(double(responses[region * 3 + tap]), double(responses[(region + 1) * 3 + tap]), mix);
        Require(std::abs(spatial[frame] - expected) < 2e-5, "Full spatial convolution uses output location and held tail");
    }
    std::cout << DeviceName(gpu) << " full response relative error " << std::sqrt(error / energy) << '\n';
}
}
int main() {
    try {
        Equations();
        Statistics();
        Quilting();
        GpuComparison();
        std::cout << "Traer tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
